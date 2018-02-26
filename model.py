from __future__ import division

import os
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from gammon.game import Game
from gammon.agents.human_agent import HumanAgent
from gammon.agents.td_gammon_agent import TDAgent
from gammon.agents.heuristic_agent import HeuristicAgent

# Is GPU available flag
use_cuda = torch.cuda.is_available()  # @UndefinedVariable
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor  # @UndefinedVariable
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor  # @UndefinedVariable
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor  # @UndefinedVariable
Tensor = FloatTensor


class Model(nn.Module):
    # Checkpoint file name
    checkpoint_file_name = 'checkpoint.pt'

    # Learning rate
    learning_rate = .0001

    # Min (x, r, x_next) history size
    min_train_history_size = 300

    # Max (x, r, x_next) history size
    max_train_history_size = 3000000

    # Train per steps
    train_steps = 4

    # Train minibatch size
    minibatch_size = 32

    def __init__(self, summaries_path, checkpoints_path, restore=False):
        super(Model, self).__init__()
        self.summary_path = summaries_path
        self.checkpoint_path = checkpoints_path

        # describe network size
        layer_size_hidden = 80
        layer_size_output = 1

        # first convolution layer
        self.c1 = nn.Conv1d(4*len(Game.PLAYERS), 16, kernel_size=6, stride=1)
#         self.bn1 = nn.BatchNorm2d(16)

        # second convolution layer
        self.c2 = nn.Conv1d(16, 32, kernel_size=4, stride=1)
#         self.bn2 = nn.BatchNorm2d(32)

        # fully connected layer
        self.fc = nn.Linear(512, layer_size_hidden)

        # output layer
        self.v = nn.Linear(layer_size_hidden, layer_size_output)

        self.optimizer = optim.Adam(self.parameters(), self.learning_rate)

        # after training a model, we can restore checkpoints here
        if restore:
            self.restore()

        if use_cuda:
            self.cuda()

    def forward(self, x):
#         x = F.relu(self.bn1(self.c1(x)))
#         x = F.relu(self.bn2(self.c2(x)))
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = x.view(-1, 512)
        x = F.relu(self.fc(x))
        x = F.sigmoid(self.v(x))
        return x

    def update(self, x_batch, v_next_batch):
        # values predicted by the network
        state_values = \
            self(Variable(torch.from_numpy(x_batch)).type(FloatTensor))  # @UndefinedVariable

        # values expected
        expected_state_values = \
            Variable(torch.from_numpy(v_next_batch)).type(FloatTensor)  # @UndefinedVariable

        # compute Huber loss
        loss = F.smooth_l1_loss(state_values, expected_state_values)

        # update the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, episode):
        checkpoint = {'episode': episode, 'state_dict': self.state_dict()}
        torch.save(checkpoint, os.path.join(self.checkpoint_path, self.checkpoint_file_name))

    def restore(self):
        checkpoint = torch.load(os.path.join(self.checkpoint_path, self.checkpoint_file_name),
                                map_location=lambda storage, _loc: storage)
        self.load_state_dict(checkpoint['state_dict'])
        print('Restored checkpoint, train episodes: {0}'.format(checkpoint['episode']))

    def extract_features(self, game, player):
        features = []
        for p in game.players:
            features.extend(self.extract_player_features(game, p, player == p))
        return features

    def extract_player_features(self, game, player, player_move):
        f = []
        f1 = []
        f2 = []
        n = game.num_pieces[player]
        for col in game.grid:
            if col and col[0] == player:
                f1.append(1.)
                f2.append(float(len(col))/n)
            else:
                f1.append(0.)
                f2.append(0.)
        f.append(f1)
        f.append(f2)
        f.append([float(len(game.off_pieces[player]))/n]*len(game.grid))
        f.append([1. if player_move else 0.]*len(game.grid))
        return f

    def get_output(self, x):
        x = torch.from_numpy(np.array(x))  # @UndefinedVariable
        return self(Variable(x).type(FloatTensor)).data.cpu().numpy()

    def play(self):
        game = Game.new()
        game.play([HumanAgent(Game.PLAYERS[0]), TDAgent(Game.PLAYERS[1], self)], draw=True)

    def test(self, episodes=100, draw=False, full_stats=True):
        player_agents = [TDAgent(Game.PLAYERS[0], self), HeuristicAgent(Game.PLAYERS[1])]
        num_steps = 0
        winners = [0, 0]
        for episode in range(1, episodes+1):
            game = Game.new()

            winner = game.play(player_agents, draw=draw)
            winners[winner] += 1
            num_steps += game.num_steps

            winners_total = sum(winners)

            win_rate = (winners[0]/winners_total)*100.

            if full_stats:
                print("[Test %d/%d] %s (%s) vs %s (%s), steps: %d, wins: %d:%d (%.1f%%)" %
                      (episode, episodes,
                       player_agents[0].name, player_agents[0].player,
                       player_agents[1].name, player_agents[1].player,
                       game.num_steps, winners[0], winners[1], win_rate))

        print("\nPlayed %d test games %s (%s) vs %s (%s), mean steps: %.1f, "
              "wins: %d/%d, win ratio: %.1f%%" %
              (winners_total, player_agents[0].name, player_agents[0].player,
               player_agents[1].name, player_agents[1].player,
               num_steps/winners_total, winners[0], winners[1], win_rate))

        return win_rate

    def train(self, episodes=5000, test_interval=1000, test_episodes=100,
              checkpoint_interval=1000, summary_interval=1000, summary_name=None):
        print("Training started.\n")

        summary_tstamp = str(int(time.time()))
        summary_name = summary_tstamp+'-'+summary_name if summary_name else summary_tstamp
        summary_writer = SummaryWriter(os.path.join(self.summary_path, summary_name))

        # the agent plays against itself, making the best move for each player
        player_agents = [TDAgent(Game.PLAYERS[0], self), TDAgent(Game.PLAYERS[1], self)]

        # (x, r, x_next) train history for experience replay
        train_history = []

        win_rate = 0.

        for episode in range(1, episodes+1):
            game = Game.new()
            player_index = random.randint(0, 1)

            global_step = 0

            x_next = self.extract_features(game, player_agents[player_index].player)

            while not game.is_over():
                x = x_next

                game.next_step(player_agents[player_index])
                global_step += 1
                player_index = (player_index+1) % 2

                if game.is_over():
                    x_next = None
                    r = float(game.winner())
                else:
                    x_next = self.extract_features(game, player_agents[player_index].player)
                    r = 0

                train_history.append([x, r, x_next])

                # trim (x, r, x_next) train_history
                if len(train_history) > self.max_train_history_size:
                    del train_history[:(len(train_history)-self.max_train_history_size)]

                # update network every train_steps if enough train history data is obtained
                if len(train_history) >= self.min_train_history_size and \
                        (global_step % self.train_steps == 0 or
                            (episode == episodes and game.is_over())):
                    train_minibatch = np.array(random.sample(train_history,
                                                             self.minibatch_size*self.train_steps))

                    # get boolean mask array of terminal states
                    states_terminal = np.zeros(train_minibatch.shape[0], dtype=bool)
                    states_terminal[np.where(np.equal(train_minibatch[:, 2], None))] = True
                    states_non_terminal = ~states_terminal

                    # array of minibatch state rewards
                    r_batch = train_minibatch[:, 1]

                    # array of v_next_batch values to compute
                    v_next_batch = np.zeros(train_minibatch.shape[0], dtype=np.float32)

                    # for terminal states: v_next_batch = r_batch
                    v_next_batch[states_terminal] = r_batch[states_terminal]

                    # for non-terminal states: v_next_batch = r_batch + get_output(x_next)
                    v_next_batch[states_non_terminal] = r_batch[states_non_terminal] + \
                        self.get_output(list(train_minibatch[states_non_terminal, 2])).flatten()

                    x_batch = train_minibatch[:, 0]

                    # reshape v_next to (32, 1) shape
                    v_next_batch = v_next_batch.reshape(-1, 1)

                    # update network by minibatch
                    self.update(np.array(list(x_batch)), v_next_batch)

            winner = game.winner()

            print("[Train %d/%d] (winner: '%s') in %d turns" % (episode, episodes,
                                                                player_agents[winner].player,
                                                                game.num_steps))

            # play test games every test_interval
            if episode % test_interval == 0 or episode == episodes:
                win_rate = self.test(episodes=test_episodes, full_stats=False)
            # write summary every summary_interval
            if episode % summary_interval == 0 or episode == episodes:
                summary_writer.add_scalar('game/win_rate', win_rate, global_step=episode)
            # save checkpoint every checkpoint_interval
            if episode % checkpoint_interval == 0 or episode == episodes:
                self.save(episode)

        summary_writer.close()

        print("\nTraining completed.")
