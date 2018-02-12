from __future__ import division

import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import nn

from gammon.game import Game
from gammon.agents.human_agent import HumanAgent
from gammon.agents.td_gammon_agent import TDAgent
from gammon.agents.heuristic_agent import HeuristicAgent


# helper to create a dense, fully-connected layer
def dense_layer(x, size, activation, name):
    return layers.fully_connected(x, size, activation_fn=activation, scope=name)


class Model(object):
    # Learning rate
    learning_rate = .0001

    # Min (x, r, x_next) history size
    min_train_history_size = 300

    # Max (x, r, x_next) history size
    max_train_history_size = 3000000

    # Train minibatch size
    minibatch_size = 32

    def __init__(self, sess, model_path, summary_path, checkpoint_path, restore=False):
        self.model_path = model_path
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path

        # setup our session
        self.sess = sess
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # describe network size
        layer_size_input = 100
        layer_size_hidden = 64
        layer_size_output = 1

        # placeholders for input and target output
        self.x = tf.placeholder('float', [None, layer_size_input], name='x')
        self.V_next = tf.placeholder('float', [None, layer_size_output], name='V_next')

        # build network arch
        hidden = dense_layer(self.x, layer_size_hidden, tf.sigmoid, name='hidden')
        self.V = dense_layer(hidden, layer_size_output, tf.sigmoid, name='V')

        # watch the individual value predictions over time
        tf.summary.scalar('V_next', tf.reduce_sum(self.V_next))
        tf.summary.scalar('V', tf.reduce_sum(self.V))

        # mean squared error of the difference between the next state and the current state
        loss_op = tf.reduce_mean(tf.squared_difference(self.V_next, self.V), name='loss')

        # track the number of steps and average loss for the current game
        with tf.variable_scope('game'):
            loss_ema = tf.train.ExponentialMovingAverage(decay=0.999)

            loss_avg_op = loss_ema.apply([loss_op])

            tf.summary.scalar('game/loss_ema', loss_ema.average(loss_op))

        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # as part of training we want to update our step and other monitoring variables
        with tf.control_dependencies([
            loss_avg_op,
        ]):
            # define single operation to apply all gradient updates
            self.train_op = optimizer.minimize(loss_op, global_step=global_step, name='train')

        # merge summaries for TensorBoard
        self.summaries_op = tf.summary.merge_all()

        # create a saver for periodic checkpoints
        self.saver = tf.train.Saver(max_to_keep=1)

        # run variable initializers
        self.sess.run(tf.global_variables_initializer())

        # after training a model, we can restore checkpoints here
        if restore:
            self.restore()

    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.saver.restore(self.sess, latest_checkpoint_path)

    def extract_features(self, game, player):
        features = []
        for p in game.players:
            for col in game.grid:
                if col and col[0] == p:
                    features += [1., float(len(col))/game.num_pieces[p]]
                else:
                    features += [0., 0.]
            features.append(float(len(game.off_pieces[p]))/game.num_pieces[p])
        if player == game.players[0]:
            features += [1., 0.]
        else:
            features += [0., 1.]
        return features

    def get_output(self, x):
        return self.sess.run(self.V, feed_dict={self.x: x})

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

            if full_stats:
                print("[Test %d/%d] %s (%s) vs %s (%s), steps: %d, wins: %d:%d (%.1f%%)" %
                      (episode, episodes,
                       player_agents[0].name, player_agents[0].player,
                       player_agents[1].name, player_agents[1].player,
                       game.num_steps, winners[0], winners[1],
                       (winners[0]/winners_total)*100.0))

        print("\nPlayed %d test games %s (%s) vs %s (%s), mean steps: %.1f, "
              "wins: %d/%d, win ratio: %.1f%%" %
              (winners_total, player_agents[0].name, player_agents[0].player,
               player_agents[1].name, player_agents[1].player,
               num_steps/winners_total, winners[0], winners[1],
               (winners[0]/winners_total)*100.0))

    def train(self, episodes=5000, test_interval=1000, test_episodes=100,
              checkpoint_interval=1000, summary_interval=100):
        print("Training started.\n")

        tf.train.write_graph(self.sess.graph_def, self.model_path, 'td_gammon.pb', as_text=False)
        summary_writer = tf.summary.FileWriter('{0}{1}'.format(self.summary_path, int(time.time()),
                                                               self.sess.graph_def))

        # the agent plays against itself, making the best move for each player
        player_agents = [TDAgent(Game.PLAYERS[0], self), TDAgent(Game.PLAYERS[1], self)]

        # train (x, r, x_next) history for experience replay
        train_history = []

        for episode in range(1, episodes+1):
            game = Game.new()
            player_index = random.randint(0, 1)

            x_next = self.extract_features(game, player_agents[player_index].player)

            while not game.is_over():
                x = x_next

                game.next_step(player_agents[player_index])
                player_index = (player_index+1) % 2

                if game.is_over():
                    x_next = None
                    r = float(game.winner())
                else:
                    x_next = self.extract_features(game, player_agents[player_index].player)
                    r = 0

                train_history.append([x, r, x_next])

                # update network only if enough train history data is obtained
                if len(train_history) >= self.min_train_history_size:
                    # Trim (x, r, x_next) train_history
                    if len(train_history) > self.max_train_history_size:
                        del train_history[:(len(train_history)-self.max_train_history_size)]

                    train_minibatch = np.array(random.sample(train_history, self.minibatch_size))

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
                    self.sess.run(self.train_op, feed_dict={self.x: list(x_batch),
                                                            self.V_next: v_next_batch})

            winner = game.winner()

            print("[Train %d/%d] (winner: '%s') in %d turns" % (episode, episodes,
                                                                player_agents[winner].player,
                                                                game.num_steps))

            # write summary every summary_interval
            if episode % summary_interval == 0 or episode == episodes:
                summaries = self.sess.run(
                    self.summaries_op,
                    feed_dict={self.x: [x], self.V_next: np.array([[r]], dtype='float')}
                )
                summary_writer.add_summary(summaries, global_step=episode)
            # save checkpoint every checkpoint_interval
            if episode % checkpoint_interval == 0 or episode == episodes:
                self.saver.save(self.sess, self.checkpoint_path+'checkpoint',
                                global_step=episode)
            # play test games every test_interval
            if episode % test_interval == 0 or episode == episodes:
                self.test(episodes=test_episodes, full_stats=False)

        summary_writer.close()

        print("\nTraining completed.")
