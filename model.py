from __future__ import division

import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

from gammon.game import Game
from gammon.agents.human_agent import HumanAgent
from gammon.agents.td_gammon_agent import TDAgent
from gammon.agents.heuristic_agent import HeuristicAgent


# helper to create a dense, fully-connected layer
def dense_layer(x, size, activation, name):
    return layers.fully_connected(x, size, activation_fn=activation, scope=name)


class Model(object):
    def __init__(self, sess, model_path, summary_path, checkpoint_path, restore=False):
        self.model_path = model_path
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path

        # setup our session
        self.sess = sess
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # lambda decay
        lambda_ = tf.maximum(0.7, tf.train.exponential_decay(
            0.9, self.global_step, 30000, 0.96, staircase=True), name='lambda')

        # learning rate decay
        alpha = tf.maximum(0.01, tf.train.exponential_decay(
            0.1, self.global_step, 40000, 0.96, staircase=True), name='alpha')

        tf.summary.scalar('lambda', lambda_)
        tf.summary.scalar('alpha', alpha)

        # describe network size
        layer_size_input = 100
        layer_size_hidden = 64
        layer_size_output = 1

        # placeholders for input and target output
        self.x = tf.placeholder('float', [None, layer_size_input], name='x')
        self.V_next = tf.placeholder('float', [1, layer_size_output], name='V_next')

        # build network arch
        hidden = dense_layer(self.x, layer_size_hidden, tf.sigmoid, name='hidden')
        self.V = dense_layer(hidden, layer_size_output, tf.sigmoid, name='V')

        # watch the individual value predictions over time
        tf.summary.scalar('V_next', tf.reduce_sum(self.V_next))
        tf.summary.scalar('V', tf.reduce_sum(self.V))

        # delta = V_next - V
        delta_op = tf.reduce_sum(self.V_next - self.V, name='delta')

        # mean squared error of the difference between the next state and the current state
        loss_op = tf.reduce_mean(tf.squared_difference(self.V_next, self.V), name='loss')

        # check if the model predicts the correct state
        accuracy_op = tf.reduce_sum(tf.cast(tf.equal(tf.round(self.V_next), tf.round(self.V)),
                                            dtype='float'), name='accuracy')

        # track the number of steps and average loss for the current game
        with tf.variable_scope('game'):
            game_step = tf.Variable(tf.constant(0.0), name='game_step', trainable=False)
            game_step_op = game_step.assign_add(1.0)

            loss_sum = tf.Variable(tf.constant(0.0), name='loss_sum', trainable=False)
            delta_sum = tf.Variable(tf.constant(0.0), name='delta_sum', trainable=False)
            accuracy_sum = tf.Variable(tf.constant(0.0), name='accuracy_sum', trainable=False)

            loss_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            delta_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            accuracy_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)

            loss_sum_op = loss_sum.assign_add(loss_op)
            delta_sum_op = delta_sum.assign_add(delta_op)
            accuracy_sum_op = accuracy_sum.assign_add(accuracy_op)

            loss_avg_op = loss_sum / tf.maximum(game_step, 1.0)
            delta_avg_op = delta_sum / tf.maximum(game_step, 1.0)
            accuracy_avg_op = accuracy_sum / tf.maximum(game_step, 1.0)

            loss_avg_ema_op = loss_avg_ema.apply([loss_avg_op])
            delta_avg_ema_op = delta_avg_ema.apply([delta_avg_op])
            accuracy_avg_ema_op = accuracy_avg_ema.apply([accuracy_avg_op])

            tf.summary.scalar('game/loss_avg', loss_avg_op)
            tf.summary.scalar('game/delta_avg', delta_avg_op)
            tf.summary.scalar('game/accuracy_avg', accuracy_avg_op)
            tf.summary.scalar('game/loss_avg_ema', loss_avg_ema.average(loss_avg_op))
            tf.summary.scalar('game/delta_avg_ema', delta_avg_ema.average(delta_avg_op))
            tf.summary.scalar('game/accuracy_avg_ema', accuracy_avg_ema.average(accuracy_avg_op))

            # reset per-game monitoring variables
            game_step_reset_op = game_step.assign(0.0)
            loss_sum_reset_op = loss_sum.assign(0.0)
            self.reset_op = tf.group(*[loss_sum_reset_op, game_step_reset_op])

        # increment global step: we keep this as a variable so it's saved with checkpoints
        global_step_op = self.global_step.assign_add(1)

        # get gradients of output V wrt trainable variables (weights and biases)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.V, tvars)

        # watch the weight and gradient distributions
        for grad, var in zip(grads, tvars):
            tf.summary.histogram(var.name, var)
            tf.summary.histogram(var.name + '/gradients/grad', grad)

        # for each variable, define operations to update the var with delta,
        # taking into account the gradient as part of the eligibility trace
        apply_gradients = []
        with tf.variable_scope('apply_gradients'):
            for grad, var in zip(grads, tvars):
                with tf.variable_scope('trace'):
                    # e-> = lambda * e-> + <grad of output w.r.t weights>
                    trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                    trace_op = trace.assign((lambda_ * trace) + grad)
                    tf.summary.histogram(var.name + '/traces', trace)

                # grad with trace = alpha * delta * e
                grad_trace = alpha * delta_op * trace_op
                tf.summary.histogram(var.name + '/gradients/trace', grad_trace)

                grad_apply = var.assign_add(grad_trace)
                apply_gradients.append(grad_apply)

        # as part of training we want to update our step and other monitoring variables
        with tf.control_dependencies([
            global_step_op,
            game_step_op,
            loss_sum_op,
            delta_sum_op,
            accuracy_sum_op,
            loss_avg_ema_op,
            delta_avg_ema_op,
            accuracy_avg_ema_op
        ]):
            # define single operation to apply all gradient updates
            self.train_op = tf.group(*apply_gradients, name='train')

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
        return np.array(features).reshape(1, -1)

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

        for episode in range(1, episodes+1):
            game = Game.new()
            player_index = random.randint(0, 1)

            x = self.extract_features(game, player_agents[player_index].player)

            while not game.is_over():
                game.next_step(player_agents[player_index])
                player_index = (player_index+1) % 2

                x_next = self.extract_features(game, player_agents[player_index].player)
                V_next = self.get_output(x_next)
                self.sess.run(self.train_op, feed_dict={self.x: x, self.V_next: V_next})

                x = x_next

            winner = game.winner()

            print("[Train %d/%d] (winner: '%s') in %d turns" % (episode, episodes,
                                                                player_agents[winner].player,
                                                                game.num_steps))

            _, global_step, _ = self.sess.run([
                self.train_op,
                self.global_step,
                self.reset_op
            ], feed_dict={self.x: x, self.V_next: np.array([[winner]], dtype='float')})

            # write summary every summary_interval
            if episode % summary_interval == 0 or episode == episodes:
                summaries = self.sess.run(
                    self.summaries_op,
                    feed_dict={self.x: x, self.V_next: np.array([[winner]], dtype='float')}
                )
                summary_writer.add_summary(summaries, global_step=global_step)
            # save checkpoint every checkpoint_interval
            if episode % checkpoint_interval == 0 or episode == episodes:
                self.saver.save(self.sess, self.checkpoint_path+'checkpoint',
                                global_step=global_step)
            # play test games every test_interval
            if episode % test_interval == 0 or episode == episodes:
                self.test(episodes=test_episodes, full_stats=False)

        summary_writer.close()

        print("\nTraining completed.")
