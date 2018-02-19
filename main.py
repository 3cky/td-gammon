import os
import argparse

from model import Model

flags = argparse.ArgumentParser(description='TD-Gammon')

flags.add_argument('--train', action='store_true', help='Train a TD-Gammon model.')
flags.add_argument('--test', action='store_true', help='Test against a random strategy.')
flags.add_argument('--play', action='store_true',
                   help='Play against a trained TD-Gammon strategy.')
flags.add_argument('--norestore', action='store_true', help='Do not restore a checkpoint.')
flags.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train/test.')

FLAGS = flags.parse_args()

model_path = os.environ.get('MODEL_PATH', 'models/')
summary_path = os.environ.get('SUMMARY_PATH', 'summaries/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

if not os.path.exists(summary_path):
    os.makedirs(summary_path)

if __name__ == '__main__':
    model = Model(checkpoint_path, restore=not FLAGS.norestore)
    if FLAGS.test:
        model.test(episodes=FLAGS.episodes)
    elif FLAGS.play:
        model.play()
    elif FLAGS.train:
        model.train(episodes=FLAGS.episodes)
    else:
        print("Please specify mode (--train/--test/--play")
