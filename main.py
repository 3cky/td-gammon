import os
import argparse

from model import Model

flags = argparse.ArgumentParser(description='TD-Gammon')

flags.add_argument('--train', action='store_true', help='Train a TD-Gammon model.')
flags.add_argument('--summaries_path', default='summaries', help='Summaries path.')
flags.add_argument('--summary_name', default='', help='Summary name.')
flags.add_argument('--checkpoints_path', default='checkpoints', help='Checkpoints path.')
flags.add_argument('--test', action='store_true', help='Test against a random strategy.')
flags.add_argument('--play', action='store_true',
                   help='Play against a trained TD-Gammon strategy.')
flags.add_argument('--norestore', action='store_true', help='Do not restore a checkpoint.')
flags.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train/test.')

f = flags.parse_args()

if __name__ == '__main__':
    checkpoints_path = f.checkpoints_path
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    summaries_path = f.summaries_path
    if not os.path.exists(summaries_path):
        os.makedirs(summaries_path)

    model = Model(summaries_path, checkpoints_path, restore=not f.norestore)

    if f.test:
        model.test(episodes=f.episodes)
    elif f.play:
        model.play()
    elif f.train:
        model.train(episodes=f.episodes, summary_name=f.summary_name)
    else:
        print("Please specify mode (--train/--test/--play")
