import random


class RandomAgent(object):

    def __init__(self, player):
        self.player = player
        self.name = 'Random'

    def choose_action(self, moves, game=None):
        return random.choice(list(moves)) if moves else None
