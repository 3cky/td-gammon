import numpy as np


class TDAgent(object):

    def __init__(self, player, model, name='TD-Gammon'):
        self.player = player
        self.model = model
        self.name = name

    def choose_action(self, actions, game):
        """
        Return best action according to self.evaluationFunction,
        with no lookahead.
        """
        actions = list(actions)

        if len(actions) == 1:
            return actions[0]

        features = []

        for a in actions:
            game.take_action(a, self.player)
            features.append(self.model.extract_features(game, game.opponent(self.player)))
            game.undo_action(a, self.player)

        v = self.model.get_output(features)
        v = 1. - v if self.player == game.players[0] else v

        return actions[np.argmax(v)]
