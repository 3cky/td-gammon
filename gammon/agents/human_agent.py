from ..game import Game


class HumanAgent(object):
    def __init__(self, player):
        self.player = player
        self.name = 'Human'

    def choose_action(self, moves, game=None):
        if not moves:
            input("No moves for you...(hit enter)")
            return None

        moves = list(moves)

        choice_index = -1
        while choice_index < 0 or choice_index > len(moves):
            print('\nChoose move for \'%s\' (%d-%d):\n' %
                  (self.player, 1, len(moves)))
            for i, move in enumerate(moves):
                m = []
                for s, e in move:
                    s += 1
                    if e != Game.OFF:
                        e += 1
                    m.append((s, e))
#                 action_name = " ".join([str(self.hand[self.hand.find(suit, value)])
#                                              for s, e in move])
                print('%d) %s' % (i+1, m))
            try:
                choice = input('Choose move (1): ')
                if not choice:
                    choice_index = 0
                else:
                    choice_index = int(choice)-1
            except ValueError:
                pass

        return moves[choice_index]
