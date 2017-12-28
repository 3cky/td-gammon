import os
import copy
import time
import random


class Game:

    LAYOUT = "11-15-o,23-15-x"
    NUM_POSITIONS = 24
    NUM_PIECES = 15
    QUAD = 6
    OFF = 'off'
    TOKENS = ['x', 'o']

    def __init__(self, layout=LAYOUT, grid=None, off_pieces=None,
                 num_pieces=None, players=None):
        """
        Define a new game object
        """
        self.num_steps = 0
        self.die = Game.QUAD
        self.layout = layout
        if grid:
            self.grid = copy.deepcopy(grid)
            self.off_pieces = copy.deepcopy(off_pieces)
            self.num_pieces = copy.deepcopy(num_pieces)
            self.players = players
            return
        self.players = Game.TOKENS
        self.grid = [[] for _ in range(Game.NUM_POSITIONS)]
        self.off_pieces = {}
        self.num_pieces = {}
        for t in self.players:
            self.off_pieces[t] = []
            self.num_pieces[t] = 0

    @staticmethod
    def new():
        game = Game()
        game.reset()
        return game

    def roll_dice(self):
        return (random.randint(1, self.die), random.randint(1, self.die))

    def play(self, players, draw=False):
        player_num = random.randint(0, 1)
        while not self.is_over():
            self.next_step(players[player_num], player_num, draw=draw)
            player_num = (player_num + 1) % 2
        if draw:
            print("\nGame is over, winner: '%s'" % self.players[self.winner()])
        return self.winner()

    def next_step(self, player, player_num, draw=False):
        self.num_steps += 1

        roll = self.roll_dice()

        if draw:
            self.draw()

        self.take_turn(player, roll, draw=draw)

    def take_turn(self, player, roll, draw=False):
        if draw:
            print("'%s' rolled <%d, %d>" % (player.player, roll[0], roll[1]))
            time.sleep(1)

        if player.player == 'o':
            self.flip_board()

        moves = self.get_actions(roll, player.player)

        if player.player == 'o':
            self.flip_board()
            moves = self.flip_moves(moves)

        if draw:
            print("'%s' moves available: %d>" % (player.player, len(moves)))
            time.sleep(1)

        move = player.choose_action(moves, self) if moves else None

        if move:
            self.take_action(move, player.player)

    def clone(self):
        """
        Return an exact copy of the game. Changes can be made
        to the cloned version without affecting the original.
        """
        return Game(None, self.grid, self.off_pieces, self.num_pieces, self.players)

    def take_action(self, action, token):
        """
        Makes given move for player, assumes move is valid
        """
        for s, e in action:
            piece = self.grid[s].pop()
            if e == Game.OFF:
                self.off_pieces[token].append(piece)
            else:
                self.grid[e].append(piece)

    def undo_action(self, action, token):
        """
        Reverses given move for player, assumes move is valid
        """
        for s, e in reversed(action):
            if e == Game.OFF:
                piece = self.off_pieces[token].pop()
            else:
                piece = self.grid[e].pop()
            self.grid[s].append(piece)

    def get_actions(self, roll, player):
        """
        Get set of all possible move tuples
        """
        moves = set()

        states = set()

        r1, r2 = roll

        if r1 == r2:  # doubles
            # first check for full move
            self.find_moves(tuple([r1]*4), player, (), moves, states)
            # has no moves, allow to move two pieces from head on first turn
            if not moves and len(self.grid[-1]) == Game.NUM_PIECES:
                self.find_moves(tuple([r1]*4), player, (), moves, states, max_from_head=2)
            # keep trying until we find some moves
            i = 3
            while not moves and i > 0:
                self.find_moves(tuple([r1]*i), player, (), moves, states)
                i -= 1
        else:
            # first check for full moves
            self.find_moves((r1, r2), player, (), moves, states)
            self.find_moves((r2, r1), player, (), moves, states)
            # has no moves, try moving only one piece
            if not moves:
                roll = sorted(roll, reverse=True)  # first check max roll value
                for r in roll:
                    self.find_moves((r, ), player, (), moves, states)
                    if moves:
                        break

        return moves

    def find_moves(self, rs, player, move, moves, states, max_from_head=1, from_head=0):
        if len(rs) == 0:
            # check for duplicate end state
            state = tuple([tuple(s) for s in self.grid])
            if state not in states:
                states.add(state)
                moves.add(move)
            return

        r, rs = rs[0], rs[1:]

        start = head = len(self.grid)-1
        if from_head >= max_from_head:
            start -= 1

        # check all board positions starting from head
        for s in range(start, -1, -1):
            # end position for given roll value
            e = s-r

            # start is head flag
            h = (s == head)

            # check for on-board moves
            if self.is_valid_move(s, e, player):
                piece = self.grid[s].pop()
                self.grid[e].append(piece)
                self.find_moves(rs, player, move+((s, e), ), moves, states,
                                max_from_head, from_head+1 if h else from_head)
                self.grid[e].pop()
                self.grid[s].append(piece)

            # check for off-board moves
            if self.can_offboard(player) and self.can_remove_piece(player, s, r):
                piece = self.grid[s].pop()
                self.off_pieces[player].append(piece)
                self.find_moves(rs, player, move+((s, Game.OFF), ), moves, states,
                                max_from_head, from_head+1 if h else from_head)
                self.off_pieces[player].pop()
                self.grid[s].append(piece)

    def opponent(self, token):
        """
        Retrieve opponent players token for a given players token.
        """
        for t in self.players:
            if t != token:
                return t

    def is_won(self, player):
        """
        If game is over and player won, return True, else return False
        """
        return self.is_over() and player == self.players[self.winner()]

    def is_lost(self, player):
        """
        If game is over and player lost, return True, else return False
        """
        return self.is_over() and player != self.players[self.winner()]

    def flip_board(self):
        """
        Flip the board, so 'x' layout becomes 'o' and vice versa
        """
        h = Game.NUM_POSITIONS//2
        for i in range(h):
            self.grid[i], self.grid[i+h] = self.grid[i+h], self.grid[i]

    @staticmethod
    def flip_moves(moves):
        """
        Flip the moves, so 'x' move becomes 'o' move and vice versa
        """
        flipped_moves = set()

        h = Game.NUM_POSITIONS//2
        for move in moves:
            flipped_move = ()
            for s, e in move:
                s = (s+h) % Game.NUM_POSITIONS
                if e != Game.OFF:
                    e = (e+h) % Game.NUM_POSITIONS
                flipped_move += ((s, e), )
            flipped_moves.add(flipped_move)

        return flipped_moves

    def reset(self):
        """
        Resets game to original layout.
        """
        for col in self.layout.split(','):
            loc, num, token = col.split('-')
            self.grid[int(loc)] = [token for _ in range(int(num))]
        for col in self.grid:
            for piece in col:
                self.num_pieces[piece] += 1

    def winner(self):
        """
        Get winner.
        """
        return 0 if len(self.off_pieces[self.players[0]]) == self.num_pieces[self.players[0]] \
            else 1

    def is_over(self):
        """
        Checks if the game is over.
        """
        for p in self.players:
            if len(self.off_pieces[p]) == self.num_pieces[p]:
                return True
        return False

    def can_offboard(self, player):
        count = 0
        for i in range(self.die):
            if len(self.grid[i]) > 0 and self.grid[i][0] == player:
                count += len(self.grid[i])
        if count+len(self.off_pieces[player]) == self.num_pieces[player]:
            return True
        return False

    def can_remove_piece(self, player, start, r):
        """
        Can we remove a piece from location start with roll r ?
        In this function we assume we are cool to offboard,
        i.e. all pieces are in the home quadrant.
        """
        if start > self.die-1:
            return False
        if len(self.grid[start]) == 0 or self.grid[start][0] != player:
            return False
        if start-r == -1:
            return True
        if start-r < -1:
            for i in range(start+1, self.die):
                if len(self.grid[i]) != 0 and self.grid[i][0] == self.players[0]:
                    return False
            return True
        return False

    def is_valid_move(self, start, end, token):
        if len(self.grid[start]) > 0 and self.grid[start][0] == token:
            if end < 0 or end >= len(self.grid):
                return False
            if len(self.grid[end]) == 0 or self.grid[end][0] == token:
                return True
        return False

    def draw_col(self, i, col):
        print("|", end=" ")
        if i == -2:
            col_num = col + 1
            if col_num < 10:
                print("", end=" ")
            print(str(col_num), end=" ")
        elif i == -1:
            print("--", end=" ")
        elif len(self.grid[col]) > i:
            print(" " + self.grid[col][i], end=" ")
        else:
            print("  ", end=" ")

    def draw(self):
        os.system('clear')
        largest = max([len(self.grid[i]) for i in range(len(self.grid) // 2, len(self.grid))])
        for i in range(-2, largest):
            for col in range(len(self.grid) // 2, len(self.grid)):
                self.draw_col(i, col)
            print("|")
        print("")
        print("")
        largest = max([len(self.grid[i]) for i in range(len(self.grid) // 2)])
        for i in range(largest - 1, -3, -1):
            for col in range(len(self.grid) // 2 - 1, -1, -1):
                self.draw_col(i, col)
            print("|")
        for t in self.players:
            print("'" + t + "' offboard: [", end="")
            for _piece in self.off_pieces[t]:
                print(t, end=" ")
            print("]")
