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
    BLOCK_LENGTH = 6
    PLAYERS = ['x', 'o']

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
        self.players = Game.PLAYERS
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

    def play(self, player_agents, draw=False):
        assert len(player_agents) == 2
        assert player_agents[0].player == self.players[0]
        player_index = random.randint(0, 1)
        while not self.is_over():
            self.next_step(player_agents[player_index], draw=draw)
            player_index = (player_index+1) % 2
        if draw:
            print("\nGame is over, winner: '%s'" % self.players[self.winner()])
        return self.winner()

    def next_step(self, player_agent, draw=False):
        self.num_steps += 1

        roll = self.roll_dice()

        if draw:
            self.draw()

        self.take_turn(player_agent, roll, draw=draw)

    def take_turn(self, player_agent, roll, draw=False):
        if draw:
            print("'%s' rolled <%d, %d>" % (player_agent.player, roll[0], roll[1]))
            time.sleep(1)

        if player_agent.player == self.PLAYERS[1]:
            self.flip_board()

        moves = self.get_actions(roll, player_agent.player)

        if player_agent.player == self.PLAYERS[1]:
            self.flip_board()
            moves = self.flip_moves(moves)

        if draw:
            print("'%s' moves available: %d>" % (player_agent.player, len(moves)))
            time.sleep(1)

        move = player_agent.choose_action(moves, self) if moves else None

        if move:
            self.take_action(move, player_agent.player)

    def clone(self):
        """
        Return an exact copy of the game. Changes can be made
        to the cloned version without affecting the original.
        """
        return Game(None, self.grid, self.off_pieces, self.num_pieces, self.players)

    def take_action(self, action, player):
        """
        Makes given move for player, assumes move is valid
        """
        for s, e in action:
            piece = self.grid[s].pop()
            if e == Game.OFF:
                self.off_pieces[player].append(piece)
            else:
                self.grid[e].append(piece)

    def undo_action(self, action, player):
        """
        Reverses given move for player, assumes move is valid
        """
        for s, e in reversed(action):
            if e == Game.OFF:
                piece = self.off_pieces[player].pop()
            else:
                piece = self.grid[e].pop()
            self.grid[s].append(piece)

    def get_actions(self, roll, player):
        """
        Get set of all possible move tuples
        """
        moves = set()

        states = set()

        opp_minpos = self.opponent_min_position(player)
        assert opp_minpos is not None

        r1, r2 = roll

        if r1 == r2:  # doubles
            # first check for full move
            self.find_moves(tuple([r1]*4), player, opp_minpos, (), moves, states)
            # has no moves, allow to move two pieces from head on first turn
            if not moves and len(self.grid[-1]) == Game.NUM_PIECES:
                self.find_moves(tuple([r1]*4), player, opp_minpos,
                                (), moves, states, max_from_head=2)
            # keep trying until we find some moves
            i = 3
            while not moves and i > 0:
                self.find_moves(tuple([r1]*i), player, opp_minpos, (), moves, states)
                i -= 1
        else:
            # first check for full moves
            self.find_moves((r1, r2), player, opp_minpos, (), moves, states)
            self.find_moves((r2, r1), player, opp_minpos, (), moves, states)
            # has no moves, try moving only one piece
            if not moves:
                roll = sorted(roll, reverse=True)  # first check max roll value
                for r in roll:
                    self.find_moves((r, ), player, opp_minpos, (), moves, states)
                    if moves:
                        break

        return moves

    def find_moves(self, rs, player, opp_minpos, move, moves, states, max_from_head=1):
        if len(rs) == 0:
            # check for duplicate end state
            state = tuple([tuple(s) for s in self.grid])
            if state not in states:
                states.add(state)
                moves.add(move)
            return

        r, rs = rs[0], rs[1:]

        start = head = len(self.grid)-1
        if max_from_head <= 0:
            start -= 1

        # check all board positions starting from head
        for s in range(start, -1, -1):
            # end position for given roll value
            e = s-r

            # start is head flag
            h = (s == head)

            # check for on-board moves
            if self.is_valid_move(s, e, player, opp_minpos):
                piece = self.grid[s].pop()
                self.grid[e].append(piece)
                self.find_moves(rs, player, opp_minpos, move+((s, e), ), moves, states,
                                max_from_head-1 if h else max_from_head)
                self.grid[e].pop()
                self.grid[s].append(piece)

            # check for off-board moves
            if self.can_offboard(player) and self.can_remove_piece(player, s, r):
                piece = self.grid[s].pop()
                self.off_pieces[player].append(piece)
                self.find_moves(rs, player, opp_minpos, move+((s, Game.OFF), ), moves, states,
                                max_from_head-1 if h else max_from_head)
                self.off_pieces[player].pop()
                self.grid[s].append(piece)

    def opponent(self, player):
        """
        Retrieve opponent player for a given player.
        """
        for t in self.players:
            if t != player:
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

    def opponent_min_position(self, player):
        """
        Get player opponent min position (in its coordinate system, so 0 is home, 23 is head),
        or None if no opponent was found
        """
        for p in range(Game.NUM_POSITIONS):
            op = self.flip_position(p)
            if self.grid[op] and self.grid[op][0] != player:
                return p
        return None

    @staticmethod
    def flip_position(pos):
        return (pos+Game.NUM_POSITIONS//2) % Game.NUM_POSITIONS

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
            loc, num, player = col.split('-')
            self.grid[int(loc)] = [player for _ in range(int(num))]
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
        if self.off_pieces[player]:
            return True
        count = 0
        num_pieces = self.num_pieces[player]
        for i in range(self.die):
            if self.can_pick_piece(player, i):
                count += len(self.grid[i])
                if count == num_pieces:
                    return True
        return False

    def can_remove_piece(self, player, start, r):
        """
        Can we remove a piece from location start with roll r ?
        In this function we assume we are cool to offboard,
        i.e. all pieces are in the home quadrant.
        """
        assert start >= 0 and start < len(self.grid), "start: %s" % start
        # Can't remove piece from non-home position
        if start > self.die-1:
            return False
        # Can't remove piece from empty or opponent-occupied position
        if not self.can_pick_piece(player, start):
            return False
        # Can remove piece from position with exact roll value
        if start-r == -1:
            return True
        # Can remove piece from position only if previous positions in home are empty
        if start-r < -1:
            for i in range(start+1, self.die):
                if self.can_pick_piece(player, i):
                    return False
            return True
        return False

    def is_valid_move(self, start, end, player, opp_minpos):
        assert start >= 0 and start < len(self.grid), "start: %s" % start
        return self.can_pick_piece(player, start) and self.can_place_piece(player, end, opp_minpos)

    def can_pick_piece(self, player, pos):
        return self.grid[pos] and self.grid[pos][0] == player

    def can_place_piece(self, player, pos, opp_minpos):
        if pos < 0 or pos >= len(self.grid):
            return False
        if self.grid[pos] and self.grid[pos][0] != player:
            return False
        # Check for block ahead of opponent pieces
        if opp_minpos < self.BLOCK_LENGTH or self.flip_position(pos) > opp_minpos:
            return True
        w = 0
        for p in range(opp_minpos):
            fp = self.flip_position(p)
            if fp == pos or self.can_pick_piece(player, fp):
                w += 1
                if w >= self.BLOCK_LENGTH:
                    return False
            else:
                w = 0
        return True

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
