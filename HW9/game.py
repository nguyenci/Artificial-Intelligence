import random
from copy import deepcopy

"""
HW9 - game.py
Author: Cinthya Nguyen
Class: CS540 SP23
"""


class TeekoPlayer:
    """
    An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """
        Initializes a TeekoPlayer object by randomly selecting red or black as its piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """
        Selects a (row, col) space for the next move. You may assume that whenever this function
        is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the curr state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        drop_phase = True  # Detect drop phase

        count = 0
        for i in range(5):
            for j in range(5):
                if state[i][j] == 'r' or state[i][j] == 'b':
                    count += 1

        if count >= 8:
            drop_phase = False

        # Choose a piece to move and remove it from the board
        move = []
        a, change = self.max_value(state, 3)
        if not drop_phase:
            for i in range(5):
                for j in range(5):
                    if state[i][j] != change[i][j]:
                        if state[i][j] != " ":
                            move.insert(1, (i, j))
                        else:
                            move.insert(0, (i, j))
            return move

        for i in range(5):
            for j in range(5):
                if state[i][j] != change[i][j]:
                    move.insert(0, (i, j))
                    return move

    def succ(self, state, curr):
        """
        Takes in board state and returns list of legal successors.
        :param state: Board state
        :param curr: Current player's type
        :return: List of legal successors
        """

        count = 0
        empty = set()
        place = set()
        for i in range(5):
            for j in range(5):
                if state[i][j] == ' ':
                    empty.add((i, j))
                elif state[i][j] == curr:
                    count += 1
                    place.add((i, j))
                else:
                    count += 1

        successors = []
        if count < 8:  # Drop phase

            for i, j in empty:
                board_state = deepcopy(state)
                board_state[i][j] = curr
                successors.append(board_state)

        else:  # Continue gameplay

            for i, j in place:
                adjacent = set()
                keep = set()
                adjacent.add((i - 1, j - 1))
                adjacent.add((i - 1, j))
                adjacent.add((i - 1, j + 1))
                adjacent.add((i, j - 1))
                adjacent.add((i, j + 1))
                adjacent.add((i + 1, j - 1))
                adjacent.add((i + 1, j))
                adjacent.add((i + 1, j + 1))

                for adj in adjacent:
                    if not ((-1 in adj) or (5 in adj)):
                        keep.add(adj)

                intersect = keep.intersection(empty)
                empty.difference(intersect)

                for x, y in intersect:
                    board_state = deepcopy(state)
                    board_state[x][y] = curr
                    board_state[i][j] = " "
                    successors.append(board_state)

        return successors

    def helper(self, state, player):
        """
        Helper function for the heuristic function.
        :param state: Board state.
        :param player: Current player's type.
        :return: Distance of board pieces
        """
        
        position = []
        for i, j in enumerate(state):
            for k, color in enumerate(j):
                if color == player:
                    position.append((i, k))

        size = len(position)

        # Average of rows and cols
        row_avg = sum([a[0] for a in position]) / (size + 1)
        col_avg = sum([a[1] for a in position]) / (size + 1)

        # Distance of the pieces
        distance = sum([(a[0] - row_avg) ** 2 for a in position]) + sum([(a[1] - col_avg) ** 2
                                                                         for a in position])
        return distance

    def heuristic_game_value(self, state):
        """
        Heuristic function for the game value.
        :param state: Board state.
        :return: Heuristic value
        """

        state_copy = deepcopy(state)

        if self.game_value(state_copy) != 0:  # There is a winner
            return self.game_value(state_copy)

        my_dist = self.helper(state_copy, self.my_piece)  # Dist of my pieces
        ai_dist = self.helper(state_copy, self.opp)  # Dist of opp pieces

        heuristic = float(1 / (1 + my_dist)) - float(1 / (1 + ai_dist))  # Heuristic value

        return heuristic

    def max_value(self, state, depth):
        """
        Every call increases value of depth. Alpha-beta pruning is implemented with the call of
        min_value.
        """

        if self.game_value(state) == 1:
            return 1, state
        elif self.game_value(state) == -1:
            return -1, state

        if depth == 0:
            return self.heuristic_game_value(state), state

        alpha = float('-Inf')
        board_state = state
        for s in self.succ(state, self.my_piece):
            a = self.min_value(s, depth - 1)[0]
            if a > alpha:
                alpha = a
                board_state = s

        return alpha, board_state

    def min_value(self, state, depth):
        """
        Called by max_value.
        """

        if self.game_value(state) == -1:
            return -1, state
        elif self.game_value(state) == 1:
            return 1, state

        if depth == 0:
            return self.heuristic_game_value(state), state

        beta = float('Inf')
        board_state = state
        for st in self.succ(state, self.opp):
            b = self.max_value(st, depth - 1)[0]
            if b < beta:
                beta = b
                board_state = st

        return beta, board_state

    def opponent_move(self, move):
        """
        Validates the opponent's next move against the internal board representation. You don't
        need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """

        if len(move) > 1:  # Validate input
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")

        self.place_piece(move, self.opp)  # Make move

    def place_piece(self, move, piece):
        """
        Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """

        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """
        Formatted printing for the board
        """

        for row in range(len(self.board)):
            line = str(row) + ": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the curr board status for a win condition

        Args:
        state (list of lists): either the curr state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        """

        for row in state:  # Check horizontal wins
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        for col in range(5):  # Check vertical wins
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col] == state[i + 2][
                     col] == state[i + 3][col]:
                    return 1 if state[i][col] == self.my_piece else -1

        for row in range(2):  # Check / diagonal wins
            for col in range(3, 5):
                if state[row][col] != ' ' and state[row][col] == state[row + 1][col - 1] == state[
                        row + 2][col - 2] == state[row + 3][col - 3]:
                    return 1 if state[row][col] == self.my_piece else -1

        for row in range(2):  # Check \ diagonal wins
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row + 1][col + 1] == state[
                        row + 2][col + 2] == state[row + 3][col + 3]:
                    return 1 if state[row][col] == self.my_piece else -1

        for row in range(4):  # Check 2x2 box wins
            for col in range(4):
                if state[row][col] != ' ' and state[row][col] == state[row][col + 1] == state[
                        row + 1][col + 1] == state[row + 1][col]:
                    return 1 if state[row][col] == self.my_piece else -1

        return 0  # No winner


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved at " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved from " + chr(move[1][1] + ord("A")) + str(move[1][0]))
            print("  to " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0]) - ord("A")),
                                      (int(move_from[1]), ord(move_from[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
