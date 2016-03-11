import random
from enum import Enum
from paikin_tal_solver.inter_piece_distance import InterPieceDistance


class PuzzleType(Enum):
    '''
    Type of the puzzle to solve.  Type 1 has no piece rotation while type 2 allows piece rotation.
    '''

    type1 = 1
    type2 = 2

class PaikanTalSolver(object):

    # stores the type of the puzzle to solve.
    _puzzle_type = PuzzleType.type1

    def __init__(self, pieces, numb_puzzles, distance_function):

        # Store the number of pieces.  Shuffle for good measure.
        self._pieces = random.shuffle(pieces)

        # Store the number of puzzles these collective set of pieces comprise.
        self._numb_puzzles = numb_puzzles

        # Store the function used to calculate piece to piece distances.
        self._distance_function = distance_function

        # Calculate the interpiece distances.
        self._inter_piece_distance = InterPieceDistance(self._pieces, self._distance_function)

    @staticmethod
    def puzzle_type():
        '''
        Accessor for the current type of puzzles that are enabled.

        Returns (PuzzleType):
        Type of the puzzle to solve.

        '''
        return _puzzle_type