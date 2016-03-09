from enum import Enum


class PuzzleType(Enum):
    '''
    Type of the puzzle to solve.  Type 1 has no piece rotation while type 2 allows piece rotation.
    '''

    type1 = 1
    type2 = 2

class PaikanTalSolver(object):

    # stores the type of the puzzle to solve.
    _puzzle_type = PuzzleType.type1

    @staticmethod
    def puzzle_type():
        '''
        Accessor for the current type of puzzles that are enabled.

        Returns (PuzzleType):
        Type of the puzzle to solve.

        '''
        return _puzzle_type