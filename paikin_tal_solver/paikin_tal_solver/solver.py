import random

from hammoudeh_puzzle_solver.puzzle_importer import PuzzleType
from paikin_tal_solver.inter_piece_distance import InterPieceDistance


class PaikinTalSolver(object):
    """
    Paikin & Tal Solver
    """

    # stores the type of the puzzle to solve.
    _puzzle_type = PuzzleType.type1

    @staticmethod
    def puzzle_type():
        """
        Accessor for the current type of puzzles that are enabled.

        Returns (PuzzleType):
        Type of the puzzle to solve.
        """
        return PaikinTalSolver._puzzle_type

    def __init__(self, pieces, numb_puzzles, distance_function):
        """
        Constructor for the Paikan and Tal solver.

        Args:
            pieces ([PuzzlePiece])):
            numb_puzzles (int): Number of Puzzles to be solved.
            distance_function: Calculates the distance between two PuzzlePiece objects.

        """

        # Store the number of pieces.  Shuffle for good measure.
        self._pieces = random.shuffle(pieces)

        # Store the number of puzzles these collective set of pieces comprise.
        self._numb_puzzles = numb_puzzles

        # Store the function used to calculate piece to piece distances.
        self._distance_function = distance_function

        # Calculate the inter-piece distances.
        self._inter_piece_distance = InterPieceDistance(self._pieces, self._distance_function,
                                                        PaikinTalSolver.puzzle_type())

    # noinspection PyMethodMayBeStatic
    def run(self):
        """
        Runs the Paikin and Tal Solver.
        """
        pass
