from paikin_tal_solver.paikin_tal_solver import PaikanTalSolver, PuzzleType
from paikin_tal_solver.simple_puzzle_piece import PuzzlePieceSide


class InterPieceDistance(object):

    # Since a type 1 puzzle. then for each side of x_i, x_j can only be paired a single way.
    TYPE1_POSSIBLE_PAIRINGS = 1
    # Since a type 2 puzzle, then for each side of x_i, x_j can be rotated up to four different ways.
    TYPE2_POSSIBLE_PAIRINGS = 4

    def __init__(self, pieces, distance_function):
        """
        Stores the piece to piece distance

        Args:
            pieces ([SimplePuzzlePiece]): List of all puzzle pieces in simple form.s
            distance_function:

        Returns:
            An interpiece distance object.
        """

        # Store the number of pieces in the puzzle.
        self._piece_count = len(pieces)

        # Based on the puzzle type, determine the number of possible piece to piece pairings.
        if InterPieceDistance.TYPE1_POSSIBLE_PAIRINGS:
            numb_possible_pairings = (PaikanTalSolver.puzzle_type() == PuzzleType.type1)
        else:
            numb_possible_pairings = InterPieceDistance.TYPE2_POSSIBLE_PAIRINGS
        # Build an empty array to store the piece to piece distances
        self._piece_distances = [[[[None for _ in range(0, numb_possible_pairings)]
                                   for _ in xrange(PuzzlePieceSide.numb_sides())]
                                  for _ in xrange(self._piece_count)]
                                 for _ in xrange(self._piece_count)]

        # Calculates the piece to piece distances so we only need to do it once.
        for x_i in range(0, self._piece_count):
            for x_j in range(0, x_i):
                for x_i_side in PuzzlePieceSide.get_all_sides():

                    # For type one puzzles, only a single possible complementary side
                    if PaikanTalSolver.puzzle_type() == PuzzleType.type1:
                        complimentary_side = x_i_side.complimentary_side()
                        dist = distance_function(pieces[x_i], x_i_side, pieces[x_j], complimentary_side)
                        # Store the data on the mirrorer portion of the array.
                        self._piece_distances[x_i][x_j][x_i_side.value][0] = dist
                        self._piece_distances[x_j][x_i][complimentary_side.value][0] = dist

                    # For type two puzzles, handle all possible combinations of sides (16 in total).
                    if PaikanTalSolver.puzzle_type() == PuzzleType.type2:
                        for x_j_side in PuzzlePieceSide.get_all_sides():
                            # Calculate the distance between the two pieces.
                            dist = distance_function(pieces[x_i], x_i_side, pieces[x_j], x_j_side)
                            # Store the side to side distance information
                            self._piece_distances[x_i][x_j][x_i_side.value][x_j_side.value] = dist
                            self._piece_distances[x_j][x_i][x_j_side.value][x_i_side.value] = dist
