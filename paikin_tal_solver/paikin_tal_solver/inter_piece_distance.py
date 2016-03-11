import numpy
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePieceSide
from paikin_tal_solver.paikin_tal_solver import PaikanTalSolver, PuzzleType


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
            distance_function: Function to calculate the distance between two pieces.

        """

        # Store the number of pieces in the puzzle.
        self._piece_count = len(pieces)

        # Based on the puzzle type, determine the number of possible piece to piece pairings.
        if InterPieceDistance.TYPE1_POSSIBLE_PAIRINGS:
            numb_possible_pairings = (PaikanTalSolver.puzzle_type() == PuzzleType.type1)
        else:
            numb_possible_pairings = InterPieceDistance.TYPE2_POSSIBLE_PAIRINGS

        # Build an empty array to store the piece to piece distances
        self._piece_distances = numpy.empty((self._piece_count, self._piece_count,
                                             PuzzlePieceSide.numb_sides(), numb_possible_pairings))

        # Calculates the piece to piece distances so we only need to do it once.
        for x_i in range(0, self._piece_count):
            for x_j in range(0, self._piece_count):
                for x_i_side in PuzzlePieceSide.get_all_sides():

                    # For type one puzzles, only a single possible complementary side
                    if PaikanTalSolver.puzzle_type() == PuzzleType.type1:
                        complimentary_side = x_i_side.complimentary_side()
                        dist = distance_function(pieces[x_i], x_i_side, pieces[x_j], complimentary_side)
                        self._piece_distances[x_i, x_j, x_i_side.value, 0] = dist

                    # For type two puzzles, handle all possible combinations of sides (16 in total).
                    if PaikanTalSolver.puzzle_type() == PuzzleType.type2:
                        for x_j_side in PuzzlePieceSide.get_all_sides():
                            # Calculate the distance between the two pieces.
                            dist = distance_function(pieces[x_i], x_i_side, pieces[x_j], x_j_side)
                            self._piece_distances[x_i, x_j, x_i_side.value, x_j_side.value] = dist

        # Calculate the best buddies using the interdistance information.
        self.calculate_best_buddies()

    def calculate_best_buddies(self):
        """
        Finds the best buddies in a distance array.

        """

        # Can have a single best buddy per side.
        shape = self._piece_distances.shape
        best_distance = numpy.empty(shape[0], shape[2])

        # Find the best buddies for each piece on each side.
        for x_i in range(0, shape[0]):
            for side in range(0, shape[2]):  # Iterate through all the sides.

                # Special handle the first check
                first_check = True

                # Find the closest neighbor
                for y_i in range(0, shape[1]):
                    # A patch cannot be it own best friend so skip itself.
                    if x_i == y_i:
                        continue

                    # Check all possible pairings.
                    for other_side in range(0, shape[3]):

                        # Check if the two pieces are the closest to each other.
                        if first_check \
                                or best_distance[x_i, y_i, side] > self._piece_distances[x_i, y_i, side, other_side]:

                            # On a type 1 puzzle,
                            if PaikanTalSolver.puzzle_type() == PuzzleType.type1:
                                stored_other_side = side.complimentary_side()
                            if PaikanTalSolver.puzzle_type() == PuzzleType.type2:
                                stored_other_side = other_side

                            # Store the best distance.
                            best_distance[x_i, side] = (self._piece_distances[x_i, y_i, side, other_side],
                                                        y_i, stored_other_side)
                            first_check = False

        # Now that best distances have been found, check for best buddies.
        best_buddies = numpy.empty(shape[0], shape[1], shape[2])
        for x_i in range(0, shape[0]):
            for side in range(0, shape[2]):  # Iterate through all the sides.
                # Get the information on x_i's best buddy
                (_, x_bb, x_bb_side) = best_distance[x_i, side]
                # Get the information from the best buddy itself.
                (_, y_bb, y_bb_side) = best_buddies[x_bb, side]

                # Check if we agreed on being best buddies
                if x_i == y_bb and side == y_bb_side:
                    best_buddies[x_i, side] = y_bb
                    best_buddies[y_bb, y_bb_side] = x_i

