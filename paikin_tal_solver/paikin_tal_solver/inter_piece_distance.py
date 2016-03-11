import numpy

from hammoudeh_puzzle_solver.puzzle_importer import PuzzleType
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePieceSide


class InterPieceDistance(object):
    """
    Master class for managing inter-puzzle piece distances as well as best buddies
    and the starter puzzle pieces as defined by the Paikin and Tal paper.
    """

    # Since a type 1 puzzle. then for each side of x_i, x_j can only be paired a single way.
    TYPE1_POSSIBLE_PAIRINGS = 1
    # Since a type 2 puzzle, then for each side of x_i, x_j can be rotated up to four different ways.
    TYPE2_POSSIBLE_PAIRINGS = 4

    def __init__(self, pieces, distance_function, puzzle_type):
        """
        Stores the piece to piece distance

        Args:
            pieces ([SimplePuzzlePiece]): List of all puzzle pieces in simple form.s
            distance_function: Function to calculate the distance between two pieces.

        """

        # Store the number of pieces in the puzzle.
        self._piece_count = len(pieces)

        # store the distance function used for calculations.
        self._distance_function = distance_function

        # Store the puzzle type
        self._puzzle_type = puzzle_type

        # Initialize the data structures for this class.
        self._piece_distances = None
        self.best_buddies = None
        self._possible_start_pieces = None

        # Calculate the best buddies using the inter-distance information.
        self.find_best_buddies()

        # Find the set of valid starter pieces.
        self.find_start_piece_candidates()

    def calculate_inter_piece_distances(self, pieces):
        """
        Calculates the inter-piece distances between all pieces.

        Args:
            pieces ([PuzzlePiece]): All pieces across all puzzle(s).

        """
        # Based on the puzzle type, determine the number of possible piece to piece pairings.
        if InterPieceDistance.TYPE1_POSSIBLE_PAIRINGS:
            numb_possible_pairings = (self._puzzle_type == PuzzleType.type1)
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
                    if self._puzzle_type == PuzzleType.type1:
                        complimentary_side = x_i_side.complimentary_side()
                        dist = self._distance_function(pieces[x_i], x_i_side, pieces[x_j], complimentary_side)
                        self._piece_distances[x_i, x_j, x_i_side.value, 0] = dist

                    # For type two puzzles, handle all possible combinations of sides (16 in total).
                    if self._puzzle_type == PuzzleType.type2:
                        for x_j_side in PuzzlePieceSide.get_all_sides():
                            # Calculate the distance between the two pieces.
                            dist = self._distance_function(pieces[x_i], x_i_side, pieces[x_j], x_j_side)
                            self._piece_distances[x_i, x_j, x_i_side.value, x_j_side.value] = dist

    def find_best_buddies(self):
        """
        Finds the best buddies for this set of distance calculations.

        The best buddies information is stored with the interpiece distances.
        """

        # Can have a single best buddy per side.
        shape = self._piece_distances.shape
        best_distance = numpy.empty(shape[0], shape[2])

        # Find the best buddies for each piece on each side.
        for x_i in range(0, shape[0]):
            for side in PuzzlePieceSide.get_all_sides():  # Iterate through all the sides.

                # Special handle the first check
                first_check = True

                # Find the closest neighbor
                for y_i in range(0, shape[1]):
                    # A patch cannot be it own best friend so skip itself.
                    if x_i == y_i:
                        continue
                    # Check all possible pairings.  This is dependent on the type of possible
                    if self._puzzle_type == PuzzleType.type1:
                        all_other_sides = [side.complimentary_side()]
                    elif self._puzzle_type == PuzzleType.type2:
                        all_other_sides = PuzzlePieceSide.get_all_sides()
                    # noinspection PyUnboundLocalVariable
                    for other_side in all_other_sides:

                        # Check if the two pieces are the closest to each other.
                        if first_check or self._piece_distances[x_i, y_i, side, other_side.value] \
                                          < best_distance[x_i, y_i, side.value]:
                            # Store the best distance.
                            best_distance[x_i, side] = (self._piece_distances[x_i, y_i, side.value, other_side.value],
                                                        y_i, other_side)
                            first_check = False

        # Now that best distances have been found, check for best buddies.
        self.best_buddies = numpy.empty(shape[0], shape[1], shape[2])
        for x_i in range(0, shape[0]):
            for side in range(0, shape[2]):  # Iterate through all the sides.
                # bb = "Best Buddy"
                # Get the information on x_i's best buddy
                (_, x_bb, x_bb_side) = best_distance[x_i, side]
                # Get the information from the best buddy itself.
                (_, y_bb, y_bb_side) = best_distance[x_bb, side]

                # Check if we agreed on being best buddies
                if x_i == y_bb and side == y_bb_side:
                    self.best_buddies[x_i, side] = y_bb
                    self.best_buddies[y_bb, y_bb_side] = x_i

    def find_start_piece_candidates(self):
        """
        Creates a list of starter puzzle pieces.  This is based off the criteria defined by Paikin and Tal
        where a piece must have 4 best buddies and its best buddies must have 4 best buddies.

        This list is sorted from best starter piece to worst.
        """
        self._possible_start_pieces = []
        pass
