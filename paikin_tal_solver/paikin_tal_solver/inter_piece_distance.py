import numpy
import sys

import operator

from hammoudeh_puzzle_solver.puzzle_importer import PuzzleType
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePieceSide


class PieceDistanceInformation(object):
    """
    Stores all of the inter-piece distance information (e.g. asymmetric distance, asymetric compatibility,
    mutual compatibility, etc.) between a specific piece (based off the ID number) and all other pieces.
    """

    _PERFORM_ASSERT_CHECKS = True

    def __init__(self, id_numb, numb_pieces, puzzle_type):
        self._id = id_numb
        self._numb_pieces = numb_pieces
        self._puzzle_type = puzzle_type

        self._min_distance = None
        self._second_best_distance = None

        self._asymmetric_distances = None
        self._asymmetric_compatibilities = None
        self._mutual_compatibilities = None

        # Define the best buddies information
        self._best_buddy_candidates = [[] for _ in PuzzlePieceSide.get_all_sides()]
        self._best_buddies = [[] for _ in PuzzlePieceSide.get_all_sides()]

    @property
    def piece_id(self):
        """
        Piece ID Accessor

        Gets the piece identification number for a PieceDistanceInformation object

        Returns (int): Piece identification number
        """
        return self._id

    def asymmetric_distance(self, p_i_side, p_j, p_j_side):
        """
        Asymmetric Distance Accessor

        Returns the asymmetric distance between p_i and p_j.

        Args:
            p_i_side (PuzzlePieceSide): Side of the primary piece (p_i - implicit)
            p_j (int): Secondary puzzle piece
            p_j_side (PuzzlePieceSide):

        Returns (int): Asymmetric distance between pieces p_i (implicit) and p_j (explicit) for their
        specified sides.
        """
        p_j_side_val = InterPieceDistance.get_p_j_side_index(self._puzzle_type, p_j_side)
        return self._asymmetric_distances[p_i_side.value, p_j, p_j_side_val]

    def asymmetric_compatibility(self, p_i_side, p_j, p_j_side):
        """
        Puzzle Piece Asymmetric Compatibility Accessor

        Gets the asymmetric compatibility for a piece (p_i) to another piece p_j on their respective sides.

        Args:
            p_i_side (PuzzlePieceSide): Side of the primary piece (p_i) where p_j will be placed
            p_j (int): Secondary piece for the asymmetric distance.
            p_j_side (PuzzlePieceSide): Side of the secondary piece (p_j) which is adjacent to p_i

        Returns: Asymmetric compatibility between the two pieces on their respective sides
        """
        p_j_side_val = InterPieceDistance.get_p_j_side_index(self._puzzle_type, p_j_side)
        return self._asymmetric_compatibilities[p_i_side.value, p_j, p_j_side_val]

    def set_mutual_compatibility(self, p_i_side, p_j, p_j_side, compatibility):
        """
        Puzzle Piece Mutual Compatibility Setter

        Sets the mutual compatibility for a piece (p_i) to another piece p_j on their respective sides.

        Args:
            p_i_side (PuzzlePieceSide): Side of the primary piece (p_i) where p_j will be placed
            p_j (int): Secondary piece for the asymmetric distance.
            p_j_side (PuzzlePieceSide): Side of the secondary piece (p_j) which is adjacent to p_i
            compatibility (int): Mutual compatibility between p_i and p_j on their respective sides.
        """
        p_j_side_val = InterPieceDistance.get_p_j_side_index(self._puzzle_type, p_j_side)
        self._mutual_compatibilities[p_i_side.value, p_j, p_j_side_val] = compatibility

    def get_mutual_compatibility(self, p_i_side, p_j, p_j_side):
        """
        Puzzle Piece Mutual Compatibility Accessor

        Gets the mutual compatibility for a piece (p_i) to another piece p_j on their respective sides.

        Args:
            p_i_side (PuzzlePieceSide): Side of the primary piece (p_i) where p_j will be placed
            p_j (int): Secondary piece for the asymmetric distance.
            p_j_side (PuzzlePieceSide): Side of the secondary piece (p_j) which is adjacent to p_i

        Returns: Mutual compatibility between the two pieces on their respective sides
        """
        p_j_side_val = InterPieceDistance.get_p_j_side_index(self._puzzle_type, p_j_side)
        # Return the mutual compatibility
        return self._mutual_compatibilities[p_i_side.value, p_j, p_j_side_val]

    def best_buddy_candidates(self, side):
        """
        Best Buddy Candidate Accessor

        Gets the list of possible best buddies for this set of distance information.

        Args:
            side (PuzzlePieceSide): Reference side of the puzzle piece

        Returns ([(int, int)]):
            Returns an array of the ID numbers and the respective side for the ID number for possible best buddies.
        """
        return self._best_buddy_candidates[side.value]

    def best_buddies(self, side):
        """
        Best Buddy Accessor

        Gets a list of best buddy piece ids for a puzzle piece's side.  If a puzzle piece side has no best
        buddy, this function returns an empty list.

        Args:
            side (PuzzlePieceSide): Side of the implicit puzzle piece.

        Returns ([int]): List of best buddy pieces
        """
        return self._best_buddies[side.value]

    def add_best_buddy(self, p_i_side, p_j_id_numb, p_j_side):
        """
        Best Buddy Puzzle Piece Adder

        Adds a best best buddy to a puzzle piece's side.

        Args:
            p_i_side (PuzzlePieceSide): Identification number for a side of the puzzle piece p_i
            p_j_id_numb (int): Identification number for the best buddy puzzle piece
            p_j_side (PuzzlePieceSide): Identification number for the side of p_j where p_i is placed
        """

        # Optionally check piece is not already a best buddy candidate on this side
        if PieceDistanceInformation._PERFORM_ASSERT_CHECKS:
            assert((p_j_id_numb, p_j_side) not in self._best_buddies[p_i_side.value])

        # Add the piece to the set of valid best buddies
        # noinspection PyTypeChecker
        self._best_buddies[p_i_side.value].append((p_j_id_numb, p_j_side))

    def calculate_inter_piece_distances(self, pieces, distance_function):
        """
        Calculates the inter-piece distances between all pieces.

        Args:
            pieces ([PuzzlePiece]): All pieces across all puzzle(s).
            distance_function: Function to measure the distance between two puzzle pieces

        """
        # Based on the puzzle type, determine the number of possible piece to piece pairings.
        numb_possible_pairings = len(InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type,
                                                                                 PuzzlePieceSide.get_all_sides()[0]))

        # Build an empty array to store the piece to piece distances
        self._asymmetric_distances = numpy.zeros((PuzzlePieceSide.get_numb_sides(), self._numb_pieces,
                                                 numb_possible_pairings), numpy.uint32)
        fill_value = 2 ** 31 - 1
        self._asymmetric_distances.fill(fill_value)

        # Store the second best distances in an array
        self._second_best_distance = [sys.float_info.max for _ in range(0, PuzzlePieceSide.get_numb_sides())]
        # Use the second best distance to initialize a min best distance array.
        # It should be slightly less in value than the second best distance (e.g. subtract 1
        self._min_distance = [self._second_best_distance[i] - 1 for i in range(0, PuzzlePieceSide.get_numb_sides())]

        # Calculates the piece to piece distances so we only need to do it once.
        for p_j in range(0, self._numb_pieces):
            if self._id == p_j:  # Do not compare a piece to itself.
                continue

            # Iterate through all of the sides of p_i (i.e. this piece)
            for p_i_side in PuzzlePieceSide.get_all_sides():

                # Define the set of valid sides of p_j where p_i can be placed for the given side
                set_of_neighbor_sides = InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type, p_i_side)

                # Go through the set of x_j sides
                for p_j_side in set_of_neighbor_sides:

                    # Calculate the distance between the two pieces.
                    dist = distance_function(pieces[self._id], p_i_side, pieces[p_j], p_j_side)

                    # Store the distance
                    p_j_side_index = InterPieceDistance.get_p_j_side_index(self._puzzle_type, p_j_side)
                    self._asymmetric_distances[p_i_side.value, p_j, p_j_side_index] = dist

                    # Update the second best distance if applicable
                    if dist < self._min_distance[p_i_side.value]:
                        self._second_best_distance[p_i_side.value] = self._min_distance[p_i_side.value]
                        self._min_distance[p_i_side.value] = dist
                        self._best_buddy_candidates[p_i_side.value] = [(p_j, p_j_side)]
                    # See if there is a tie for best buddy
                    elif dist == self._min_distance[p_i_side.value]:
                        # noinspection PyTypeChecker
                        self._best_buddy_candidates[p_i_side.value].append((p_j, p_j_side))
                        self._second_best_distance[p_i_side.value] = dist
                    # If only the second best then update the second best distance
                    elif dist < self._second_best_distance[p_i_side.value]:
                        self._second_best_distance[p_i_side.value] = dist

        # Build an empty array to store the piece to piece distances
        self._asymmetric_compatibilities = numpy.zeros((PuzzlePieceSide.get_numb_sides(), self._numb_pieces,
                                                       numb_possible_pairings), numpy.float32)
        self._asymmetric_compatibilities.fill(float('inf'))

        # Calculate the asymmetric compatibility
        for p_j in range(0, self._numb_pieces):
            if self._id == p_j:  # Do not compare a piece to itself.
                continue
            for p_i_side in PuzzlePieceSide.get_all_sides():
                set_of_neighbor_sides = InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type, p_i_side)
                for p_j_side in set_of_neighbor_sides:
                    # Calculate the compatibility
                    p_j_side_index = InterPieceDistance.get_p_j_side_index(self._puzzle_type, p_j_side)
                    asym_compatibility = (1 - 1.0 * self._asymmetric_distances[p_i_side.value, p_j, p_j_side_index] /
                                          self._second_best_distance[p_i_side.value])
                    self._asymmetric_compatibilities[p_i_side.value, p_j, p_j_side_index] = asym_compatibility

        # Build an empty array to store the piece to piece distances
        self._mutual_compatibilities = numpy.zeros((PuzzlePieceSide.get_numb_sides(), self._numb_pieces,
                                                    numb_possible_pairings), numpy.float32)
        fill_value = 2 ** 31 - 1
        self._asymmetric_compatibilities.fill(float('inf'))


class InterPieceDistance(object):
    """
    Master class for managing inter-puzzle piece distances as well as best buddies
    and the starter puzzle pieces as defined by the Paikin and Tal paper.
    """

    _PERFORM_ASSERT_CHECKS = True

    def __init__(self, pieces, distance_function, puzzle_type):
        """
        Stores the piece to piece distance

        Args:
            pieces ([SimplePuzzlePiece]): List of all puzzle pieces in simple form.s
            distance_function: Function to calculate the distance between two pieces.

        """

        # Store the number of pieces in the puzzle.
        self._numb_pieces = len(pieces)

        # store the distance function used for calculations.
        assert(distance_function is not None)
        self._distance_function = distance_function

        # Store the puzzle type
        self._puzzle_type = puzzle_type

        # Initialize the data structures for this class.
        self._piece_distance_info = []
        for p_i in range(0, self._numb_pieces):
            self._piece_distance_info.append(PieceDistanceInformation(p_i, self._numb_pieces, self._puzzle_type))

        # Define the start piece ordering
        self._start_piece_ordering = []

        # Calculate the best buddies using the inter-distance information.
        self.calculate_inter_piece_distances(pieces)

        # Calculate the piece to piece mutual compatibility
        self.calculate_mutual_compatibility()

        # Calculate the best buddies using the inter-distance information.
        self.find_best_buddies()

        # Find the set of valid starter pieces.
        self.find_start_piece_candidates()

    def calculate_inter_piece_distances(self, pieces):
        """
        Inter-Piece Distance Calculator

        Calculates the inter-piece distances between all pieces.  Also calculates the compatibility between
        two parts.

        Args:
            pieces ([PuzzlePiece]): All pieces across all puzzle(s).

        """

        # Calculates the piece to piece distances so we only need to do it once.
        for p_i in range(0, self._numb_pieces):
            self._piece_distance_info[p_i].calculate_inter_piece_distances(pieces, self._distance_function)

    def calculate_mutual_compatibility(self):
        """
        Mutual Compatibility Calculator

        Calculates the mutual compatibility as defined by Paikin and Tal.
        """
        for p_i in range(0, self._numb_pieces):
            for p_i_side in PuzzlePieceSide.get_all_sides():
                for p_j in range(p_i + 1, self._numb_pieces):
                    for p_j_side in InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type, p_i_side):
                        # Get the compatibility from p_i to p_j
                        p_j_side_index = InterPieceDistance.get_p_j_side_index(self._puzzle_type, p_j_side)
                        mutual_compat = self._piece_distance_info[p_i].asymmetric_compatibility(p_i_side, p_j,
                                                                                                p_j_side)
                        # Get the compatibility from p_j to p_i
                        p_i_side_index = InterPieceDistance.get_p_j_side_index(self._puzzle_type, p_i_side)
                        mutual_compat += self._piece_distance_info[p_j].asymmetric_compatibility(p_j_side, p_i,
                                                                                                 p_i_side)

                        # Store the mutual compatibility for BOTH p_i and p_j
                        self._piece_distance_info[p_i].set_mutual_compatibility(p_i_side, p_j, p_j_side, mutual_compat)
                        self._piece_distance_info[p_j].set_mutual_compatibility(p_j_side, p_i, p_i_side, mutual_compat)

    def find_best_buddies(self):
        """
        Finds the best buddies for this set of distance calculations.

        The best buddies information is stored with the inter-piece distances.
        """

        # Go through each piece to find its best buddies.
        for p_i in range(0, self._numb_pieces):
            for p_i_side in PuzzlePieceSide.get_all_sides():  # Iterate through all the sides.
                # Get p_i's best buddy candidates.
                best_buddy_candidates = self._piece_distance_info[p_i].best_buddy_candidates(p_i_side)
                # See if the candidates match
                for (p_j, p_j_side) in best_buddy_candidates:
                    piece_dist_info = self._piece_distance_info[p_j]
                    if (p_i, p_i_side) in piece_dist_info.best_buddy_candidates(p_j_side):
                        self._piece_distance_info[p_i].add_best_buddy(p_i_side, p_j, p_j_side)

    def find_start_piece_candidates(self):
        """
        Creates a list of starter puzzle pieces.  This is based off the criteria defined by Paikin and Tal
        where a piece must have 4 best buddies and its best buddies must have 4 best buddies.

        This list is sorted from best starter piece to worst.
        """

        # Calculate each pieces best buddy count for each piece
        all_best_buddy_info = []
        for p_i in range(0, self._numb_pieces):

            side_best_dist = []

            # Iterate through all sides of p_i
            for p_i_side_cnt in range(0, len(PuzzlePieceSide.get_all_sides())):
                p_i_side = PuzzlePieceSide.get_all_sides()[p_i_side_cnt]
                # noinspection PyTypeChecker
                side_best_dist.append(None)
                # Iterate through best buddies to pick the best one
                # TODO Change the code to support multiple best buddies
                for (p_j, p_j_side) in self._piece_distance_info[p_i].best_buddies(p_i_side):
                    compatibility = self._piece_distance_info[p_i].get_mutual_compatibility(p_i_side, p_j, p_j_side)

                    # Use negative compatibility since we are using a reverse order sorting and this requires
                    # doing things in ascending order which negation does for me here.
                    side_best_dist[p_i_side_cnt] = (p_j, -compatibility)
                    break

            # Extract the info on the best neighbors
            best_neighbor_list = []
            avg_distance = 0
            for side_info in side_best_dist:
                if side_info is None:
                    continue
                best_neighbor_list.append(side_info[0])
                avg_distance += side_info[1]
            # Store the best neighbors list as well as the average distance
            if len(best_neighbor_list) > 0:
                avg_distance /= len(best_neighbor_list)
            else:
                avg_distance = 0
            all_best_buddy_info.append((best_neighbor_list, avg_distance))

        # Build the best neighbor information
        self._start_piece_ordering = []
        for p_i in range(0, self._numb_pieces):
            this_piece_bb_info = all_best_buddy_info[p_i]

            # Store the number of best buddy neighbors
            # Multiply by the number of sides here to prioritize direct neighbors in the case of a tie.
            numb_bb_neighbors = PuzzlePieceSide.get_numb_sides() * len(this_piece_bb_info[0])
            total_compatibility = this_piece_bb_info[1]

            # Include the neighbors info
            for best_buddy_id in this_piece_bb_info[0]:
                bb_info = all_best_buddy_info[best_buddy_id]
                numb_bb_neighbors += len(bb_info[0])
                total_compatibility += bb_info[1]

            # Add this pieces information to the list of possible start pieces
            self._start_piece_ordering.append((p_i, numb_bb_neighbors, total_compatibility))

        # Sort by number of best buddy neighbors (1) then by total compatibility if there is a tie (2)
        # See here for more information: http://stackoverflow.com/questions/4233476/sort-a-list-by-multiple-attributes
        self._start_piece_ordering.sort(key=operator.itemgetter(1, 2), reverse=True)

    def next_starting_piece(self, placed_pieces=None):
        """
        Next Starting Piece Accessor

        Gets the puzzle piece that is the best candidate to use as the seed of a puzzle.

        Args:
            placed_pieces (Optional [bool]): An array indicating whether each puzzle piece (by index) has been
            placed.

        Returns (int): Index of the next piece to use for starting a board.
        """
        # If no pieces are placed, then use the first piece
        if placed_pieces is None:
            return self._start_piece_ordering[0][0]

        # If some pieces are already placed, ensure that you do not use a placed piece as the
        # next seed.
        else:
            i = 0
            while placed_pieces[self._start_piece_ordering[i][0]]:
                i += 1
            return self._start_piece_ordering[i][0]

    def best_buddies(self, p_i, p_i_side):
        """

        Args:
            p_i (int):
            p_i_side  (PuzzlePieceSide):

        Returns ([int]): List of best buddy piece id numbers
        """
        return self._piece_distance_info[p_i].best_buddies(p_i_side)

    def asymmetric_distance(self, p_i, p_i_side, p_j, p_j_side):
        """
        Asymmetric Distance Accessor

        Returns the asymmetric distance for p_i's side (p_i_side) relative to p_j on its side p_j_side.

        Args:
            p_i (int): Primary piece for asymmetric distance
            p_i_side (PuzzlePieceSide): Side of the primary piece (p_i) where p_j will be placed
            p_j (int): Secondary piece for the asymmetric distance.
            p_j_side (PuzzlePieceSide): Side of the secondary piece (p_j) which is adjacent to p_i

        Returns (int): Asymmetric distance between puzzle pieces p_i and p_j.
        """
        # For a type 1 puzzles, ensure that the pu
        if InterPieceDistance._PERFORM_ASSERT_CHECKS:
            self.assert_valid_type1_side(p_i_side, p_j_side)
        return self._piece_distance_info[p_i].asymmetric_distance(p_i_side, p_j, p_j_side)

    def mutual_compatibility(self, p_i, p_i_side, p_j, p_j_side):
        """
        Mutual Compatibility Accessor

        Returns the mutual compatibility for p_i's side (p_i_side) relative to p_j on its side p_j_side.

        Args:
            p_i (int): Primary piece for asymmetric distance
            p_i_side (PuzzlePieceSide): Side of the primary piece (p_i) where p_j will be placed
            p_j (int): Secondary piece for the asymmetric distance.
            p_j_side (PuzzlePieceSide): Side of the secondary piece (p_j) which is adjacent to p_i

        Returns (int): Mutual compatibility between puzzle pieces p_i and p_j.
        """
        if InterPieceDistance._PERFORM_ASSERT_CHECKS:
            self.assert_valid_type1_side(p_i_side, p_j_side)

        p_i_mutual_compatibility = self._piece_distance_info[p_i].mutual_compatibility(p_i_side, p_j, p_j_side)

        # Verify for debug the mutual compatibility is symmetric.
        if InterPieceDistance._PERFORM_ASSERT_CHECKS:
            assert(p_i_mutual_compatibility == self._piece_distance_info[p_j].mutual_compatibility(p_j_side, p_i, p_i_side))

        return p_i_mutual_compatibility

    @staticmethod
    def get_valid_neighbor_sides(puzzle_type, p_i_side):
        """
        Valid Puzzle Piece Determiner

        For a tuple of puzzle_type and puzzle piece side, this function determins the set of valid PuzzlePieceSide
        for any neighboring piece.

        For example, if the puzzle is type 1, only complementary sides can be placed adjacent to one another.  In
        contrast, if the puzzle is type 2, then any puzzle piece side can be placed adjacent.

        Args:
            puzzle_type (PuzzleType): Puzzle type being solved.
            p_i_side (PuzzlePieceSide): Side of p_i puzzle piece where p_j will be placed.

        Returns ([PuzzlePieceSide]): List of all valid sides for a neighboring puzzle piece.
        """
        if puzzle_type == PuzzleType.type1:
            return [p_i_side.complementary_side()]
        else:
            return PuzzlePieceSide.get_all_sides()

    @staticmethod
    def get_p_j_side_index(puzzle_type, p_j_side):
        """
        Secondary Piece Side Index Lookup

        Args:
            puzzle_type (PuzzleType): Either type1 (no rotation) or type 2 (with rotation)
            p_j_side (PuzzlePieceSide): Side for the secondary piece p_j.

        Returns: For type 1 puzzles, this normalizes to an index of 0 since it is the only distance for two puzzle
        pieces on a given side of the primary piece.  For type 2 puzzles, the index is set to the p_j_side value defined
        in the PuzzlePieceSide enumerated type.
        """
        if puzzle_type == PuzzleType.type1:
            return 0
        else:
            return p_j_side.value

    def assert_valid_type1_side(self, p_i_side, p_j_side):
        """
        Valid Side Checker

        For type 1 puzzles, this function is used to verify that two puzzle piece sides are a valid pair.

        Args:
            p_i_side (PuzzlePieceSide): Side of the primary piece (p_i) where p_j will be placed.
            p_j_side (PuzzlePieceSide): Side of the secondary piece (p_j) where p_i will be placed.
        """
        if self._puzzle_type == PuzzleType.type1:
            assert(p_i_side.complementary_side() == p_j_side)
