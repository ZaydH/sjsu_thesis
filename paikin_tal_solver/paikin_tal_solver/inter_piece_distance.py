"""Inter-Puzzle Piece Distance Object

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""
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
    _ALLOW_MULTIPLE_BEST_BUDDIES = False

    def __init__(self, id_numb, numb_pieces, puzzle_type):
        """

        Args:
            id_numb (int):
            numb_pieces (int): Number of pieces in the puzzle
            puzzle_type (PuzzleType):

        Returns (PieceDistanceInformation): Distance information object for a single puzzle piece.
        """
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
        if PieceDistanceInformation._ALLOW_MULTIPLE_BEST_BUDDIES:
            return self._best_buddy_candidates[side.value]
        else:
            # If there are more than one best buddy candidate, there are none.
            if len(self._best_buddy_candidates[side.value]) <= 1:
                return self._best_buddy_candidates[side.value]
            else:
                return []

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

    def clear_best_buddy_information(self):
        """
        Best Buddy Information Clearer

        Clears the best buddy candidates and best buddy list for a puzzle piece.
        """
        # Reset the piece's best buddy information
        self._best_buddy_candidates = [[] for _ in PuzzlePieceSide.get_all_sides()]
        self._best_buddies = [[] for _ in PuzzlePieceSide.get_all_sides()]

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

        # Reset the piece's important distance values.
        self._reset_min_and_second_best_distances()

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

                    # Update the minimum and second best distances as appropriate
                    self._update_min_and_second_best_distances_and_best_buddy_candidates(p_i_side,
                                                                                         p_j,
                                                                                         p_j_side,
                                                                                         update_best_buddy_candidates=True)

        # Calculate the Asymmetric Compatibilities
        self.calculate_asymmetric_compatibility()

    def _update_min_and_second_best_distances_and_best_buddy_candidates(self, p_i_side, p_j, p_j_side,
                                                                        update_best_buddy_candidates):
        """

        Args:
            p_i_side (PuzzlePieceSide): Reference side of the implicit puzzle piece
            p_j (int): Alternate piece identification number
            p_j_side (PuzzlePieceSide): Reference side of the other piece
        """

        # Extract the distance between p_i and p_j on their sides
        p_j_side_index = InterPieceDistance.get_p_j_side_index(self._puzzle_type, p_j_side)
        dist = self._asymmetric_distances[p_i_side.value, p_j, p_j_side_index]

        # See if the minimum distance needs to be updated.
        if dist < self._min_distance[p_i_side.value]:
            self._second_best_distance[p_i_side.value] = self._min_distance[p_i_side.value]
            self._min_distance[p_i_side.value] = dist

            if update_best_buddy_candidates:
                self._best_buddy_candidates[p_i_side.value] = [(p_j, p_j_side)]
        # See if there is a tie for best buddy
        elif dist == self._min_distance[p_i_side.value]:
            # TODO Decide later what to do about best buddy ties.
            self._second_best_distance[p_i_side.value] = dist

            if update_best_buddy_candidates:
                # noinspection PyTypeChecker
                self._best_buddy_candidates[p_i_side.value].append((p_j, p_j_side))
        # If only the second best then update the second best distance
        elif dist < self._second_best_distance[p_i_side.value]:
            self._second_best_distance[p_i_side.value] = dist

    def _reset_min_and_second_best_distances(self):
        """
        Minimum and Second Best Distance Resetter

        Resets the minimum and second best distances for an individual piece.
        """
        # Store the second best distances in an array
        self._second_best_distance = [sys.maxint for _ in range(0, PuzzlePieceSide.get_numb_sides())]

        # Use the second best distance to initialize a min best distance array.
        # It should be slightly less in value than the second best distance (e.g. subtract 1) since the best
        # distance is supposed to be the minimum.
        self._min_distance = [sys.maxint - 1 for i in range(0, PuzzlePieceSide.get_numb_sides())]

    def _find_min_and_second_best_distances(self, skip_piece):
        """
        Minimum and Second Best Distance Finder

        Finds the minimum and second best distance for a piece with respect to already calculated distances.

        Args:
            skip_piece ([Bool]): A list indicating whether each piece should be skipped.  If True, the piece should
            be skipped.  If the piece should be checked, then the array should be false.
        """

        # Reset the piece's distance information.
        self._reset_min_and_second_best_distances()

        # Select whether to update the best buddy candidates
        update_best_buddy_candidates = True if skip_piece is None else False

        # Go through all the valid sides
        for p_i_side in PuzzlePieceSide.get_all_sides():

            # Go through all other valid pieces.
            for p_j in range(0, self._numb_pieces):

                # Do not compare a piece to itself or if it is already placed
                if self._skip_piece(skip_piece, p_j):
                    continue

                # Check all valid p_j sides depending on the puzzle type.
                for p_j_side in InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type, p_i_side):

                    # Update the first and second best pieces as appropriate
                    self._update_min_and_second_best_distances_and_best_buddy_candidates(p_i_side,
                                                                                         p_j,
                                                                                         p_j_side,
                                                                                         update_best_buddy_candidates)

    def calculate_asymmetric_compatibility(self, is_piece_placed=None):
        """
        Asymmetric Compatibility Calculator

        Calculates the asymmetric compatibility for this piece with respect to all other pieces.

        Args:
            is_piece_placed (Optional [Bool]): For each puzzle piece, True if the piece is placed and false otherwise.
        """

        # Based on the puzzle type, determine the number of possible piece to piece pairings.
        numb_possible_pairings = len(InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type,
                                                                                 PuzzlePieceSide.get_all_sides()[0]))

        # Regenerate the numpy array only if it has not already been generated
        if self._asymmetric_compatibilities is None:
            # Build an empty array to store the piece to piece distances
            self._asymmetric_compatibilities = numpy.zeros((PuzzlePieceSide.get_numb_sides(), self._numb_pieces,
                                                           numb_possible_pairings), numpy.float32)
            self._asymmetric_compatibilities.fill(float('inf'))

        # Calculate the asymmetric compatibility
        for p_j in range(0, self._numb_pieces):

            # Do not compare a piece to itself or if it is already placed
            if self._skip_piece(is_piece_placed, p_j):
                continue

            for p_i_side in PuzzlePieceSide.get_all_sides():
                set_of_neighbor_sides = InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type, p_i_side)
                for p_j_side in set_of_neighbor_sides:
                    # Calculate the compatibility
                    p_j_side_index = InterPieceDistance.get_p_j_side_index(self._puzzle_type, p_j_side)

                    # Prevent divide by zero
                    second_best_distance = self._second_best_distance[p_i_side.value]
                    if self._asymmetric_distances[p_i_side.value, p_j, p_j_side_index] == 0:
                        asym_compatibility = 1
                    elif second_best_distance == 0:
                        asym_compatibility = -sys.maxint
                    else:
                        # Calculate the asymmetric compatibility
                        asym_compatibility = (1 - 1.0 * self._asymmetric_distances[p_i_side.value, p_j, p_j_side_index] /
                                              second_best_distance)
                    self._asymmetric_compatibilities[p_i_side.value, p_j, p_j_side_index] = asym_compatibility

        # Build an empty array to store the piece to piece distances
        self._mutual_compatibilities = numpy.zeros((PuzzlePieceSide.get_numb_sides(), self._numb_pieces,
                                                    numb_possible_pairings), numpy.float32)
        self._mutual_compatibilities.fill(float('inf'))

    def _skip_piece(self, skip_piece, p_j=None):
        """

        Args:
            skip_piece (Optional [Bool]):
            p_j (Optional int):

        Returns: True if this piece should be skipped and False otherwise.
        """
        if (p_j is not None and self._id == p_j) or \
                (skip_piece is not None and skip_piece[p_j]):  # Do not compare a piece to itself.
                return True
        else:
            return False

    @property
    def minimum_distance(self):
        """
        Minimum (Best) Distance Property

        For this piece's inter-piece distance information, this function is used to access the MINIMUM (i.e. best)
        distance between it and any other piece.

        Returns(float):
            Minimum (i.e. best) distance for this puzzle piece
        """
        return self._min_distance

    @property
    def second_best_distance(self):
        """
        Second Best Distance Property

        For this piece's interpiece distance information, this property returns the second best distance.

        Returns (float):
            Second best distance for this puzzle piece
        """
        return self._second_best_distance


class InterPieceDistance(object):
    """
    Master class for managing inter-puzzle piece distances as well as best buddies
    and the starter puzzle pieces as defined by the Paikin and Tal paper.
    """

    _PERFORM_ASSERT_CHECKS = True

    def __init__(self, pieces, distance_function, puzzle_type):
        """
        Inter-Puzzle Piece Distance Object Constructor

        Stores the piece to piece distance

        Args:
            pieces ([SimplePuzzlePiece]): List of all puzzle pieces in simple form.s
            distance_function: Function to calculate the distance between two pieces.
        """

        # Give each piece an identification number.
        id_numb = 0
        for piece in pieces:
            piece.id_number = id_numb
            id_numb += 1

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

        # Clear the distance function for pickling purposes
        self._distance_function = None

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

    def calculate_mutual_compatibility(self, is_piece_placed=None):
        """
        Mutual Compatibility Calculator

        Calculates the mutual compatibility as defined by Paikin and Tal.

        Args:
            is_piece_placed (Optional [Bool]): List indicating whether each piece is placed
        """
        for p_i in range(0, self._numb_pieces):

            # Go through all the valid sides
            for p_i_side in PuzzlePieceSide.get_all_sides():
                for p_j in range(p_i + 1, self._numb_pieces):

                    # Skip placed pieces
                    # No Need to check p_i == p_j since doing a diagonal calculation
                    if p_i == p_j or (InterPieceDistance._skip_piece(p_i, is_piece_placed)
                                      and InterPieceDistance._skip_piece(p_j, is_piece_placed)):
                        continue

                    # Check all valid p_j sides depending on the puzzle type.
                    for p_j_side in InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type, p_i_side):
                        # Get the compatibility from p_i to p_j
                        mutual_compat = self._piece_distance_info[p_i].asymmetric_compatibility(p_i_side, p_j,
                                                                                                p_j_side)
                        # Get the compatibility from p_j to p_i
                        mutual_compat += self._piece_distance_info[p_j].asymmetric_compatibility(p_j_side, p_i,
                                                                                                 p_i_side)
                        # Divide the mutual compatibility by 2.
                        mutual_compat /= 2

                        # Store the mutual compatibility for BOTH p_i and p_j
                        self._piece_distance_info[p_i].set_mutual_compatibility(p_i_side, p_j, p_j_side, mutual_compat)
                        self._piece_distance_info[p_j].set_mutual_compatibility(p_j_side, p_i, p_i_side, mutual_compat)

    def recalculate_all_compatibilities_and_best_buddy_info(self, is_piece_placed,
                                                            is_piece_placed_with_no_open_neighbors):
        """
        Comptability Recalculator

        When no best buddy is in the pool, this function is called to recalculate the best buddies and compatibilities.

        Args:
            is_piece_placed ([Bool]): List indicating whether each piece is placed

            is_piece_placed_with_no_open_neighbors ([Bool]): Subset of the elements in the array "is_piece_placed."  It
            is only those pieces that are placed AND have no open locations amongst their direct neighbors.
        """

        if InterPieceDistance._PERFORM_ASSERT_CHECKS:
            assert len(is_piece_placed) == len(is_piece_placed_with_no_open_neighbors)

        # # Clear the best buddy information for placed pieces
        # self._clear_placed_piece_best_buddy_information()

        # Find the minimum and second best distance information for the placed pieces
        pieces_with_changed_dist = self._find_min_and_second_best_distances(is_piece_placed,
                                                                            is_piece_placed)

        # Calculate the asymmetric compatibilities using the updated min and second best distances.
        self._recalculate_asymmetric_compatibilities(pieces_with_changed_dist, is_piece_placed_with_no_open_neighbors)

        # Recalculate the mutual probabilities
        self.calculate_mutual_compatibility(pieces_with_changed_dist)
        #
        # # Find the updated best buddies
        # self.find_best_buddies(is_piece_placed)
        #
        # # Find the starting pieces
        # self.find_start_piece_candidates(is_piece_placed)

    def _find_min_and_second_best_distances(self, is_piece_placed=None, is_piece_placed_with_no_open_neighbors=None):
        """

        Args:
            is_piece_placed (Optional [Bool]): List indicating whether each piece is placed
        """

        # Determine whether anything for this piece needs to be recalculated.
        min_or_second_best_distance_unchanged = [True] * len(is_piece_placed_with_no_open_neighbors)

        # Initialize the minimum distance to prevent warnings in PyCharm
        prev_min_dist = None
        prev_second_best_dist = None

        # Iterate through all of the pieces
        for p_i in range(0, self._numb_pieces):

            # Skip placed pieces
            if InterPieceDistance._skip_piece(p_i, is_piece_placed_with_no_open_neighbors):
                continue

            # If these change, it means asymmetric distance or mutual compatibility need to be recalculated
            if is_piece_placed is not None:
                prev_min_dist = self._piece_distance_info[p_i].minimum_distance
                prev_second_best_dist = self._piece_distance_info[p_i].second_best_distance

            # Find the minimum and second best distance for each piece and side
            # noinspection PyProtectedMember
            self._piece_distance_info[p_i]._find_min_and_second_best_distances(is_piece_placed)

            # Check if the distances changed
            if is_piece_placed is not None and (prev_min_dist != self._piece_distance_info[p_i].minimum_distance
                                                or prev_second_best_dist != self._piece_distance_info[p_i].second_best_distance):
                min_or_second_best_distance_unchanged[p_i] = False

        # Return those parts whose distance actually changed to speed up calculations.
        if is_piece_placed is not None:
            return min_or_second_best_distance_unchanged

    def _clear_placed_piece_best_buddy_information(self):
        """
        Best Buddy Information Clearer

        Clears the best buddy information for all placed pieces.
        """
        # Clears the best buddy information for placed pieces
        for p_i in range(0, self._numb_pieces):
            self._piece_distance_info[p_i].clear_best_buddy_information()

    def _recalculate_asymmetric_compatibilities(self, min_or_second_best_distance_unchanged,
                                                is_piece_placed_with_no_open_neighbors):
        """
        Asymmetric Compatibility Recalculator

        Recalculate the ASYMMETRIC compatibilities of unplaced pieces after minimum and second best distances were
        found.

        Args:
            min_or_second_best_distance_unchanged: (Optional [Bool]): List indicating whether each piece is placed
        """
        # Iterate through all pieces skipping placed ones.
        for p_i in range(0, self._numb_pieces):
            # Skip placed pieces
            if InterPieceDistance._skip_piece(p_i, min_or_second_best_distance_unchanged):
                continue

            # Recalculate the
            self._piece_distance_info[p_i].calculate_asymmetric_compatibility(is_piece_placed_with_no_open_neighbors)

    def find_best_buddies(self, is_piece_placed=None):
        """
        Finds the best buddies for this set of distance calculations.

        The best buddies information is stored with the inter-piece distances.

        Args:
            is_piece_placed (Optional [Bool]): List indicating whether each piece is placed
        """

        # Go through each piece to find its best buddies.
        for p_i in range(0, self._numb_pieces):

            # Skip placed pieces
            if InterPieceDistance._skip_piece(p_i, is_piece_placed):
                continue

            # Find the best buddies of all sides.
            for p_i_side in PuzzlePieceSide.get_all_sides():  # Iterate through all the sides.
                # Get p_i's best buddy candidates.
                best_buddy_candidates = self._piece_distance_info[p_i].best_buddy_candidates(p_i_side)
                # See if the candidates match
                for (p_j, p_j_side) in best_buddy_candidates:
                    piece_dist_info = self._piece_distance_info[p_j]
                    if (p_i, p_i_side) in piece_dist_info.best_buddy_candidates(p_j_side):
                        self._piece_distance_info[p_i].add_best_buddy(p_i_side, p_j, p_j_side)

    def find_start_piece_candidates(self, is_piece_placed=None):
        """
        Creates a list of starter puzzle pieces.  This is based off the criteria defined by Paikin and Tal
        where a piece must have 4 best buddies and its best buddies must have 4 best buddies.

        This list is sorted from best starter piece to worst.

        Args:
            is_piece_placed (Optional [Bool]): List indicating whether each piece is placed
        """

        # Calculate each pieces best buddy count for each piece
        all_best_buddy_info = []
        for p_i in range(0, self._numb_pieces):

            # Skip placed pieces
            if InterPieceDistance._skip_piece(p_i, is_piece_placed):
                all_best_buddy_info.append(([], 0))
                continue

            # Best buddies and compatibility storage.
            p_i_best_buddies_and_compat = []

            # Iterate through all sides of p_i
            for p_i_side in PuzzlePieceSide.get_all_sides():

                p_i_best_buddies_and_compat.append([])

                # Iterate through best buddies to pick the best one
                for (p_j, p_j_side) in self._piece_distance_info[p_i].best_buddies(p_i_side):
                    compatibility = self._piece_distance_info[p_i].get_mutual_compatibility(p_i_side, p_j, p_j_side)

                    # Use negative compatibility since we are using a reverse order sorting and this requires
                    # doing things in ascending order which negation does for me here.
                    p_i_best_buddies_and_compat[p_i_side.value].append((p_j, compatibility))

            # Extract the info on the best neighbors
            p_i_best_buddies = []
            avg_compatibility = 0
            for bb_and_compat in p_i_best_buddies_and_compat:

                # Check if no best buddies.  If so, exit.
                if not bb_and_compat:
                    continue

                # TODO Change the code to support multiple best buddies and to pick the best one.
                bestest_best_buddy_index = 0  # Out of all of the possible best buddies on this side, this is the best one
                p_i_best_buddies.append(bb_and_compat[bestest_best_buddy_index][0])  # Get p_j
                avg_compatibility += bb_and_compat[bestest_best_buddy_index][1]  # Get NEGATED mutual compatibility

            # Store the best neighbors list as well as the average distance
            if len(p_i_best_buddies) > 0:
                avg_compatibility /= len(p_i_best_buddies)
            else:
                avg_compatibility = 0
            all_best_buddy_info.append((p_i_best_buddies, avg_compatibility))

        # Build the best neighbor information
        self._start_piece_ordering = []
        for p_i in range(0, self._numb_pieces):

            # Skip placed pieces
            if InterPieceDistance._skip_piece(p_i, is_piece_placed):
                continue

            this_piece_bb_info = all_best_buddy_info[p_i]

            # Store the number of best buddy neighbors
            # Multiply by the number of sides here to prioritize direct neighbors in the case of a tie.
            numb_bb_neighbors = PuzzlePieceSide.get_numb_sides() * len(this_piece_bb_info[0])
            total_compatibility = this_piece_bb_info[1]

            # Include the neighbors info
            p_i_best_buddy_ids = this_piece_bb_info[0]
            for bb_id in p_i_best_buddy_ids:
                numb_bb_neighbors += len(all_best_buddy_info[bb_id][0])
                total_compatibility += all_best_buddy_info[bb_id][1]

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

    def asymmetric_compatibility(self, p_i, p_i_side, p_j, p_j_side):
        """
        Asymmetric Compatibility Accessor

        Returns the asymmetric compatibility for p_i's side (p_i_side) relative to p_j on its side p_j_side.

        Args:
            p_i (int): Primary piece for asymmetric distance
            p_i_side (PuzzlePieceSide): Side of the primary piece (p_i) where p_j will be placed
            p_j (int): Secondary piece for the asymmetric distance.
            p_j_side (PuzzlePieceSide): Side of the secondary piece (p_j) which is adjacent to p_i

        Returns (int): Asymmetric compatibility between puzzle pieces p_i and p_j.
        """
        # For a type 1 puzzles, ensure that the pu
        if InterPieceDistance._PERFORM_ASSERT_CHECKS:
            self.assert_valid_type1_side(p_i_side, p_j_side)

        return self._piece_distance_info[p_i].asymmetric_compatibility(p_i_side, p_j, p_j_side)

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

        p_i_mutual_compatibility = self._piece_distance_info[p_i].get_mutual_compatibility(p_i_side, p_j, p_j_side)

        # Verify for debug the mutual compatibility is symmetric.
        if InterPieceDistance._PERFORM_ASSERT_CHECKS:
            assert(p_i_mutual_compatibility == self._piece_distance_info[p_j].get_mutual_compatibility(p_j_side,
                                                                                                       p_i, p_i_side))
        # Return the mutual compatibility
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
            return [p_i_side.complementary_side]
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
            assert(p_i_side.complementary_side == p_j_side)

    @staticmethod
    def _skip_piece(p_i, is_piece_placed_with_no_open_neighbors):
        """
        Piece Skip Checker

        Checks whether a puzzle piece should be skipped based off whether it is placed.
        Args:
            p_i (int): Identification number of the puzzle piece
            is_piece_placed_with_no_open_neighbors ([Bool]):  List indicating whether each piece is placed

        Returns: True if piece p_i should be skipped and False otherwise
        """
        if is_piece_placed_with_no_open_neighbors is not None and is_piece_placed_with_no_open_neighbors[p_i]:
            return True
        else:
            return False
