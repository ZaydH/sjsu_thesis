"""Inter-Puzzle Piece Distance Object

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""
import numpy as np
import sys
import math  # Used for ceiling function
import copy

import operator
import multiprocessing as mp
import logging

import time

from hammoudeh_puzzle.puzzle_importer import PuzzleType
from hammoudeh_puzzle.puzzle_piece import PuzzlePieceSide
from hammoudeh_puzzle.solver_helper_classes import print_elapsed_time


class PieceDistanceInformation(object):
    """
    Stores all of the inter-piece distance information (e.g. asymmetric distance, asymmetric compatibility,
    mutual compatibility, etc.) between a specific piece (based off the ID number) and all other pieces.
    """

    _PERFORM_ASSERT_CHECKS = True
    _ALLOW_MULTIPLE_BEST_BUDDIES = False

    _ENABLE_AVERAGE_CHECK_SPEED_UP = False
    _MAXIMUM_AVERAGE_SEPARATION = 20

    def __init__(self, id_numb, numb_pieces, puzzle_type):
        """

        Args:
            id_numb (int):
            numb_pieces (int): Number of pieces in the puzzle
            puzzle_type (PuzzleType):

        Returns (PieceDistanceInformation):
            Distance information object for a single puzzle piece.
        """
        self._id = id_numb
        self._numb_pieces = numb_pieces
        self._puzzle_type = puzzle_type

        self._min_distance = None
        self._second_best_distance = None

        self._asymmetric_distances = None
        self._asymmetric_compatibilities = None

        self._mutual_compatibilities = None
        self._initial_mutual_compatibilities = None
        self._mutual_compatibility_has_changed = False

        # Define the best buddies information
        self._best_buddy_candidates = [[] for _ in PuzzlePieceSide.get_all_sides()]
        self._best_buddies = [[] for _ in PuzzlePieceSide.get_all_sides()]

    @property
    def piece_id(self):
        """
        Piece ID Accessor

        Gets the piece identification number for a PieceDistanceInformation object

        Returns (int):
            Piece identification number
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

        Returns (int):
            Asymmetric distance between pieces p_i (implicit) and p_j (explicit) for their specified sides.
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

        Returns (double):
            Asymmetric compatibility between the two pieces on their respective sides
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
            compatibility (float): Mutual compatibility between p_i and p_j on their respective sides.
        """
        p_j_side_val = InterPieceDistance.get_p_j_side_index(self._puzzle_type, p_j_side)
        self._mutual_compatibilities[p_i_side.value, p_j, p_j_side_val] = compatibility

        # This indicates whether mutual compatibilities need to be replaced on a restore or the original
        # values can be used.
        self._mutual_compatibility_has_changed = True

    def get_mutual_compatibility(self, p_i_side, p_j, p_j_side):
        """
        Puzzle Piece Mutual Compatibility Accessor

        Gets the mutual compatibility for a piece (p_i) to another piece p_j on their respective sides.

        Args:
            p_i_side (PuzzlePieceSide): Side of the primary piece (p_i) where p_j will be placed
            p_j (int): Secondary piece for the asymmetric distance.
            p_j_side (PuzzlePieceSide): Side of the secondary piece (p_j) which is adjacent to p_i

        Returns (double):
            Mutual compatibility between the two pieces on their respective sides
        """
        p_j_side_val = InterPieceDistance.get_p_j_side_index(self._puzzle_type, p_j_side)
        # Return the mutual compatibility
        return self._mutual_compatibilities[p_i_side.value, p_j, p_j_side_val]

    def store_initial_mutual_compatibility(self):
        """
        Store the initial mutual compatibilities for reuse later if needed.
        """
        self._initial_mutual_compatibilities = copy.deepcopy(self._mutual_compatibilities)

        # Stored so no need to reset.
        self._mutual_compatibility_has_changed = False

    def restore_initial_mutual_compatibility(self):
        """
        In cases where mutual compatibility may have changed, this function restores the initial mutual compatibilities.
        """
        if self._mutual_compatibility_has_changed:
            self._mutual_compatibilities = copy.deepcopy(self._initial_mutual_compatibilities)
            self._mutual_compatibility_has_changed = False

    def best_buddy_candidates(self, side):
        """
        Best Buddy Candidate Accessor

        Gets the list of possible best buddies for this set of distance information.

        Args:
            side (PuzzlePieceSide): Reference side of the puzzle piece

        Returns (List[ Tuple[int] ]):
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

        Returns ([int]):
            List of best buddy pieces
        """
        return self._best_buddies[side.value]

    def all_best_buddies(self):
        """
        Best Buddy Accessor

        Gets all the best buddies of a piece

        Returns (List[int]): List of best buddy pieces
        """
        return self._best_buddies

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
        self._asymmetric_distances = np.full((PuzzlePieceSide.get_numb_sides(), self._numb_pieces, numb_possible_pairings),
                                             fill_value=sys.maxint, dtype=np.uint32)
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

                    # See if the calculation can be skipped.
                    p_i_avg_color = pieces[self._id].border_average_color(p_i_side)
                    p_j_avg_color = pieces[p_j].border_average_color(p_j_side)
                    # If the pieces are too different, the calculations can be sped-
                    if (PieceDistanceInformation._ENABLE_AVERAGE_CHECK_SPEED_UP
                            and abs(p_i_avg_color - p_j_avg_color) >= PieceDistanceInformation._MAXIMUM_AVERAGE_SEPARATION):
                        dist = sys.maxint
                    else:
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
        For a given puzzle piece's side, this function will check if the other piece and its side is it
        best or second best distance.

        Optionally, this function will also store possible best buddy candidates for best buddy analysis.

        Args:
            p_i_side (PuzzlePieceSide): Reference side of the implicit puzzle piece
            p_j (int): Alternate piece identification number
            p_j_side (PuzzlePieceSide): Reference side of the other piece
            update_best_buddy_candidates (bool): If True, potentially update the best buddy information.
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
                # Append because of a tie with two pieces for best distance
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
        self._min_distance = [sys.maxint - 1 for _ in range(0, PuzzlePieceSide.get_numb_sides())]

    def _find_min_and_second_best_distances(self, piece_valid_for_placement):
        """
        Minimum and Second Best Distance Finder

        Finds the minimum and second best distance for a piece with respect to already calculated distances.

        Args:
            piece_valid_for_placement ([Bool]): A list indicating whether each piece can be used for placement.
                If True for any piece, then the piece is used.  If False, the piece is skipped.
        """

        # Reset the piece's distance information.
        self._reset_min_and_second_best_distances()

        # # Select whether to update the best buddy candidates
        # update_best_buddy_candidates = True if piece_valid_for_placement is None else False

        # Go through all the valid sides
        for p_i_side in PuzzlePieceSide.get_all_sides():

            # Go through all other valid pieces.
            for p_j in range(0, self._numb_pieces):

                # Do not compare a piece to itself or if it is already placed
                if self._skip_piece(piece_valid_for_placement, p_j):
                    continue

                # Check all valid p_j sides depending on the puzzle type.
                for p_j_side in InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type, p_i_side):

                    # Update the first and second best pieces as appropriate
                    self._update_min_and_second_best_distances_and_best_buddy_candidates(p_i_side,
                                                                                         p_j,
                                                                                         p_j_side,
                                                                                         update_best_buddy_candidates=False)

    def calculate_asymmetric_compatibility(self, is_piece_valid_for_placement=None):
        """
        Asymmetric Compatibility Calculator

        Calculates the asymmetric compatibility for this piece with respect to all other pieces.

        Args:
            is_piece_valid_for_placement (List[Bool]): For each puzzle piece, indicate whether it is valid for placement.
        """

        # Based on the puzzle type, determine the number of possible piece to piece pairings.
        numb_possible_pairings = len(InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type,
                                                                                 PuzzlePieceSide.get_all_sides()[0]))

        # Regenerate the np array only if it has not already been generated
        if self._asymmetric_compatibilities is None:
            # Build an empty array to store the piece to piece distances
            self._asymmetric_compatibilities = np.full((PuzzlePieceSide.get_numb_sides(), self._numb_pieces, numb_possible_pairings),
                                                       fill_value=np.finfo(np.float32).max,
                                                       dtype=np.float32)

        # Calculate the asymmetric compatibility
        for p_j in range(0, self._numb_pieces):

            # Do not compare a piece to itself or if it is already placed
            if self._skip_piece(is_piece_valid_for_placement, p_j):
                continue

            for p_i_side in PuzzlePieceSide.get_all_sides():
                set_of_neighbor_sides = InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type, p_i_side)
                for p_j_side in set_of_neighbor_sides:
                    # Calculate the compatibility
                    p_j_side_index = InterPieceDistance.get_p_j_side_index(self._puzzle_type, p_j_side)

                    # Get the parameters need to either specially set or calculate asymmetric compatibility
                    piece_to_piece_distance = self._asymmetric_distances[p_i_side.value, p_j, p_j_side_index]
                    second_best_distance = self._second_best_distance[p_i_side.value]

                    # Check if calculations can be skipped
                    if piece_to_piece_distance == sys.maxint:
                        asym_compatibility = -sys.maxint
                    elif second_best_distance == 0:
                        asym_compatibility = -sys.maxint
                    # Prevent divide by zero
                    elif piece_to_piece_distance == 0:
                        asym_compatibility = 1
                    else:
                        # Calculate the asymmetric compatibility
                        asym_compatibility = (1 - 1.0 * piece_to_piece_distance / second_best_distance)
                    self._asymmetric_compatibilities[p_i_side.value, p_j, p_j_side_index] = asym_compatibility

        # Build an empty array to store the piece to piece distances
        self._mutual_compatibilities = np.full((PuzzlePieceSide.get_numb_sides(), self._numb_pieces, numb_possible_pairings),
                                               fill_value=np.finfo(np.float32).max, dtype=np.float32)

    def _skip_piece(self, piece_valid_for_placement, p_j=None):
        """
        Determine if a piece should be skilled during placement.  A piece is skipped if the piece being compared
        is the same as the implicit piece or if the

        Args:
            piece_valid_for_placement (List[bool]): For each piece in the puzzle, True represents the piece
                is valid to be placed and False means the piece's placement is disallowed or the piece is already
                placed.

            p_j (int): Identification number of the piece being compared to the implicit piece

        Returns (bool):
            True if this piece should be skipped and False otherwise.
        """

        if (p_j is not None and self._id == p_j) \
                or (piece_valid_for_placement is not None and not piece_valid_for_placement[p_j]):
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

        For this piece's inter-piece distance information, this property returns the second best distance.

        Returns (float):
            Second best distance for this puzzle piece
        """
        return self._second_best_distance


class InterPieceDistance(object):
    """
    Master class for managing inter-puzzle piece distances as well as best buddies
    and the starter puzzle pieces as defined by the Paikin and Tal paper.
    """

    # If true, assertion checks are run.  Optional to increase algorithm stability and speedup runtime.
    _PERFORM_ASSERT_CHECKS = True

    # These are used for picking the starting piece.
    _USE_ONLY_NEIGHBORS_FOR_STARTING_PIECE_TOTAL_COMPATIBILITY = True
    _NEIGHBOR_COMPATIBILITY_SCALAR = 4  # If using only neighbors for start piece, this value does not matter if >0

    # Items related to multi-process computation to improve performance
    _MIN_NUMBER_PIECES_PER_THREAD = 10
    _MAX_NUMBER_OF_PARALLEL_PROCESSES = 5
    _USE_MULTIPLE_PROCESSES = True

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
        self._initial_start_piece_ordering = copy.deepcopy(self._start_piece_ordering)
        self._reinitialize_start_piece_order = False

        # Clear the distance function for pickling purposes
        self._distance_function = None

    def restore_initial_distance_values(self):
        """
        Resets the piece distance data structure as it was when it was first built.  Hence, it overwrites any changes
        made during the placement process.
        """
        for dist_info in self._piece_distance_info:
            dist_info.restore_initial_mutual_compatibility()

        # Optionally reinitialize start piece ordering.
        if self._reinitialize_start_piece_order:
            self._start_piece_ordering = copy.deepcopy(self._initial_start_piece_ordering)
        self._reinitialize_start_piece_order = False

    def invalidate_start_piece_ordering(self):
        """
        Marks that the start pieces need to be recalculated.
        """
        self._reinitialize_start_piece_order = True

    def calculate_inter_piece_distances(self, pieces):
        """
        Inter-Piece Distance Calculator

        Calculates the inter-piece distances between all pieces.  Also calculates the compatibility between
        two parts.

        Args:
            pieces ([PuzzlePiece]): All pieces across all puzzle(s).

        """

        start_time = time.time()
        logging.info("Starting inter-piece distance calculations.")

        # If no speed up benefit then do the operations serially
        if self._numb_pieces < InterPieceDistance._MIN_NUMBER_PIECES_PER_THREAD \
                or not InterPieceDistance._USE_MULTIPLE_PROCESSES:

            # Calculates the piece to piece distances so we only need to do it once.
            for p_i in range(0, self._numb_pieces):
                self._piece_distance_info[p_i].calculate_inter_piece_distances(pieces, self._distance_function)

        # Calculate distance in a multi-process environment for improved throughput
        else:

            # Do not always use maximum number of threads.  Base on the workload.
            numb_processes = self._calculate_numb_parallel_calculation_processes()
            max_elements_per_thread = long(math.ceil(1.0 * self._numb_pieces / numb_processes))

            # Populate the data for the thread pool
            thread_elements = []
            for i in xrange(0, numb_processes):
                distance_calc_element = {"first": i * max_elements_per_thread,
                                         "last": min((i+1) * max_elements_per_thread, self._numb_pieces),  # Exclusive
                                         "all_pieces": pieces,  # Piece distance information
                                         "puzzle_type": self._puzzle_type,
                                         "distance_function": self._distance_function}
                thread_elements.append(distance_calc_element)

            # Build the thread pool
            process_pool = mp.Pool(numb_processes)
            calculated_distances = process_pool.map(_multiprocess_interpiece_distances_calc,
                                                    thread_elements)
            process_pool.close()
            process_pool.join()

            # Merge the data back into the main structure
            for p_i in xrange(0, self._numb_pieces):
                piece_dist = calculated_distances[p_i // max_elements_per_thread][p_i % max_elements_per_thread]
                self._piece_distance_info[p_i] = piece_dist

        logging.info("Inter-distance calculation completed.")
        print_elapsed_time(start_time, "inter-piece distance calculation")

    @staticmethod
    def calculate_elements_per_process_for_diagonal_matrix(numb_pieces, numb_processes):
        """
        The calculation of the data for the inter-piece distance diagonal matrix can be spread across multiple
        processes.  This function enumerates how many elements to assign to the different processes to ensure each
        process is assigned a comparable amount of work to reduce the overall execution time.

        The design of this function uses the concept that the number of elements in a diagonal matrix is essentially
        the same as calculating the area of a triangle.

        Args:
            numb_pieces (int): Number of pieces in the puzzle
            numb_processes (int): Number of processes to be spawned.

        Returns (List(int)):
            Number of the first element that will be calculated for each of the processes.
        """

        if InterPieceDistance._PERFORM_ASSERT_CHECKS:
            assert numb_pieces >= 1 and numb_processes >= 1

        # Stores the first element to be processed for each process
        first_element_for_each_process = [0]

        for i in xrange(1, numb_processes):
            percentage_of_area = 1.0 * i / numb_processes
            first_element = int(numb_pieces * math.sqrt(percentage_of_area))
            first_element_for_each_process.append(first_element)

        return first_element_for_each_process

    def get_total_best_buddy_count(self):
        """
        Get the total number of best buddies.

        This function is mostly used for assert checking.

        Returns (int): Total number of best buddies
        """
        bb_total_count = 0
        # Iterate through all pieces
        for piece_dist_info in self._piece_distance_info:
            # iterate though the side of each piece.
            for side in PuzzlePieceSide.get_all_sides():
                bb_total_count += len(piece_dist_info.best_buddies(side))
        return bb_total_count

    def _calculate_mutual_compatibility_single_process(self, is_piece_valid_for_placement=None):
        """
        Single Process Mutual Compatibility Calculator

        Calculates the mutual compatibility as defined by Paikin and Tal.

        <b>Note:</b> This is done in a single process only.  Hence, it operates on the results data structures directly.

        Args:
            is_piece_valid_for_placement (List[bool]): List indicating whether each piece is valid for placement
                or should be ignored.
        """
        for p_i in range(0, self._numb_pieces):

            # Go through all the valid sides
            for p_i_side in PuzzlePieceSide.get_all_sides():
                for p_j in range(p_i + 1, self._numb_pieces):

                    # Skip placed pieces
                    # No Need to check p_i == p_j since doing a diagonal calculation
                    skip_piece = InterPieceDistance.skip_piece(p_i, is_piece_valid_for_placement) \
                                 and InterPieceDistance.skip_piece(p_j, is_piece_valid_for_placement)
                    if p_i == p_j or skip_piece:
                        continue

                    # Check all valid p_j sides depending on the puzzle type.
                    for p_j_side in InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type, p_i_side):

                        p_i_to_p_j = self._piece_distance_info[p_i].asymmetric_compatibility(p_i_side, p_j,
                                                                                             p_j_side)
                        p_j_to_p_i = self._piece_distance_info[p_j].asymmetric_compatibility(p_j_side, p_i,
                                                                                             p_i_side)
                        # Check if the calculation can be skipped for speed-up
                        if p_i_to_p_j == -sys.maxint or p_j_to_p_i == -sys.maxint:
                            mutual_compat = -sys.maxint
                        else:
                            # Get the compatibility from p_i to p_j
                            mutual_compat = (p_i_to_p_j + p_j_to_p_i) / 2

                        # Store the mutual compatibility for BOTH p_i and p_j
                        self._piece_distance_info[p_i].set_mutual_compatibility(p_i_side, p_j, p_j_side, mutual_compat)
                        self._piece_distance_info[p_j].set_mutual_compatibility(p_j_side, p_i, p_i_side, mutual_compat)

    def calculate_mutual_compatibility(self, is_piece_valid_for_placement=None):
        """
        Mutual Compatibility Calculator

        Calculates the mutual compatibility as defined by Paikin and Tal.

        Args:
            is_piece_valid_for_placement (List[bool]): For each piece, True indicates it can be used for placement
                and False means it is not usable for placement.
        """
        start_time = time.time()
        logging.info("Starting mutual compatibility calculations.")

        # Optionally run the single process version
        if True and (is_piece_valid_for_placement or not InterPieceDistance._USE_MULTIPLE_PROCESSES):
            self._calculate_mutual_compatibility_single_process(is_piece_valid_for_placement)
        else:
            self._calculate_mutual_compatibility_multiprocess(is_piece_valid_for_placement)

        for dist_info in self._piece_distance_info:
            dist_info.store_initial_mutual_compatibility()

        logging.info("Mutual compatibility calculations completed.")
        print_elapsed_time(start_time, "mutual compatibility calculation")

    def _calculate_mutual_compatibility_multiprocess(self, is_piece_valid_for_placement=None):
        """
        Multiprocess version of the mutual compatibility calculator.

        Args:
            is_piece_valid_for_placement (Optional List[Bool]): List indicating whether each puzzle piece can be used
                for placement.
        """
        # Calculate the information passed to
        numb_processes = self._calculate_numb_parallel_calculation_processes()
        first_elem_per_process = InterPieceDistance.calculate_elements_per_process_for_diagonal_matrix(self._numb_pieces,
                                                                                                       numb_processes)
        # For simplified calculation append number of pieces
        first_elem_per_process.append(self._numb_pieces)

        # Build the data structure to be pickled and shared with each process
        calc_process_input_elements = []
        for i in xrange(0, numb_processes):
            mutual_compat_process_data = {"first_piece": first_elem_per_process[i],
                                          "last_piece": first_elem_per_process[i + 1],  # Exclusive
                                          "puzzle_type": self._puzzle_type,
                                          "is_piece_valid_for_placement": is_piece_valid_for_placement,
                                          "piece_distance_info": self._piece_distance_info}
            calc_process_input_elements.append(mutual_compat_process_data)

        # Build the thread pool
        process_pool = mp.Pool(numb_processes)
        new_piece_distance = process_pool.map(_multiprocess_mutual_compatibility_calc,
                                              calc_process_input_elements)
        process_pool.close()
        process_pool.join()

        # Transfer the data from the data structures calculated by each process to the master data structure.
        for p_i in xrange(0, self._numb_pieces):
            if InterPieceDistance.skip_piece(p_i, is_piece_valid_for_placement):
                continue

            # Go through all the valid sides
            for p_i_side in PuzzlePieceSide.get_all_sides():

                    process_id = 0  # Used to select from which process to read the calculated results
                    # Iterate through all p_j calculated in this segment
                    for p_j in xrange(p_i + 1, self._numb_pieces):

                        # Go to the process that contains the results for this pairing of p_i and p_j
                        while first_elem_per_process[process_id + 1] <= p_j:
                            process_id += 1

                        # Skip placed pieces
                        # No Need to check p_i == p_j since doing a diagonal calculation
                        if InterPieceDistance.skip_piece(p_j, is_piece_valid_for_placement):
                            continue

                        # Check all valid p_j sides depending on the puzzle type.
                        for p_j_side in InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type, p_i_side):
                            # Extract the mutual compatibility from the results
                            mutual_compat = new_piece_distance[process_id][p_i, p_i_side.value, p_j, p_j_side.value]
                            # Make sure a value is present
                            if InterPieceDistance._PERFORM_ASSERT_CHECKS:
                                assert mutual_compat != np.NAN

                            # Store the mutual compatibility for BOTH p_i and p_j
                            self._piece_distance_info[p_i].set_mutual_compatibility(p_i_side, p_j,
                                                                                    p_j_side, mutual_compat)
                            self._piece_distance_info[p_j].set_mutual_compatibility(p_j_side, p_i,
                                                                                    p_i_side, mutual_compat)

    # Do not always use maximum number of threads.  Base on the workload.
    def _calculate_numb_parallel_calculation_processes(self):
        """
        Used to calculate based on the number of pieces in the puzzle the ideal number of parallel processes
        that will be used.

        Returns (int): Number of processes to be used to perform parallel calculations.

        """
        return min(InterPieceDistance._MAX_NUMBER_OF_PARALLEL_PROCESSES,
                   self._numb_pieces / InterPieceDistance._MIN_NUMBER_PIECES_PER_THREAD)

    def recalculate_remaining_piece_compatibilities(self, piece_valid_for_placement):
        """
        Compatibility Recalculator

        When no best buddy is in the pool, this function is called to recalculate the best buddies and compatibilities.

        Args:
            piece_valid_for_placement (List[bool]): List indicating whether the piece's distance should be
                recalculated.
        """

        # Find the minimum and second best distance information for the placed pieces
        pieces_with_changed_dist = self._update_min_and_second_best_distances(piece_valid_for_placement)

        # Calculate the asymmetric compatibilities using the updated min and second best distances.
        self._recalculate_asymmetric_compatibilities(pieces_with_changed_dist, piece_valid_for_placement)

        # Recalculate the mutual probabilities
        self.calculate_mutual_compatibility(pieces_with_changed_dist)

    def _update_min_and_second_best_distances(self, piece_valid_for_placement):
        """
        During placement, this function will update the minimum and second best distance for those pieces which remain
        valid for placement.

        Args:
            piece_valid_for_placement (List[bool]): List indicating whether each puzzle piece is valid for placement
                If a piece is valid, the associated index is "True."  Otherwise, it is "False."
        """

        # Determine whether anything for this piece needs to be recalculated.
        min_or_second_best_distance_unchanged = [True] * len(piece_valid_for_placement)

        # Iterate through all of the pieces
        for p_i in range(0, self._numb_pieces):

            # # Skip placed pieces
            # if InterPieceDistance.skip_piece(p_i, is_piece_placed_with_no_open_neighbors):
            #     continue

            prev_min_dist = self._piece_distance_info[p_i].minimum_distance
            prev_second_best_dist = self._piece_distance_info[p_i].second_best_distance

            # Find the minimum and second best distance for each piece and side
            # noinspection PyProtectedMember
            self._piece_distance_info[p_i]._find_min_and_second_best_distances(piece_valid_for_placement)

            # Check if the distances changed
            if prev_min_dist != self._piece_distance_info[p_i].minimum_distance \
                    or prev_second_best_dist != self._piece_distance_info[p_i].second_best_distance:
                min_or_second_best_distance_unchanged[p_i] = False

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
                                                piece_valid_for_placement):
        """
        Asymmetric Compatibility Recalculator

        Recalculate the ASYMMETRIC compatibilities of unplaced pieces after minimum and second best distances were
        found.

        Args:
            min_or_second_best_distance_unchanged: (Optional [Bool]): List indicating whether each piece is placed
            piece_valid_for_placement (List[bool]): For each piece, this marks if that piece can be used for
                placement (and in turn compatibility calculation)
        """
        # Iterate through all pieces skipping placed ones.
        for p_i in range(0, self._numb_pieces):
            # Skip placed pieces
            if min_or_second_best_distance_unchanged is not None and not min_or_second_best_distance_unchanged[p_i]:
                continue

            # Recalculate the
            self._piece_distance_info[p_i].calculate_asymmetric_compatibility(piece_valid_for_placement)

    def find_best_buddies(self):  # , is_piece_valid_for_placement=None):
        """
        Finds the best buddies for this set of distance calculations.

        The best buddies information is stored with the inter-piece distances.
        """
        start_time = time.time()
        logging.info("Starting finding of best buddies.")

        # Go through each piece to find its best buddies.
        for p_i in range(0, self._numb_pieces):

            # # Skip placed pieces
            # if InterPieceDistance.skip_piece(p_i, is_piece_valid_for_placement):
            #     continue

            # Find the best buddies of all sides.
            for p_i_side in PuzzlePieceSide.get_all_sides():  # Iterate through all the sides.
                # Get p_i's best buddy candidates.
                best_buddy_candidates = self._piece_distance_info[p_i].best_buddy_candidates(p_i_side)
                # See if the candidates match
                for (p_j, p_j_side) in best_buddy_candidates:
                    piece_dist_info = self._piece_distance_info[p_j]
                    if (p_i, p_i_side) in piece_dist_info.best_buddy_candidates(p_j_side):
                        self._piece_distance_info[p_i].add_best_buddy(p_i_side, p_j, p_j_side)

        logging.info("Finding best buddies completed.")
        print_elapsed_time(start_time, "finding best buddies")

    def find_start_piece_candidates(self, piece_valid_for_placement=None):
        """
        Creates a list of starter puzzle pieces.  This is based off the criteria defined by Paikin and Tal
        where a piece must have 4 best buddies and its best buddies must have 4 best buddies.

        This list is sorted from best starter piece to worst.

        Args:
            piece_valid_for_placement (List[bool]): List indicating whether the selected piece should be considered
                for placement.
        """
        start_time = time.time()
        logging.info("Finding start piece candidates")

        # Calculate each pieces best buddy count for each piece
        all_best_buddy_info = []
        for p_i in range(0, self._numb_pieces):

            # Skip placed pieces
            if InterPieceDistance.skip_piece(p_i, piece_valid_for_placement):
                all_best_buddy_info.append(([], 0))
                continue

            # Best buddies and compatibility storage.
            p_i_best_buddies_and_compat = []

            # Iterate through all sides of p_i
            for p_i_side in PuzzlePieceSide.get_all_sides():

                p_i_best_buddies_and_compat.append([])

                # Iterate through best buddies to pick the best one
                for (p_j, p_j_side) in self._piece_distance_info[p_i].best_buddies(p_i_side):

                    if InterPieceDistance.skip_piece(p_i, piece_valid_for_placement):
                        continue

                    compatibility = self._piece_distance_info[p_i].get_mutual_compatibility(p_i_side, p_j, p_j_side)

                    # Use negative compatibility since we are using a reverse order sorting and this requires
                    # doing things in ascending order which negation does for me here.
                    p_i_best_buddies_and_compat[p_i_side.value].append((p_j, compatibility))

            # Extract the info on the best neighbors
            p_i_best_buddies = []
            neighbor_compatibility = 0
            for bb_and_compat in p_i_best_buddies_and_compat:

                # Check if no best buddies.  If so, go to the next.
                if not bb_and_compat:
                    continue

                # TODO Change the code to support multiple best buddies and to pick the best one.
                bestest_best_buddy_index = 0  # Out of all of the possible best buddies on this side, this is the best one
                p_i_best_buddies.append(bb_and_compat[bestest_best_buddy_index][0])  # Get p_j
                neighbor_compatibility += bb_and_compat[bestest_best_buddy_index][1]  # Get NEGATED mutual compatibility

            # Calculate the average (Not needed since based off the sum and sort off number of best buddies
            # if len(p_i_best_buddies) > 0:
            #     neighbor_compatibility /= len(p_i_best_buddies)
            # else:
            #     neighbor_compatibility = 0

            # Store the best neighbors list as well as the average distance
            all_best_buddy_info.append((p_i_best_buddies, neighbor_compatibility))

        # Build the best neighbor information
        self._start_piece_ordering = []
        for p_i in range(0, self._numb_pieces):

            # Skip placed pieces
            if InterPieceDistance.skip_piece(p_i, piece_valid_for_placement):
                continue

            this_piece_bb_info = all_best_buddy_info[p_i]

            # Store the number of best buddy neighbors
            # Multiply by the number of sides here to prioritize direct neighbors in the case of a tie.
            numb_bb_neighbors = PuzzlePieceSide.get_numb_sides() * len(this_piece_bb_info[0])
            total_compatibility = InterPieceDistance._NEIGHBOR_COMPATIBILITY_SCALAR * this_piece_bb_info[1]

            # Include the neighbors info
            p_i_best_buddy_ids = this_piece_bb_info[0]
            for bb_id in p_i_best_buddy_ids:
                numb_bb_neighbors += len(all_best_buddy_info[bb_id][0])
                if not InterPieceDistance._USE_ONLY_NEIGHBORS_FOR_STARTING_PIECE_TOTAL_COMPATIBILITY:
                    total_compatibility += all_best_buddy_info[bb_id][1]

            # Add this pieces information to the list of possible start pieces
            self._start_piece_ordering.append((p_i, numb_bb_neighbors, total_compatibility))

        # Sort by number of best buddy neighbors (1) then by total compatibility if there is a tie (2)
        # See here for more information: http://stackoverflow.com/questions/4233476/sort-a-list-by-multiple-attributes
        self._start_piece_ordering.sort(key=operator.itemgetter(1, 2), reverse=True)

        self._reinitialize_start_piece_order = True  # In case need to restore initial values later
        logging.info("Finding the start pieces completed.")
        print_elapsed_time(start_time, "finding start pieces")

    def next_starting_piece(self, piece_valid_for_placement=None):
        """
        Next Starting Piece Accessor

        Gets the puzzle piece that is the best candidate to use as the seed of a puzzle.

        Args:
            piece_valid_for_placement (Optional [bool]): An array indicating whether each puzzle piece (by index) has been
                placed.

        Returns (int):
            Index of the next piece to use for starting a board.
        """
        # If no pieces are placed, then use the first piece
        if piece_valid_for_placement is None:
            return self._start_piece_ordering[0][0]

        # If some pieces are already placed, ensure that you do not use a placed piece as the
        # next seed.
        else:
            i = 0
            while not piece_valid_for_placement[self._start_piece_ordering[i][0]]:
                i += 1
            return self._start_piece_ordering[i][0]

    def get_initial_starting_piece_order(self):
        """
        Access the initial starting piece ordering for pieces.

        Returns (List[int]): Descending order of the pieces from best candidate to be a seed piece
          to worst candidate to be a seed piece.
        """
        # return copy.deepcopy(self._initial_start_piece_ordering)  # Seems to take a long time
        piece_ordering = []
        # Remove everything but the piece ID number
        for piece_info in self._initial_start_piece_ordering:
            piece_ordering.append(piece_info[0])
        return piece_ordering

    def best_buddies(self, p_i, p_i_side):
        """
        Gets the best buddy information (if any) for a specified piece's side

        Args:
            p_i (int): Identification number of the piece who best buddy information is to be retrieved
            p_i_side  (PuzzlePieceSide): Side of piece whose best buddy is being retrieved

        Returns (List[int]):
            List of best buddy piece id numbers
        """
        return self._piece_distance_info[p_i].best_buddies(p_i_side)

    def all_best_buddies(self, p_i):
        """
        Gets an array of all best buddies information as a list for a specified puzzle piece.

        Args:
            p_i (int): Piece identification number

        Returns (List[Tuple[int, PuzzlePieceSide]]):
            Best buddy information for the specified piece.

        """
        return self._piece_distance_info[p_i].all_best_buddies()

    def asymmetric_distance(self, p_i, p_i_side, p_j, p_j_side):
        """
        Asymmetric Distance Accessor

        Returns the asymmetric distance for p_i's side (p_i_side) relative to p_j on its side p_j_side.

        Args:
            p_i (int): Primary piece for asymmetric distance
            p_i_side (PuzzlePieceSide): Side of the primary piece (p_i) where p_j will be placed
            p_j (int): Secondary piece for the asymmetric distance.
            p_j_side (PuzzlePieceSide): Side of the secondary piece (p_j) which is adjacent to p_i

        Returns (int):
            Asymmetric distance between puzzle pieces p_i and p_j.
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

        Returns (int):
            Asymmetric compatibility between puzzle pieces p_i and p_j.
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

        Returns (int):
            Mutual compatibility between puzzle pieces p_i and p_j.
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

        For a tuple of puzzle_type and puzzle piece side, this function determines the set of valid PuzzlePieceSide
        for any neighboring piece.

        For example, if the puzzle is type 1, only complementary sides can be placed adjacent to one another.  In
        contrast, if the puzzle is type 2, then any puzzle piece side can be placed adjacent.

        Args:
            puzzle_type (PuzzleType): Puzzle type being solved.
            p_i_side (PuzzlePieceSide): Side of p_i puzzle piece where p_j will be placed.

        Returns (List[PuzzlePieceSide]):
            List of all valid sides for a neighboring puzzle piece.
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

        Returns (int):
            For type 1 puzzles, this normalizes to an index of 0 since it is the only distance for two puzzle pieces
            on a given side of the primary piece.  For type 2 puzzles, the index is set to the p_j_side value defined
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
    def skip_piece(p_i, exclude_piece_list=None):
        """
        Piece Skip Checker

        Checks whether a puzzle piece should be skipped based off whether it is placed or is disallowed in placement.

        Args:
            p_i (int): Identification number of the puzzle piece
            exclude_piece_list (Optional [Bool]):  List indicating the piece should be used

        Returns (bool):
            True if piece p_i should be skipped and False otherwise
        """
        if exclude_piece_list is not None and not exclude_piece_list[p_i]:
            return True
        else:
            return False

    def is_pieces_best_buddies(self, first_piece, first_piece_side, second_piece, second_piece_side):
        """
        Checks whether two pieces pieces are best buddies on the specified sides.

        Args:
            first_piece (PuzzlePiece): A puzzle piece
            first_piece_side (PuzzlePieceSide):  The side of the first puzzle piece
            second_piece (PuzzlePiece): A second puzzle piece
            second_piece_side (PuzzlePieceSide): The side of the second piece being compared

        Returns (Bool): True if the two pieces are best buddies on their respective sides and False otherwise.
        """
        distance_info = self._piece_distance_info[first_piece.id_number]
        return (second_piece.id_number, second_piece_side) in distance_info.best_buddies(first_piece_side)


# Functions are only pickle-able if defined at the top level of a module.
# Pickle-ability is needed to make the function passable to a multiprocess pool.
def _multiprocess_interpiece_distances_calc(interpiece_distance_data):
    """
    Allows for multi-process calculation of inter-piece distance.  Needs to be declared at top level
    otherwise the

    Args:
        interpiece_distance_data (dict): Dictionary containing the distance calculation information
         for interpiece distance.

    Returns (List[PieceDistanceInformation]): Inter-piece distance between the specified indices and
     all other pieces.
    """

    # extract the data from the dictionary containing the input data
    first_element_numb = interpiece_distance_data["first"]
    last_element_numb = interpiece_distance_data["last"]  # Exclusive
    all_pieces = interpiece_distance_data["all_pieces"]
    distance_function = interpiece_distance_data["distance_function"]
    puzzle_type = interpiece_distance_data["puzzle_type"]

    # Calculate the number of pieces
    numb_pieces = len(all_pieces)

    # Calculate each puzzle piece's distance
    calculated_distance_info = []
    for p_i in xrange(first_element_numb, last_element_numb):
        piece_dist_info = PieceDistanceInformation(p_i, numb_pieces, puzzle_type)
        # Calculate the distance information for this piece and append to the list
        piece_dist_info.calculate_inter_piece_distances(all_pieces, distance_function)
        calculated_distance_info.append(piece_dist_info)

    # Return all of the calculated distances
    return calculated_distance_info


def _multiprocess_mutual_compatibility_calc(params):
    """
    Multiprocess Mutual Compatibility Calculator

    Calculates the mutual compatibility as defined by Paikin and Tal.  This supports the use of multiprocessing so
    additional analysis of the data is required.

    A basic diagram of how this function iterates over the diagonal matrix is shown below.  Note that along the
    rows of the matrix, all elements from the first row to the "last_piece" row (exclusive) are traversed.  In
    contrast, only the columns between "first_piece" (inclusive) and "last_piece" (exclusive" are traversed.
    
    -----------
    \   ***   |
     \  ***   |
      \ ***   |
       \ **   |
        \ *   |
         \    |
          \   |
           \  |
            \ |
             \|

    Args:
        params (Dict): Dictionary containing the input parameters to the function.  This must be pickle-able
        to work with the multiprocessing Python class.

    Returns (Numpy[float]): This contains the mutual compatibility data calculated by this process.  The size of the
        NumPy array return is [total_numb_pieces x total_numb_pieces].  However, only the values actually calculated by
        this process are populated.
    """

    first_piece = params["first_piece"]
    last_piece = params["last_piece"]
    puzzle_type = params["puzzle_type"]
    piece_distance_info = params["piece_distance_info"]
    is_piece_valid_for_placement = params["is_piece_valid_for_placement"]

    # Build an array to store the results.  Initialize to NaN
    mutual_compat_data = np.empty([last_piece, PuzzlePieceSide.get_numb_sides(),
                                   last_piece, PuzzlePieceSide.get_numb_sides()], np.float32)
    mutual_compat_data[:] = np.NAN

    # For the given section, calculate all pieces that fall in that sliver of a region.
    # Hence, must start from 0 as shown in the figure below:
    for p_i in range(0, last_piece):

        if InterPieceDistance.skip_piece(p_i, is_piece_valid_for_placement):
            continue

        # Go through all the valid sides
        for p_i_side in PuzzlePieceSide.get_all_sides():
            for p_j in range(first_piece, last_piece):

                # Skip placed pieces
                # No Need to check p_i == p_j since doing a diagonal calculation
                if p_i == p_j or InterPieceDistance.skip_piece(p_j, is_piece_valid_for_placement):
                    continue

                # Check all valid p_j sides depending on the puzzle type.
                for p_j_side in InterPieceDistance.get_valid_neighbor_sides(puzzle_type, p_i_side):

                    p_i_to_p_j = piece_distance_info[p_i].asymmetric_compatibility(p_i_side, p_j, p_j_side)
                    p_j_to_p_i = piece_distance_info[p_j].asymmetric_compatibility(p_j_side, p_i, p_i_side)
                    # Check if the calculation can be skipped for speed-up
                    if p_i_to_p_j == -sys.maxint or p_j_to_p_i == -sys.maxint:
                        mutual_compat = -sys.maxint
                    else:
                        # Get the compatibility from p_i to p_j
                        mutual_compat = (p_i_to_p_j + p_j_to_p_i) / 2

                    # Store the mutual compatibility for BOTH p_i and p_j
                    mutual_compat_data[p_i, p_i_side.value, p_j, p_j_side.value] = mutual_compat
    return mutual_compat_data
