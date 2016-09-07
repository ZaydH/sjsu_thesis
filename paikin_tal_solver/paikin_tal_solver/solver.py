"""Paikin Tal Solver Master Module

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""
import copy
import heapq
import logging

import numpy

from hammoudeh_puzzle.best_buddy_placer import BestBuddyPlacerCollection
from hammoudeh_puzzle.puzzle_importer import PuzzleType, PuzzleDimensions, BestBuddyResultsCollection, Puzzle
from hammoudeh_puzzle.puzzle_piece import PuzzlePieceRotation, PuzzlePieceSide
from hammoudeh_puzzle.solver_helper_classes import NextPieceToPlace, PuzzleLocation, NeighborSidePair
from paikin_tal_solver.inter_piece_distance import InterPieceDistance


class BestBuddyPoolInfo(object):
    """
    Used to encapsulate best buddy objects in the pool of pieces to be placed.
    """
    def __init__(self, piece_id):
        self.piece_id = piece_id
        self._key = str(piece_id)

    @property
    def key(self):
        """

        Returns (int):
            Best Buddy Pool Info Key.
        """
        return self._key


class BestBuddyHeapInfo(object):
    """
    A heap is used to store the best buddy matches.  This class encapsulates all the requisite data for the heap objects.

    It must implement the "__cmp__" method for sorting with the heap.  Note that cmp is used to create a
    maximum heap.
    """

    def __init__(self, bb_id, bb_side, neighbor_id, neighbor_side,
                 location, mutual_compatibility):
        self.bb_id = bb_id
        self.bb_side = bb_side
        self.neighbor_id = neighbor_id
        self.neighbor_side = neighbor_side
        self.location = location
        self.mutual_compatibility = mutual_compatibility

    def __cmp__(self, other):
        """
        Best Buddy Heap Comparison

        Used to organize information in the best buddy info heap.

        Args:
            other:

        Returns (int):
            Maximum heap so the piece with the higher mutual compatibility is given higher priority in the
            priority queue.
        """
        # Swapping to make a MAXIMUM heap
        return cmp(other.mutual_compatibility, self.mutual_compatibility)


class PuzzleOpenSlot(object):
    """
    As pieces are placed in the puzzle, invariably open slots on the puzzle board will be opened or closed.

    This data structure stores that information inside a Python dictionary.
    """

    def __init__(self, location, piece_id, open_side):
        self.location = location
        self.piece_id = piece_id
        self.open_side = open_side

        # Get the information on the row and column
        row = location.row
        column = location.column
        self._key = str(location.puzzle_id) + "_" + str(row) + "_" + str(column) + "_" + str(open_side.value)

    @property
    def key(self):
        """
        Dictionary key for the an open slot in the dictionary.
        """
        return self._key


class PaikinTalSolver(object):
    """
    Paikin & Tal Solver
    """

    # stores the type of the puzzle to solve.
    DEFAULT_PUZZLE_TYPE = PuzzleType.type1

    # Define the minimum mutual compatibility to spawn a new board
    DEFAULT_MINIMUM_MUTUAL_COMPATIBILITY_FOR_NEW_BOARD = 0.5

    # Used to simplify debugging without affecting test time by enabling assertion checks
    _PERFORM_ASSERTION_CHECK = True

    # # Prints progress messages while the puzzle is running
    # _PRINT_PROGRESS_MESSAGES = True

    # Select whether to clear the BB heap on completion
    _CLEAR_BEST_BUDDY_HEAP_ON_SPAWN = True

    # Used to refer to an unplaced piece in the numpy matrix showing the board placement
    _UNPLACED_PIECE_ID = -1

    # Number of pieces to be placed between heap clean-ups
    _ENABLE_BEST_BUDDY_HEAP_HOUSEKEEPING = True
    _MINIMUM_CLEAN_HEAP_SIZE = 1 * (10 ** 6)
    _MINIMUM_CLEAN_HEAP_FREQUENCY = 100

    # Defines how many often the number of remaining pieces to be placed is logged
    _PIECE_COUNT_LOGGING_FREQUENCY = 50

    # Select whether to use the best_buddy_placer
    use_best_buddy_placer = False

    def __init__(self, numb_puzzles, pieces, distance_function, puzzle_type=None,
                 new_board_mutual_compatibility=None, fixed_puzzle_dimensions=None):
        """
        Constructor for the Paikin and Tal solver.

        Args:
            numb_puzzles (int): Number of Puzzles to be solved.
            pieces ([PuzzlePiece])): List of puzzle pieces
            distance_function: Calculates the distance between two PuzzlePiece objects.
            puzzle_type (Optional PuzzleType): Type of Paikin Tal Puzzle
            puzzle_type (Optional Float): Minimum mutual compatibility when new boards are spawned
            fixed_puzzle_dimensions(Optional [int]): Size of the puzzle as a Tuple (number_rows, number_columns)
        """

        if numb_puzzles < 0:
            raise ValueError("At least a single puzzle is required.")
        if numb_puzzles > 1 and fixed_puzzle_dimensions is not None:
            raise ValueError("When specifying puzzle dimensions, only a single puzzle is allowed.")

        # Store the number of pieces.  Shuffle for good measure.
        self._pieces = pieces
        self._piece_placed = [False] * len(pieces)
        self._numb_unplaced_pieces = len(pieces)

        # Define the puzzle dimensions
        self._initialize_open_slots()
        self._piece_locations = []

        # Store the number of puzzles these collective set of pieces comprise.
        self._actual_numb_puzzles = numb_puzzles

        # Store the function used to calculate piece to piece distances.
        self._distance_function = distance_function

        # Quantifies the number of best buddies that
        self._best_buddy_accuracy = BestBuddyResultsCollection()

        # Stores the best buddy placer information
        self._best_buddy_placer = BestBuddyPlacerCollection()

        # Store the puzzle dimensions if any
        self._actual_puzzle_dimensions = fixed_puzzle_dimensions
        self._placed_puzzle_dimensions = []  # Store the dimensions of the puzzle

        # Select the puzzle type.  If the user did not specify one, use the default.
        self._puzzle_type = puzzle_type if puzzle_type is not None else PaikinTalSolver.DEFAULT_PUZZLE_TYPE

        if new_board_mutual_compatibility is not None:
            self._new_board_mutual_compatibility = new_board_mutual_compatibility
        else:
            self._new_board_mutual_compatibility = PaikinTalSolver.DEFAULT_MINIMUM_MUTUAL_COMPATIBILITY_FOR_NEW_BOARD

        # Stores the best buddies which are prioritized for placement.
        self._best_buddy_open_slot_heap = None  # Initialize here to prevent warnings in PyCharm
        self._initialize_best_buddy_pool_and_heap()
        self._numb_puzzles = 0
        self._last_best_buddy_heap_housekeeping = None

        # Calculate the inter-piece distances.
        self._inter_piece_distance = InterPieceDistance(self._pieces, self._distance_function, self._puzzle_type)

        # Release the Inter-piece distance function to allow pickling.
        self._distance_function = None

    def run(self, skip_initial=False):
        """
        Runs the Paikin and Tal Solver.

        Args:
            skip_initial (Optional bool): Used with Pickling.  Skips initial setup.
        """
        if not skip_initial:
            # Place the initial seed piece
            self._place_seed_piece()
            # Mark the last heap clear as now
            self._last_best_buddy_heap_housekeeping = self._numb_unplaced_pieces

        # Place pieces until no pieces left to be placed.
        while self._numb_unplaced_pieces > 0:

            # Log the remaining piece count at some frequency
            if self._numb_unplaced_pieces % PaikinTalSolver._PIECE_COUNT_LOGGING_FREQUENCY == 0:
                logging.debug(str(self._numb_unplaced_pieces) + " remain to be placed.")
                self._best_buddy_accuracy.print_results()

            # if len(self._best_buddies_pool) == 0:
            #     return

            # Get the next piece to place
            next_piece = self._find_next_piece()

            if self._numb_puzzles < self._actual_numb_puzzles \
                    and next_piece.mutual_compatibility < self._new_board_mutual_compatibility:
                # PickleHelper.exporter(self, "paikin_tal_board_spawn.pk")
                # return
                self._spawn_new_board()
            else:
                # Place the next piece
                self._place_normal_piece(next_piece)

        logging.info("Placement complete.\n\n")

        # If no pieces left to place, clean the heap to reduce the size for pickling.
        if self._numb_unplaced_pieces == 0:
            self._initialize_best_buddy_pool_and_heap()
            self._best_buddy_accuracy.print_results()
            total_numb_bb_in_dataset = self._inter_piece_distance.get_total_best_buddy_count()
            logging.info("Total number of Best Buddies: %d" % total_numb_bb_in_dataset)
            # Once all pieces have been placed verify that no best buddies remain unaccounted for.
            if PaikinTalSolver._PERFORM_ASSERTION_CHECK:
                for best_buddy_acc in self._best_buddy_accuracy:
                    assert best_buddy_acc.numb_open_best_buddies == 0
                assert self._best_buddy_accuracy.total_best_buddy_count() == total_numb_bb_in_dataset

    def get_solved_puzzles(self):
        """
        Paikin and Tal Results Accessor

        Gets the results for the set of the Paikin and Tal solver.

        Returns (List[PuzzlePiece]):
            Multiple puzzles each of which is a set of puzzle pieces.
        """
        # A puzzle is an array of puzzle pieces that can then be reconstructed.
        solved_puzzles = [[] for _ in range(self._actual_numb_puzzles)]
        unassigned_pieces = []

        # Iterate through each piece and assign it to the array of pieces
        for piece in self._pieces:
            puzzle_id = piece.puzzle_id

            # If piece is not yet assigned, then group with other unassigned pieces
            if puzzle_id is None:
                unassigned_pieces.append(piece)
            # If piece is assigned, then put with other pieces from its puzzle
            else:
                solved_puzzles[puzzle_id].append(piece)

        # Returns the set of solved puzzles
        return solved_puzzles, unassigned_pieces

    def _place_normal_piece(self, next_piece_info):
        """
        Piece Placer

        This method is used to place all pieces except a board seed piece.

        Args:
            next_piece_info (NextPieceToPlace):  Information on the next piece to place
        """

        puzzle_id = next_piece_info.open_slot_location.puzzle_id

        # Get the neighbor pieces id
        next_piece_id = next_piece_info.next_piece_id
        next_piece = self._pieces[next_piece_id]
        next_piece_side = next_piece_info.next_piece_side

        # Get the neighbor piece's id
        neighbor_piece = self._pieces[next_piece_info.neighbor_piece_id]
        neighbor_piece_side = next_piece_info.neighbor_piece_side

        # Set the parameters of the placed piece
        next_piece.set_placed_piece_rotation(next_piece_side, neighbor_piece_side, neighbor_piece.rotation)
        next_piece.puzzle_id = puzzle_id
        next_piece.location = next_piece_info.open_slot_location.location

        # Update the board dimensions
        self._updated_puzzle_dimensions(next_piece_info.open_slot_location)

        # Update the data structures used for Paikin and Tal
        self._piece_locations[puzzle_id][next_piece.location] = next_piece.id_number
        self._update_best_buddy_accuracy(puzzle_id, next_piece.id_number)

        self._mark_piece_placed(next_piece.id_number)

        self._remove_open_slot(next_piece_info.open_slot_location)
        if next_piece_info.is_best_buddy:
            self._remove_best_buddy_from_pool(next_piece.id_number)

        self._add_best_buddies_to_pool(next_piece.id_number)
        self._update_open_slots(next_piece)
        self._update_best_buddy_collection_neighbor_slots(next_piece.id_number)

    def _remove_open_slot(self, slot_to_remove):
        """
        Open Slot Remover

        For a given puzzle identification number and location (row, column), the removes any locations in the
        open slot list that has that puzzle ID and location.

        Args:
            slot_to_remove (PuzzleLocation): Location in a puzzle to remove.
        """

        puzzle_id = slot_to_remove.puzzle_id
        loc_to_remove = slot_to_remove.location

        i = 0
        while i < len(self._open_locations):
            open_slot_info = self._open_locations[i]
            open_slot_puzzle_id = open_slot_info.location.puzzle_id
            open_slot_loc = open_slot_info.location.location
            # If this open slot has the same location, remove it.
            # noinspection PyUnresolvedReferences
            if open_slot_puzzle_id == puzzle_id and open_slot_loc == loc_to_remove:
                del self._open_locations[i]

            # If not the same location then go to the next open slot
            else:
                i += 1

        # Remove the open slot from the best buddy placer
        if PaikinTalSolver.use_best_buddy_placer:
            self._best_buddy_placer.remove_open_slot(slot_to_remove)

    def _remove_best_buddy_from_pool(self, piece_id):
        """
        Best Buddy Pool Remover

        This function removes best buddies from the best buddy pool.

        Args:
            piece_id (int):  Identification number of best buddy to be removed.
        """
        # If the best buddy is in the pool then delete it.
        bb_info = BestBuddyPoolInfo(piece_id)

        # Verify the key is in the pool.
        if PaikinTalSolver._PERFORM_ASSERTION_CHECK:
            assert bb_info.key in self._best_buddies_pool

        # Delete the best buddy
        del self._best_buddies_pool[bb_info.key]

    def _find_next_piece(self):
        """
        Next Piece to Place Finder

        If the best buddy pool (and accompanying heap) are not empty, then the next piece to place comes from
        the best buddy pool.  If the pool is empty, the mutual compatibilities are recalculated and the piece
        with the highest mutual compatibility with an open slot is selected.

        Returns (NextPieceToPlace):
            Information on the next piece to be placed.
        """

        # Prioritize placing from BB pool
        if len(self._best_buddies_pool) > 0:
            next_piece = None

            # Clean the BB Heap
            if self._check_if_perform_best_buddy_heap_housecleaning():
                self._clean_best_buddy_heap()

            # Use Best Buddy Placer By Default
            if PaikinTalSolver.use_best_buddy_placer:

                next_piece = self._select_piece_using_best_buddies()

            # Use Standard Paikin Tal Placer Always or if no best buddy found
            if not PaikinTalSolver.use_best_buddy_placer or next_piece is None:
                # Keep popping from the heap until a valid next piece is found.
                while next_piece is None:
                    # Get the best next piece from the heap.
                    heap_info = heapq.heappop(self._best_buddy_open_slot_heap)
                    # Make sure the piece is not already placed and/or the slot not already filled.
                    if not self._piece_placed[heap_info.bb_id] and self._is_slot_open(heap_info.location):
                        next_piece = NextPieceToPlace(heap_info.location,
                                                      heap_info.bb_id, heap_info.bb_side,
                                                      heap_info.neighbor_id, heap_info.neighbor_side,
                                                      heap_info.mutual_compatibility, True)

            return next_piece

        else:
            logging.debug("Need to recalculate the compatibilities.  Number of pieces left: "
                          + str(self._numb_unplaced_pieces) + "\n")

            placed_and_open_pieces = copy.copy(self._piece_placed)
            for open_location in self._open_locations:
                placed_and_open_pieces[open_location.piece_id] = False
            # Recalculate the inter-piece distances
            self._inter_piece_distance.recalculate_remaining_piece_compatibilities(self._piece_placed,
                                                                                   placed_and_open_pieces)

            # Get all unplaced pieces
            unplaced_pieces = []
            for p_i in range(0, len(self._pieces)):
                # If the piece is not placed, then append to the list
                if not self._piece_placed[p_i]:
                    unplaced_pieces.append(p_i)
            # Use the unplaced pieces to determine the best location.
            return self._get_next_piece_from_pool(unplaced_pieces)

    def _select_piece_using_best_buddies(self):
        """
        Places a piece using the best buddy placing technique.

        Returns (NextPieceToPlace):
            If a next piece is found, then it returns the information on the best piece to place and None otherwise.

        """

        # Select the next piece to place
        next_piece_to_place = None
        for numb_neighbors in xrange(PuzzlePieceSide.get_numb_sides(), 0, -1):

            # If the piece already has more best buddies than is available for the remaining pieces, then return
            if next_piece_to_place is not None and next_piece_to_place.numb_best_buddies > numb_neighbors:
                return next_piece_to_place

            # Get the open slots associated with the neighbor count
            open_slot_dict = self._best_buddy_placer.get_open_slot_dictionary(numb_neighbors)
            # If no open slots with this neighbor count, go to next count
            if open_slot_dict is None or len(open_slot_dict) == 0:
                continue
            open_slots_with_neighbor_count = open_slot_dict.values()

            # Iterate through all pieces in the best buddy pool
            for bb_id in self._best_buddies_pool.values():

                # Get the best matching open slot for this piece.
                candidate_next_piece = self._get_best_location_for_best_buddy(bb_id, open_slots_with_neighbor_count,
                                                                              numb_neighbors)

                # Check if the next piece should be updated.
                if ((next_piece_to_place is None and candidate_next_piece.numb_best_buddies > 0)
                        or (next_piece_to_place is not None and candidate_next_piece > next_piece_to_place)):
                    next_piece_to_place = candidate_next_piece

        return next_piece_to_place

    def _get_best_location_for_best_buddy(self, bb_id, neighbor_count_open_slots, numb_neighbor_sides):
        """
        For a given best buddy piece id and a list of open slots for a given number of neighbors, this function
        returns the best open slot for that best buddy.

        Args:
            bb_id (int): Information on a best buddy in the pool
            neighbor_count_open_slots (List[MultisidePuzzleOpenSlot]): Open slot information
            numb_neighbor_sides (int): Number of sides with a neighbor

        Returns (NextPieceToPlace):
            Information on a possible candidate for next piece to place
        """

        # Get the information about the piece
        best_buddy_piece = self._pieces[bb_id]

        # Get all the best buddies of the piece
        all_best_buddies = self._inter_piece_distance.all_best_buddies(bb_id)

        # Initialize the next piece to place
        next_piece_to_place = None

        # Iterate through all possible rotations
        if self._puzzle_type == PuzzleType.type1:
            valid_rotations = [PuzzlePieceRotation.degree_0]
        else:
            valid_rotations = PuzzlePieceRotation.all_rotations()
        for rotation in valid_rotations:

            # Iterate through each open slot for the given neighbor count
            for multiside_open_slot in neighbor_count_open_slots:

                # Store number of best buddies
                numb_best_buddies = 0
                mutual_compat = 0

                # Check each side of the piece
                for side in PuzzlePieceSide.get_all_sides():
                    # Check if the neighbor exists.  If not, then skip.
                    neighbor_side_pair = multiside_open_slot.get_neighbor_info(side)
                    if neighbor_side_pair is None:
                        continue

                    # Get the information on the neighbor
                    neighbor_piece_id = neighbor_side_pair.id_number
                    neighbor_side = neighbor_side_pair.side

                    # Calculate an adjusted side
                    adjusted_side_val = (side.value + rotation.value / PuzzlePieceRotation.degree_90.value)
                    adjusted_side_val %= PuzzlePieceSide.get_numb_sides()
                    adjusted_side = PuzzlePieceSide(adjusted_side_val)

                    # Check if the best buddy is right
                    bb_test_candidate = (neighbor_side_pair.id_number, neighbor_side_pair.side)
                    if bb_test_candidate in all_best_buddies[adjusted_side.value]:
                        numb_best_buddies += 1
                    else:
                        numb_best_buddies -= 1

                    # Update the mutual compatibility
                    mutual_compat += self._inter_piece_distance.mutual_compatibility(best_buddy_piece.id_number, adjusted_side,
                                                                                     neighbor_piece_id, neighbor_side)

                # Ensure the number of best buddies does not exceed the number of neighbors
                if PaikinTalSolver._PERFORM_ASSERTION_CHECK and numb_best_buddies > numb_neighbor_sides:
                    assert numb_best_buddies <= numb_neighbor_sides

                # Normalize the mutual compatibility
                mutual_compat /= numb_neighbor_sides
                # noinspection PyUnboundLocalVariable
                candidate_next_piece = NextPieceToPlace(multiside_open_slot.location, best_buddy_piece.id_number,
                                                        adjusted_side, neighbor_piece_id, neighbor_side,
                                                        mutual_compat, True, numb_best_buddies=numb_best_buddies)
                # If this candidate piece is better than the next piece, then update the next piece
                if next_piece_to_place is None or next_piece_to_place < candidate_next_piece:
                    next_piece_to_place = candidate_next_piece

            # Rotate the best buddies
            temp_all_bb = [[] for _ in xrange(0, PuzzlePieceSide.get_numb_sides())]
            for i in xrange(0, PuzzlePieceSide.get_numb_sides()):
                index = (i + 1) % PuzzlePieceSide.get_numb_sides()
                temp_all_bb[index] = all_best_buddies[i]
            all_best_buddies = temp_all_bb

        # Return the piece to place
        return next_piece_to_place

    def _is_slot_open(self, slot_location):
        """
        Open Slot Checker

        Checks whether the specified location is open in the associated puzzle.

        Args:
            slot_location (PuzzleLocation): Unique location in the puzzle

        Returns (bool):
            True of the location in the specified puzzle is open and false otherwise.
        """
        return (self._piece_locations[slot_location.puzzle_id][slot_location.location] == PaikinTalSolver._UNPLACED_PIECE_ID
                and self._check_board_dimensions(slot_location))

    def _check_if_perform_best_buddy_heap_housecleaning(self):
        """
        Determines whether best buddy heap housecleaning should be performed.

        Returns (bool):
            True if BB heap house cleaning should not be performed and False otherwise.

        """
        if not PaikinTalSolver._ENABLE_BEST_BUDDY_HEAP_HOUSEKEEPING:
            return False
        if (len(self._best_buddy_open_slot_heap) >= PaikinTalSolver._MINIMUM_CLEAN_HEAP_SIZE
                and self._last_best_buddy_heap_housekeeping - self._numb_unplaced_pieces >= PaikinTalSolver._MINIMUM_CLEAN_HEAP_FREQUENCY):
            return True
        else:
            return False

    def _clean_best_buddy_heap(self):
        """
        Removes elements in teh BB heap that are no longer valid.  This can be used to speed up placement
        in particular when there are a lot of pieces.
        """

        logging.debug("Cleaning best buddy heap...")

        elements_deleted = 0  # Stores the number of elements in the heap removed
        new_bb_heap = []
        # Go through all the heap elements and if a slot is full or a best buddy was placed, remove
        # Do not add it to the new heap
        for bb_heap_info in self._best_buddy_open_slot_heap:
            if (not self._is_slot_open(bb_heap_info.location)
                    or self._piece_placed[bb_heap_info.bb_id]):
                elements_deleted += 1
                continue
            else:
                new_bb_heap.append(bb_heap_info)
        # Mark when BB heap was last cleaned.
        self._last_best_buddy_heap_housekeeping = self._numb_unplaced_pieces

        # Turn the cleaned list into a heap and replace the existing heap
        heapq.heapify(new_bb_heap)
        self._best_buddy_open_slot_heap = new_bb_heap

        # Print the number of elements deleted
        total_numb_elements = elements_deleted + len(new_bb_heap)
        logging.debug("%d out of %d elements removed in the heap cleanup." % (elements_deleted, total_numb_elements))

    def _check_board_dimensions(self, puzzle_location):
        """
        Checks if the location is an illegal location based off an optional set of puzzle dimensions.

        Args:
            puzzle_location (PuzzleLocation): Unique location in the puzzles

        Returns (bool):
            True if not an illegal based off the board location, False otherwise.

        """

        # Get the specific location information
        puzzle_id = puzzle_location.puzzle_id
        location = puzzle_location.location

        # If no puzzled dimensions, then slot is definitely open
        actual_dimensions = self._actual_puzzle_dimensions
        if actual_dimensions is None:
            return True
        else:
            puzzle_dimensions = self._placed_puzzle_dimensions[puzzle_id]
            for dim in xrange(0, len(actual_dimensions)):
                # Check if too far from from upper left
                if location[dim] - puzzle_dimensions.top_left[dim] + 1 > actual_dimensions[dim]:
                    return False
                # Check if too far from from bottom right
                if puzzle_dimensions.bottom_right[dim] - location[dim] + 1 > actual_dimensions[dim]:
                    return False
        # If puzzle dimensions are not too wide, then the location is open
        return True

    def _initialize_best_buddy_pool_and_heap(self):
        """
        Best Buddy Heap and Pool Initializer

        Initializes a best buddy heap and pool
        """
        self._best_buddies_pool = {}
        # Clear the best buddy heap
        self._best_buddy_open_slot_heap = []

        # Mark the last heap clear as now
        self._last_best_buddy_heap_housekeeping = self._numb_unplaced_pieces

    def _get_next_piece_from_pool(self, unplaced_pieces):
        """
        When the best buddy pool is empty, pick the best piece from the unplaced pieces as the next
        piece to be placed.

        Args:
            unplaced_pieces ([BestBuddyPoolInfo]): Set of unplaced pieces

        Returns (NextPieceToPlace):
            Information on the piece that was selected as the best to be placed.
        """
        is_best_buddy = False
        best_piece = None
        # Get the first object from the pool
        for pool_obj in unplaced_pieces:
            # Get the piece id of the next piece to place
            if is_best_buddy:
                next_piece_id = pool_obj.piece_id
            # When not best buddy, next piece ID is the pool object itself.
            else:
                next_piece_id = pool_obj

            # For each piece check each open slot
            for open_slot in self._open_locations:

                # Ignore any invalid slots
                if not self._is_slot_open(open_slot.location):
                    continue

                # Get the information on the piece adjacent to the open slot
                neighbor_piece_id = open_slot.piece_id
                neighbor_side = open_slot.open_side

                # Check the set of valid sides for each slot.
                for next_piece_side in InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type, neighbor_side):
                    mutual_compat = self._inter_piece_distance.mutual_compatibility(next_piece_id, next_piece_side,
                                                                                    neighbor_piece_id, neighbor_side)
                    # Check if need to update the best_piece
                    if best_piece is None or mutual_compat > best_piece.mutual_compatibility:
                        best_piece = NextPieceToPlace(open_slot.location,
                                                      next_piece_id, next_piece_side,
                                                      neighbor_piece_id, neighbor_side,
                                                      mutual_compat, is_best_buddy)
        # noinspection PyUnboundLocalVariable
        return best_piece

    def _initialize_open_slots(self):
        """
        Initializes the set of open locations.
        """
        self._open_locations = []

    def _spawn_new_board(self):
        """
        New Board Spawner

        This function handles spawning a new board including any associated data structure resetting.
        """
        # Perform any post processing.
        if PaikinTalSolver._CLEAR_BEST_BUDDY_HEAP_ON_SPAWN:
            self._initialize_best_buddy_pool_and_heap()

        # Place the next seed piece
        # noinspection PyUnreachableCode
        self._place_seed_piece()

    def _place_seed_piece(self):
        """
        Seed Piece Placer

        Whenever a new puzzle board is started, this function should be called.  It removes the best seed piece
        from the set of possible pieces, then places it at the center of the new puzzle with no rotation (for
        simplicity as this using no rotation has no effect on the final solution).

        The function then adds the seed piece's best buddies to the pool.
        """

        # Increment the number of puzzles
        self._numb_puzzles += 1

        logging.debug("Board #" + str(self._numb_puzzles) + " was created.")

        # Account for placed piece when calculating starting piece candidates.
        if self._numb_puzzles > 1:
            self._inter_piece_distance.find_start_piece_candidates(self._piece_placed)
        # Get the first piece for the puzzle
        seed_piece_id = self._inter_piece_distance.next_starting_piece(self._piece_placed)
        seed = self._pieces[seed_piece_id]
        self._mark_piece_placed(seed_piece_id)

        # Set the first piece's puzzle id
        seed.puzzle_id = self._numb_puzzles - 1

        # Mark the last heap clear as now
        self._last_best_buddy_heap_housekeeping = self._numb_unplaced_pieces

        # Initialize the piece locations list
        shape = (len(self._pieces), len(self._pieces))
        self._piece_locations.append(numpy.zeros(shape, numpy.int32))
        self._piece_locations[seed.puzzle_id].fill(PaikinTalSolver._UNPLACED_PIECE_ID)

        # Place the piece unrotated in the center of the board.
        board_center = (int(shape[0] / 2), int(shape[1]) / 2)
        seed.location = board_center
        seed.rotation = PuzzlePieceRotation.degree_0
        self._piece_locations[seed.puzzle_id][board_center] = seed.id_number  # Note that this piece has been placed

        # Define new puzzle dimensions with the board center as the top left and bottom right
        puzzle_dimensions = PuzzleDimensions(seed.puzzle_id, board_center)
        self._placed_puzzle_dimensions.append(puzzle_dimensions)

        # Set the best buddy score to zero by default.
        self._best_buddy_accuracy.create_best_buddy_accuracy_for_new_puzzle(seed.puzzle_id)
        self._update_best_buddy_accuracy(seed.puzzle_id, seed.id_number)

        # Add the placed piece's best buddies to the pool.
        self._add_best_buddies_to_pool(seed.id_number)
        self._update_open_slots(seed)
        self._update_best_buddy_collection_neighbor_slots(seed.id_number)

    def _update_best_buddy_collection_neighbor_slots(self, placed_piece_id):
        """
        Updates the information on the open slots in the best buddy placer data structures.

        Args:
            placed_piece_id (int): Identification number of the placed piece.

        """

        # Get the information on the placed piece.
        placed_piece = self._pieces[placed_piece_id]
        placed_piece_location = PuzzleLocation(placed_piece.puzzle_id, placed_piece.location[0],
                                               placed_piece.location[1])

        # Get the open slots
        neighbor_location_and_side = placed_piece.get_neighbor_locations_and_sides()

        # Iterate through the pairings of sides and location
        for i in xrange(0, len(neighbor_location_and_side)):
            # Build a puzzle location object
            (location, side) = neighbor_location_and_side[i]
            neighbor_location = PuzzleLocation(placed_piece.puzzle_id, location[0], location[1])
            # Check if the slot is open
            if self._is_slot_open(neighbor_location):
                # Store the neighbor side
                neighbor_side = Puzzle.get_side_of_primary_adjacent_to_other_piece(neighbor_location,
                                                                                   placed_piece_location)
                # Update the neighbor location information
                placed_piece_and_side = NeighborSidePair(placed_piece_id, side)
                self._best_buddy_placer.update_open_slot(neighbor_location, neighbor_side, placed_piece_and_side)

    @staticmethod
    def get_side_of_primary_adjacent_to_other_piece(primary_piece_location, other_piece_location):
        """
        Given two adjacent pieces (i.e. a primary piece and an other piece), return the side of the primary
        piece that is adjacent (i.e. touching) the other piece.

        Args:
            primary_piece_location (PuzzleLocation): Location of the primary piece
            other_piece_location (PuzzleLocation): Location of the other piece

        Returns (PuzzlePieceSide):
            Side of the primary piece adjacent to the other piece.
        """

        diff_row = primary_piece_location.location[0] - other_piece_location.location[0]
        diff_col = primary_piece_location.location[1] - other_piece_location.location[1]

        # Verify the locations are actually adjacent
        # noinspection PyProtectedMember
        if PaikinTalSolver._PERFORM_ASSERTION_CHECK:
            assert primary_piece_location.puzzle_id == other_piece_location.puzzle_id
            # Verify the locations are exactly one space away
            assert abs(diff_row) + abs(diff_col) == 1

        if diff_row == 1:
            return PuzzlePieceSide.left
        if diff_row == -1:
            return PuzzlePieceSide.right
        if diff_col == 1:
            return PuzzlePieceSide.top
        if diff_col == -1:
            return PuzzlePieceSide.bottom

    def _updated_puzzle_dimensions(self, placed_piece_location):
        """
        Puzzle Dimensions Updater

        Args:
            placed_piece_location (PuzzleLocation): Location of the newly placed piece.
        """
        # Get the specifics of the placed piece
        puzzle_id = placed_piece_location.puzzle_id
        location = placed_piece_location.location

        board_dimensions = self._placed_puzzle_dimensions[puzzle_id]
        # Make sure the dimensions are somewhat plausible.
        if PaikinTalSolver._PERFORM_ASSERTION_CHECK:
            assert (board_dimensions.top_left[0] <= board_dimensions.bottom_right[0] and
                    board_dimensions.top_left[1] <= board_dimensions.bottom_right[1])

        # Store the puzzle dimensions.
        dimensions_changed = False
        for dim in range(0, len(board_dimensions.top_left)):
            if board_dimensions.top_left[dim] > location[dim]:
                board_dimensions.top_left[dim] = location[dim]
                dimensions_changed = True
            elif board_dimensions.bottom_right[dim] < location[dim]:
                board_dimensions.bottom_right[dim] = location[dim]
                dimensions_changed = True

        # If the dimensions changed, the update the board size and store it back in the array
        if dimensions_changed:
            board_dimensions.update_dimensions()
            self._placed_puzzle_dimensions[puzzle_id] = board_dimensions

    @property
    def best_buddy_accuracy(self):
        """
        Access all of the best buddy accuracy information associated with the puzzle.

        Returns (List[BestBuddyAccuracy]):
            All the best buddy accuracy results in the puzzle.
        """
        return self._best_buddy_accuracy

    def _update_best_buddy_accuracy(self, puzzle_id, placed_piece_id):
        """

        Args:
            puzzle_id (int): Identification number for the SOLVED puzzle
            placed_piece_id (int): Identification number of the placed piece
        """

        # Get the place piece's neighbors and the corresponding side the piece.
        neighbor_loc_and_side = self._pieces[placed_piece_id].get_neighbor_locations_and_sides()

        # Iterate through all neighbor locations and sides.
        for (neighbor_loc, placed_side) in neighbor_loc_and_side:

            # Get the neighbor and best buddy ids
            neighbor_id = self._piece_locations[puzzle_id][neighbor_loc]
            is_neighbor_open = (neighbor_id == PaikinTalSolver._UNPLACED_PIECE_ID)

            # Check this piece's info.
            placed_piece_bb_info = self._inter_piece_distance.best_buddies(placed_piece_id, placed_side)
            # If BB list is not empty, then get the BB info.
            if placed_piece_bb_info:
                # TODO This code only supports a single best buddy
                (placed_piece_bb_id, placed_piece_bb_side) = placed_piece_bb_info[0]

            # Handle the neighbor first.
            # Only be need to handle it if it is not empty.
            if not is_neighbor_open:

                neighbor_side = self._pieces[neighbor_id].side_adjacent_to_location(self._pieces[placed_piece_id].location)
                neighbor_best_buddy = self._inter_piece_distance.best_buddies(neighbor_id, neighbor_side)

                # Only need to analyze if no best buddy
                if neighbor_best_buddy:

                    # Delete the best buddy from the open list since definitely has a piece next to it.
                    self._best_buddy_accuracy[puzzle_id].delete_open_best_buddy(neighbor_id, neighbor_side)

                    # If neighbor matches, then add to the list
                    if placed_piece_bb_info and placed_piece_bb_id == neighbor_id and placed_piece_bb_side == neighbor_side:
                        self._best_buddy_accuracy[puzzle_id].add_correct_best_buddy(neighbor_id, neighbor_side)
                        self._best_buddy_accuracy[puzzle_id].add_correct_best_buddy(placed_piece_id, placed_side)
                        continue

            # Check if the placed piece has a best buddy
            # If so, it (and potentially its BB) must be processed
            if placed_piece_bb_info:

                # If the BB is already placed, delete from open list if applicable and add to wrong list
                # if applicable
                if self._piece_placed[placed_piece_bb_id]:
                    # Get the placed piece's puzzle id number
                    bb_puzzle_id = self._pieces[placed_piece_bb_id].puzzle_id
                    # If it is open, delete it from the open list
                    self._best_buddy_accuracy[bb_puzzle_id].delete_open_best_buddy(placed_piece_bb_id,
                                                                                   placed_piece_bb_side)
                    # Neighbor does not match BB so mark as wrong
                    self._best_buddy_accuracy[bb_puzzle_id].add_wrong_best_buddy(placed_piece_bb_id,
                                                                                 placed_piece_bb_side)
                    # Neighbor does not match BB so mark as wrong
                    self._best_buddy_accuracy[bb_puzzle_id].add_wrong_best_buddy(placed_piece_id,
                                                                                 placed_side)
                # If no neighbor and placed piece has a best buddy, add to the open list and move on.
                elif is_neighbor_open:
                    self._best_buddy_accuracy[puzzle_id].add_open_best_buddy(placed_piece_id, placed_side)

    def _update_open_slots(self, placed_piece):
        """
        Open Slots Updater

        When a piece is placed, this function is run and updates the open slots that may have been created
        by that piece's placement.  For example, when the first piece in a puzzle is placed, this function, will
        open up four new slots.

        Whenever a new slot is opened, it must be compared against all best buddies in the pool and the pairing
        of that open slot and the best buddy added to the heap.

        Args:
            placed_piece (PuzzlePiece): Last piece placed
        """
        # Get the placed piece's ID number
        piece_id = placed_piece.id_number

        # Get the puzzle ID number
        puzzle_id = placed_piece.puzzle_id

        # Get the set of open location puzzle pieces and sides
        location_and_sides = placed_piece.get_neighbor_locations_and_sides()

        # TODO Open slot checker should be made far more efficient
        for location_side in location_and_sides:
            location = location_side[0]
            piece_side = location_side[1]

            open_slot_loc = PuzzleLocation(puzzle_id, location[0], location[1])
            if self._is_slot_open(open_slot_loc):
                # noinspection PyTypeChecker
                self._open_locations.append(PuzzleOpenSlot(open_slot_loc, piece_id, piece_side))

                # For each Best Buddy already in the pool, add an object to the heap.
                for bb_id in self._best_buddies_pool.values():

                    # Go through all valid best_buddy sides
                    valid_sides = InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type, piece_side)
                    for bb_side in valid_sides:
                        mutual_compat = self._inter_piece_distance.mutual_compatibility(piece_id, piece_side,
                                                                                        bb_id, bb_side)
                        # Create a heap info object and push it onto the heap.
                        bb_location = PuzzleLocation(puzzle_id, location[0], location[1])
                        heap_info = BestBuddyHeapInfo(bb_id, bb_side, piece_id, piece_side,
                                                      bb_location, mutual_compat)
                        heapq.heappush(self._best_buddy_open_slot_heap, heap_info)

    def _mark_piece_placed(self, piece_id):
        """
        Mark Puzzle Piece as Placed

        This function marks a puzzle piece as placed in the Paikin-Tal Puzzle Solver structure.

        Args:
            piece_id (int): Identification number for the puzzle piece
        """
        self._piece_placed[piece_id] = True
        self._numb_unplaced_pieces -= 1

    def _add_best_buddies_to_pool(self, piece_id):
        """
        Pool Best Buddy Adder

        Per Paikin and Tal's algorithm, when a piece is added to the puzzle, any of its unplaced best buddies are added
        to the pool of best buddies to place.  This function of adding best buddies to the pool is done here.

        Args:
            piece_id (int): Identification number for piece p_i that is being placed.
        """

        # Get the list of best buddies for each side.
        for p_i_side in PuzzlePieceSide.get_all_sides():

            # Get the best buddies for p_i on side i
            best_buddies_for_side = self._inter_piece_distance.best_buddies(piece_id, p_i_side)

            # Buddy/Side Pairs
            for bb in best_buddies_for_side:

                # Create a best buddy pool info object
                bb_id = bb[0]
                bb_pool_info = BestBuddyPoolInfo(bb_id)

                # If the best buddy is already placed or in the pool, skip it.
                if self._piece_placed[bb_id] or bb_pool_info.key in self._best_buddies_pool:
                    continue

                # Add the best buddy to the pool
                self._best_buddies_pool[bb_pool_info.key] = bb_pool_info.piece_id

                # Get the open slots
                for open_slot_info in self._open_locations:

                    # Depending on the puzzle type, only look at the valid sides.
                    valid_sides = InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type,
                                                                              open_slot_info.open_side)
                    for bb_side in valid_sides:
                        # Get the mutual compatibility
                        mutual_compat = self._inter_piece_distance.mutual_compatibility(bb_id, bb_side,
                                                                                        open_slot_info.piece_id,
                                                                                        open_slot_info.open_side)
                        # Build a heap info object.
                        bb_heap_info = BestBuddyHeapInfo(bb_id, bb_side,
                                                         open_slot_info.piece_id, open_slot_info.open_side,
                                                         open_slot_info.location, mutual_compat)
                        # Push the best buddy onto the heap
                        heapq.heappush(self._best_buddy_open_slot_heap, bb_heap_info)

    @property
    def puzzle_type(self):
        """
        Puzzle Type Accessor

        Gets whether the puzzle is type 1 or type 2

        Returns (PuzzleType):
            Type of the puzzle (either 1 or 2)
        """
        return self._puzzle_type
