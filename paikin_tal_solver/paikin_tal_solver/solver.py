"""Paikin Tal Solver Master Module

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""
import Queue
import cStringIO
import heapq
import logging
import time

import numpy as np

from hammoudeh_puzzle.best_buddy_placer import BestBuddyPlacerCollection
from hammoudeh_puzzle.puzzle_importer import PuzzleType, PuzzleDimensions, BestBuddyResultsCollection, Puzzle
from hammoudeh_puzzle.puzzle_piece import PuzzlePieceRotation, PuzzlePieceSide
from hammoudeh_puzzle.puzzle_segment import PuzzleSegment, SegmentColor
from hammoudeh_puzzle.solver_helper import NextPieceToPlace, PuzzleLocation, NeighborSidePair, \
    print_elapsed_time
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
        Gets the key associated with the best buddy pool piece.

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

    max_numb_pieces_to_place_in_stitching_piece_solver = 100

    def __init__(self, pieces, distance_function, numb_puzzles=None, puzzle_type=None,
                 new_board_mutual_compatibility=None, fixed_puzzle_dimensions=None):
        """
        Constructor for the Paikin and Tal solver.

        Args:
            pieces (List[PuzzlePiece])): List of puzzle pieces
            distance_function: Calculates the distance between two PuzzlePiece objects.
            numb_puzzles (int): Number of Puzzles to be solved.
            puzzle_type (PuzzleType): Type of Paikin Tal Puzzle
            puzzle_type (float): Minimum mutual compatibility when new boards are spawned
            fixed_puzzle_dimensions(Optional [int]): Size of the puzzle as a Tuple (number_rows, number_columns)
        """

        if numb_puzzles is not None:
            if numb_puzzles < 0:
                raise ValueError("At least a single puzzle is required.")
            if numb_puzzles > 1 and fixed_puzzle_dimensions is not None:
                raise ValueError("When specifying puzzle dimensions, only a single puzzle is allowed.")

        # Store the number of pieces.  Shuffle for good measure.
        self._pieces = pieces
        self.allow_placement_of_all_pieces()

        # Store the number of puzzles these collective set of pieces comprise.
        self._actual_numb_puzzles = numb_puzzles

        # Standard method that re-initializes all of the placed piece information
        self._reset_solved_puzzle_info()

        # Select the puzzle type.  If the user did not specify one, use the default.
        if puzzle_type is None:
            self._puzzle_type = PaikinTalSolver.DEFAULT_PUZZLE_TYPE
        else:
            self._puzzle_type = puzzle_type
            if self._puzzle_type != PuzzleType.type1 and self._puzzle_type != PuzzleType.type2:
                raise ValueError("Invalid puzzle type passed to Paikin Tal Solver constructor.")

        # Store the puzzle dimensions if any
        self._actual_puzzle_dimensions = fixed_puzzle_dimensions

        # Calculates the asymmetric distance, asymmetric & mutual compatibility, best buddies, and starting piece
        # information
        self._calculate_initial_interpiece_distances(distance_function, new_board_mutual_compatibility)

    def _reset_solved_puzzle_info(self):
        """
        Resets all data structures required to begin resolving a puzzle of any type.
        """

        self._numb_puzzles = 0

        # No open slots since solving has not begun
        self._initialize_open_slots()

        # All pieces are unassigned
        self._piece_locations = []
        self._placed_puzzle_dimensions = []  # Stores the dimensions of the puzzle

        # Mark all pieces as unplaced and reset placed piece count
        self._initialize_placed_pieces()

        # Reinitialize all segments
        self._reset_segment_info()

        # Quantifies the number of best buddies that are correct
        self._best_buddy_accuracy = BestBuddyResultsCollection()

        # Reset best buddy accuracy and placer information
        self._best_buddy_accuracy = BestBuddyResultsCollection()
        self._initialize_best_buddy_placer()

        # Initialize best buddy pool and heap for the placer
        self._initialize_best_buddy_pool_and_heap()

    def _calculate_initial_interpiece_distances(self, distance_function, new_board_mutual_compatibility):
        """
        Initializes the data structures for interpiece distance.  If any information currently exists in these data
        structures, they will be totally cleared and replaced by this function.

        Args:
            distance_function: Calculates the distance between two PuzzlePiece objects.
            new_board_mutual_compatibility (float): Minimum mutual compatibility when new boards are spawned
        """

        # # Store the function used to calculate piece to piece distances.
        # self._distance_function = distance_function

        if new_board_mutual_compatibility is not None:
            self._new_board_mutual_compatibility = new_board_mutual_compatibility
        else:
            self._new_board_mutual_compatibility = PaikinTalSolver.DEFAULT_MINIMUM_MUTUAL_COMPATIBILITY_FOR_NEW_BOARD

        # Calculate the inter-piece distances.
        self._inter_piece_distance = InterPieceDistance(self._pieces, distance_function, self._puzzle_type)

        # # Release the Inter-piece distance function to allow pickling.
        # self._distance_function = None

    def _initialize_placed_pieces(self):
        """
        Initializes all placed piece information.  This includes the data structures inside the Paikin and Tal solver
        as well as those local to the individual pieces.
        """
        self._piece_valid_for_placement = [True] * len(self._pieces)
        self._numb_initial_placeable_pieces = len(self._pieces)

        # Use the pieces not allowed for placement based off placement being disallowed
        for i in xrange(0, len(self._pieces)):
            if self._pieces[i].placement_disallowed:
                self._piece_valid_for_placement[i] = False
                self._numb_initial_placeable_pieces -= 1

        self._numb_unplaced_valid_pieces = self._numb_initial_placeable_pieces

        self.reset_all_pieces_placement()

    def reset_all_pieces_placement(self):
        """
        Reinitializes the placement information of all pieces.
        """
        # Initialize all information stored with the individual piece
        for piece in self._pieces:
            piece.reset_placement()

    def _initialize_best_buddy_placer(self):
        """
        If the use of the best buddy placer is selected, then this function initializes all data structures associated
        with best buddy placing.  Otherwise, it has no effect.
        """
        if PaikinTalSolver.use_best_buddy_placer:
            self._best_buddy_placer = BestBuddyPlacerCollection()
        else:
            self._best_buddy_placer = None

    def allow_placement_of_all_pieces(self):
        """
        Enables the placement of all pieces in the solver
        """
        for i in xrange(0, len(self._pieces)):
            self._pieces[i].placement_disallowed = False

    def disallow_piece_placement(self, piece_id):
        """
        Disallows the placement of a particular puzzle piece

        Args:
            piece_id (int): Puzzle piece identification number
        """
        self._pieces[piece_id].placement_disallowed = True

    def _reset_segment_info(self):
        """
        Reset the segments data structure(s) so it is as if the puzzle has no segments.
        """
        self._segments = []

    def restore_initial_placer_settings_and_distances(self):
        """
        Restores all of the initial solver settings that were present when the solver is first created.

        It also resets all of the InterPieceDistance data structures so all of the interpiece distance information
        also matches the original settings.
        """

        self._reset_solved_puzzle_info()

        self._inter_piece_distance.restore_initial_distance_values()

        # The starting piece candidates are reset based off the pieces valid for placement only
        self._inter_piece_distance.find_start_piece_candidates(self._piece_valid_for_placement)

    def run_stitching_piece_solver(self, seed_piece_id):
        """
        Specialized solver used for solving with seed pieces.

        Args:
            seed_piece_id (int): Identification number of the piece to be used as the puzzle seed.
        """

        # Start the placement with the specified seed piece
        self._place_seed_piece(seed_piece_id)

        # Run the solver
        self._run_configurable(max_numb_output_puzzles=1,
                               numb_pieces_to_place=PaikinTalSolver.max_numb_pieces_to_place_in_stitching_piece_solver,
                               skip_initial=True,
                               stop_solver_if_need_to_respawn=False)

    def run_solver_with_specified_seeds(self, seed_piece_ids):
        """
        Runs the solver with the specified piece IDs as seeds.

        When this solver runs, the maximum number of boards is fixed based off of the number of seed pieces.

        Args:
            seed_piece_ids (List[int]): Identification number of all the seed pieces.
        """

        # Build all the boards
        for seed_id in seed_piece_ids:
            self._place_seed_piece(seed_id)

        # Run the solver
        self._run_configurable(max_numb_output_puzzles=len(seed_piece_ids),
                               numb_pieces_to_place=len(self._pieces) - len(seed_piece_ids),
                               skip_initial=True,
                               stop_solver_if_need_to_respawn=False)

    def run_single_puzzle_solver(self):
        """
        Performs placement while allowing only a single output puzzle.
        """

        self._run_configurable(max_numb_output_puzzles=1,
                               numb_pieces_to_place=self._numb_unplaced_valid_pieces,
                               skip_initial=False,
                               stop_solver_if_need_to_respawn=False)

    def run_standard(self, skip_initial=False):
        """
        Runs the Paikin and Tal solver normally.

        Args:
            skip_initial (bool): Used with Pickling.  Skips initial setup of running
        """
        self._run_configurable(max_numb_output_puzzles=self._actual_numb_puzzles,
                               numb_pieces_to_place=self._numb_pieces,
                               skip_initial=skip_initial)

    def _run_configurable(self, max_numb_output_puzzles, numb_pieces_to_place, skip_initial=False,
                          stop_solver_if_need_to_respawn=False):
        """
        Runs the Paikin and Tal Solver.  This function is called by other "run" functions based on the configuration
        required by the solver.

        If the maximum number of output puzzles is ever reached, the solver runs normally and continues placing pieces.
        The only change in the execution is that no new puzzles can be spawned.

        Args:
            max_numb_output_puzzles (int): Maximum number of possible output puzzles.  Actual number of output puzzles
                may be less than this number.

            skip_initial (bool): Used with Pickling.  Skips initial setup.

            stop_solver_if_need_to_respawn (bool): If True, whenever the solver would otherwise spawn a new puzzle,
                the solver will stop.
        """
        if numb_pieces_to_place > self._numb_unplaced_valid_pieces:
            raise ValueError("Number of pieces to place must equal or exceed the number of initial unplaced pieces.")

        if not skip_initial:
            # Place the initial seed piece
            self._place_seed_piece()

        # Store the initial seed piece ordering in case it is needed for segmentation
        self._inter_piece_distance.store_placement_initial_starting_piece_order()

        # Continue placing pieces until the maximum number has been placed
        while self._numb_placed_pieces < numb_pieces_to_place:

            # Log the remaining piece count at some frequency
            if self._numb_unplaced_valid_pieces % PaikinTalSolver._PIECE_COUNT_LOGGING_FREQUENCY == 0:
                logging.debug(str(self._numb_unplaced_valid_pieces) + " remain to be placed.")
                self._best_buddy_accuracy.print_results()

            # if len(self._best_buddies_pool) == 0:
            #     return

            # Get the next piece to place
            next_piece = self._find_next_piece()

            # Handle the case when ordinarily would spawn a new board.
            if next_piece.mutual_compatibility < self._new_board_mutual_compatibility:
                # PickleHelper.exporter(self, "paikin_tal_board_spawn.pk")
                # return

                # Spawning new boards is prevent so break and do not place more pieces
                if stop_solver_if_need_to_respawn:
                    break

                if self._numb_puzzles < max_numb_output_puzzles:
                    self._spawn_new_board()
                    continue

            # Place the next piece
            self._place_normal_piece(next_piece)

        logging.info("Placement complete.\n")

        if self._numb_unplaced_valid_pieces == 0:
            # Clean the heap to reduce the size for pickling.
            self._initialize_best_buddy_pool_and_heap()

            # Print the best buddy result information
            self._best_buddy_accuracy.print_results()
            total_numb_bb_in_dataset = self._inter_piece_distance.get_total_best_buddy_count()
            logging.info("Total number of Best Buddies: %d" % total_numb_bb_in_dataset)
            # Once all pieces have been placed verify that no best buddies remain unaccounted for.
            if PaikinTalSolver._PERFORM_ASSERTION_CHECK:
                for best_buddy_acc in self._best_buddy_accuracy:
                    assert best_buddy_acc.numb_open_best_buddies == 0
                # Removed because when pieces excluded, this will not always be equal.
                # assert self._best_buddy_accuracy.total_best_buddy_count() == total_numb_bb_in_dataset

    @property
    def _numb_placed_pieces(self):
        """
        Gets the number of pieces already placed by the solver.

        Returns (int):
            Number of placed already placed
        """
        return self._numb_initial_placeable_pieces - self._numb_unplaced_valid_pieces

    @property
    def _numb_pieces(self):
        """
        Gets the total number of pieces in the puzzle

        Returns (int):
            Number of pieces in the puzzle.
        """
        return len(self._pieces)

    def get_solved_puzzles(self):
        """
        Paikin and Tal Results Accessor

        Gets the results for the set of the Paikin and Tal solver.

        Returns (List[PuzzlePiece]):
            Multiple puzzles each of which is a set of puzzle pieces.
        """
        # A puzzle is an array of puzzle pieces that can then be reconstructed.
        solved_puzzles = [[] for _ in range(self._numb_puzzles)]
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
                    if self._piece_valid_for_placement[heap_info.bb_id] and self._is_slot_open(heap_info.location):
                        next_piece = NextPieceToPlace(heap_info.location,
                                                      heap_info.bb_id, heap_info.bb_side,
                                                      heap_info.neighbor_id, heap_info.neighbor_side,
                                                      heap_info.mutual_compatibility, True)

            return next_piece

        else:
            logging.debug("Need to recalculate the compatibilities.  Number of pieces left: "
                          + str(self._numb_unplaced_valid_pieces) + "\n")
            #
            # piece_placed_with_open_neighbor = [False] * len(self._pieces)
            # for open_location in self._open_locations:
            #     piece_placed_with_open_neighbor[open_location.piece_id] = True
            # Recalculate the inter-piece distances
            self._inter_piece_distance.recalculate_remaining_piece_compatibilities(self._piece_valid_for_placement)

            # Get all unplaced pieces
            unplaced_pieces = []
            for p_i in range(0, len(self._pieces)):
                # If the piece is not placed, then append to the list
                if self._piece_valid_for_placement[p_i]:
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

    def _get_piece_in_puzzle_location(self, puzzle_location):
        """
        Puzzle Piece Accessor via a Location

        Returns the puzzle piece at the specified location.

        Args:
            puzzle_location (PuzzleLocation): Location in the puzzle from which to get a piece

        Returns (PuzzlePiece): Puzzle piece at the specified location

        """

        # Optionally verify the piece exists in the specified location
        if PaikinTalSolver._PERFORM_ASSERTION_CHECK:
            assert not self._is_slot_open(puzzle_location)

        piece_id = self._piece_locations[puzzle_location.puzzle_id][puzzle_location.location]
        return self._pieces[piece_id]

    def _check_if_perform_best_buddy_heap_housecleaning(self):
        """
        Determines whether best buddy heap housecleaning should be performed.

        Returns (bool):
            True if BB heap house cleaning should not be performed and False otherwise.

        """
        numb_pieces_placed_since_last_housekeeping = (self._last_best_buddy_heap_housekeeping
                                                      - self._numb_unplaced_valid_pieces)
        if not PaikinTalSolver._ENABLE_BEST_BUDDY_HEAP_HOUSEKEEPING:
            return False
        if (len(self._best_buddy_open_slot_heap) >= PaikinTalSolver._MINIMUM_CLEAN_HEAP_SIZE
                and numb_pieces_placed_since_last_housekeeping >= PaikinTalSolver._MINIMUM_CLEAN_HEAP_FREQUENCY):
            return True
        else:
            return False

    def get_initial_starting_piece_ordering(self):
        """
        Gets a copy of the initial starting piece ordering (i.e., based off the input puzzles unaltered).

        Returns (List(int)):
            Piece identification numbers for the starting pieces order from best to worst
        """
        return self._inter_piece_distance.get_placement_initial_starting_piece_order()

    @property
    def numb_unplaced_valid_pieces(self):
        """
        Gets the number of pieces that are valid for placement (i.e., their placement has not been selectively
        disallowed), but that have not yet been placed.

        Returns (int):
            Number of pieces that were allowed to be placed but that have not been.
        """
        return self._numb_unplaced_valid_pieces

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
            if not self._is_slot_open(bb_heap_info.location) or not self._piece_valid_for_placement[bb_heap_info.bb_id]:
                elements_deleted += 1
                continue
            else:
                new_bb_heap.append(bb_heap_info)
        # Mark when BB heap was last cleaned.
        self._last_best_buddy_heap_housekeeping = self._numb_unplaced_valid_pieces

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

        self._best_buddy_open_slot_heap = None  # Initialize here to prevent warnings in PyCharm

        self._best_buddies_pool = {}
        # Clear the best buddy heap
        self._best_buddy_open_slot_heap = []

        # Mark the last heap clear as now
        self._last_best_buddy_heap_housekeeping = self._numb_unplaced_valid_pieces

        self._last_best_buddy_heap_housekeeping = None

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

    def _place_seed_piece(self, seed_piece_id=None):
        """
        Seed Piece Placer

        Whenever a new puzzle board is started, this function should be called.  It removes the best seed piece
        from the set of possible pieces, then places it at the center of the new puzzle with no rotation (for
        simplicity as this using no rotation has no effect on the final solution).

        This function allows for a specific piece to be specified externally as the seed.  If no seed piece is
        specified, the function uses the seed piece based off the start piece candidates.

        Args:
            seed_piece_id (int): Identification number for the seed piece to be used.

        The function then adds the seed piece's best buddies to the pool.
        """

        # Increment the number of puzzles
        self._numb_puzzles += 1

        logging.info("Board #" + str(self._numb_puzzles) + " was created.")

        # Handle the case where no seed piece is specified
        if seed_piece_id is None:
            # Account for placed piece when calculating starting piece candidates.
            if self._numb_puzzles > 1:
                self._inter_piece_distance.find_start_piece_candidates(self._piece_valid_for_placement)
            # Get the first piece for the puzzle
            used_seed_piece_id = self._inter_piece_distance.next_starting_piece(self._piece_valid_for_placement)
        else:
            used_seed_piece_id = seed_piece_id

        # Extract and process seed piece
        self._mark_piece_placed(used_seed_piece_id)
        seed = self._pieces[used_seed_piece_id]

        # Set the seed piece's puzzle id
        seed.puzzle_id = self._numb_puzzles - 1
        # Mark the last heap clear as now
        self._last_best_buddy_heap_housekeeping = self._numb_unplaced_valid_pieces

        # Print information about the seed
        string_io = cStringIO.StringIO()
        print >> string_io, "Seed Piece Information for new Board #%d" % seed.puzzle_id
        print >> string_io, "\tOriginal Puzzle ID:\t%d" % seed.original_puzzle_id
        print >> string_io, "\tOriginal Piece ID:\t%d" % seed.original_piece_id
        # Print the original location information
        original_location = seed.original_puzzle_location
        print >>string_io, "\tOriginal Location:\t(%d, %d)" % (original_location.row, original_location.column)
        print >> string_io, "\n\tSolver Piece ID:\t%d\n" % seed.id_number
        # log the result
        logging.info(string_io.getvalue())
        string_io.close()

        # Initialize the piece locations list
        shape = (len(self._pieces), len(self._pieces))
        self._piece_locations.append(np.full(shape, fill_value=PaikinTalSolver._UNPLACED_PIECE_ID, dtype=np.int32))

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
                if PaikinTalSolver.use_best_buddy_placer:
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
        if abs(diff_row) + abs(diff_col) != 1:
            raise ValueError("The two specified locations are not adjacent.")

        if primary_piece_location.puzzle_id != other_piece_location.puzzle_id:
            raise ValueError("The two specified locations come from different puzzles.")

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

                # Ignore best buddy info for disallowed placement.
                if self._pieces[placed_piece_bb_id].placement_disallowed:
                    self._best_buddy_accuracy[puzzle_id].add_excluded_best_buddy(placed_piece_bb_id,
                                                                                 placed_piece_bb_side)

                # If the BB is already placed, delete from open list if applicable and add to wrong list
                # if applicable
                elif not self._piece_valid_for_placement[placed_piece_bb_id]:
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
        self._piece_valid_for_placement[piece_id] = False
        self._numb_unplaced_valid_pieces -= 1

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
                if not self._piece_valid_for_placement[bb_id] or bb_pool_info.key in self._best_buddies_pool:
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

    def segment(self, perform_segment_cleaning=False, color_segments=False):
        """
        This function divides the set of solved puzzles into a set of disjoint segments.

        Args:
            perform_segment_cleaning (Optional bool): If True, perform additional steps to improve the accuracy
             of the segments.  This will result in smaller segments overall.
            color_segments (Optional bool): Optionally color the individual segments.
        """
        self._perform_segmentation(perform_segment_cleaning)

        self._finding_stitching_pieces()

        # Color all segments
        if color_segments:
            self.color_segments()

        return self._segments

    def save_segment_to_image_file(self, puzzle_id, segment_id, filename_descriptor, image_filenames, start_timestamp):
        """
        Creates an image with just the contents of the solved image.

        Also creates the best buddy image.

        Args:
            puzzle_id (int): Identification number of the solved puzzle
            segment_id (int): Identification number of the segment
            filename_descriptor (str): File descriptor for the image file.
            image_filenames (List(str)): File names of the image
            start_timestamp (int): Timestamp the solver was started.
        """
        segment_piece_ids = self._segments[puzzle_id][segment_id].get_piece_ids()

        # Get the pieces with the identification numbers in the segment
        puzzle_pieces = [self._pieces[piece_id] for piece_id in segment_piece_ids]

        # Build the reconstructed image
        puzzle = Puzzle.reconstruct_from_pieces(puzzle_pieces, puzzle_id)
        image_filename = Puzzle.make_image_filename(image_filenames, filename_descriptor,
                                                    Puzzle.OUTPUT_IMAGE_DIRECTORY,
                                                    self._puzzle_type, start_timestamp)
        puzzle.save_to_file(image_filename)

        # Build the best buddy image
        filename_descriptor += "_best_buddy_acc"
        image_filename = Puzzle.make_image_filename(image_filenames, filename_descriptor,
                                                    Puzzle.OUTPUT_IMAGE_DIRECTORY,
                                                    self._puzzle_type, start_timestamp)
        self.best_buddy_accuracy.output_results_images(image_filenames, [puzzle], self.puzzle_type, start_timestamp,
                                                       output_filenames=[image_filename])

    def _perform_segmentation(self, perform_segment_cleaning):
        """
        Performs the actual segmentation of the solved images.  Each segment is disjoint.

        Args:
            perform_segment_cleaning (Optional bool): If True, perform additional steps to improve the accuracy
             of the segments.  This will result in smaller segments overall.
        """
        start_time = time.time()
        logging.info("Beginning segmentation.")

        # Create a dictionary containing all of the unsegmented pieces
        unassigned_pieces = {}
        for piece in self._pieces:
            if not piece.placement_disallowed:
                unassigned_pieces[piece.key()] = piece.id_number

        # Use the seed priority to determine the order pieces are added to segments.
        piece_segment_priority = self._inter_piece_distance.get_placement_initial_starting_piece_order()

        # Initialize the segment placeholder
        self._segments = []
        while len(self._segments) < self._numb_puzzles:
            self._segments.append([])

        # Initialize the seed and segment building information
        priority_cnt = 0
        segment_piece_queue = Queue.Queue()

        # Continue segmenting
        while unassigned_pieces:

            # Find the next seed piece - Essentially a do while loop
            while True:
                seed_piece_id_number = piece_segment_priority[priority_cnt]
                seed_piece = self._pieces[seed_piece_id_number]
                if seed_piece.key() not in unassigned_pieces:
                    priority_cnt += 1
                    # # If end of the list reached then break.
                    # if priority_cnt == len(self._pieces[seed_piece_id_number]):
                    #     break
                else:
                    break

            # Create a new segment
            new_segment = PuzzleSegment(seed_piece.puzzle_id, len(self._segments[seed_piece.puzzle_id]))
            segment_piece_queue.put(seed_piece)
            del unassigned_pieces[seed_piece.key()]  # Piece now in the queue to be assigned

            # Add pieces to the segment
            while not segment_piece_queue.empty():

                # Add the next piece in the queue and keep looping
                queue_piece = segment_piece_queue.get()

                # Add the piece to the segment
                queue_piece.segment_number = new_segment.id_number
                new_segment.add_piece(queue_piece)

                # Iterate through all the sides and determine if should be added
                for (neighbor_loc, queue_piece_side) in queue_piece.get_neighbor_puzzle_location_and_sides():
                    # If no neighbor is present, go to next side
                    if self._is_slot_open(neighbor_loc):
                        continue

                    # Get the neighbor piece
                    neighbor_piece = self._get_piece_in_puzzle_location(neighbor_loc)

                    # Verify the puzzle identification numbers match
                    if self._PERFORM_ASSERTION_CHECK:
                        assert neighbor_piece.puzzle_id == new_segment.puzzle_id

                    # If piece already assigned a segment go to next piece
                    if str(neighbor_piece.id_number) not in unassigned_pieces:
                        continue

                    neighbor_piece_side = neighbor_piece.side_adjacent_to_location(queue_piece.puzzle_location)

                    # Add the piece to the queue if they are best buddies
                    if self._is_pieces_best_buddies(queue_piece, queue_piece_side, neighbor_piece, neighbor_piece_side):
                        segment_piece_queue.put(neighbor_piece)
                        del unassigned_pieces[neighbor_piece.key()]  # Piece now in the queue to be assigned

            if perform_segment_cleaning:  # Not yet supported.
                # TODO Implement the code to clean segments.
                assert False

            # Add the segment to the list of segments
            self._segments[new_segment.puzzle_id].append(new_segment)

        # Mark which segments are physically adjacent to each other
        self._update_segment_neighbors()
        logging.info("Segmentation completed.")
        print_elapsed_time(start_time, "segmentation")
        self._log_segment_information()

    def _finding_stitching_pieces(self):
        """
        This function finds the stitching pieces in each of the solved segments.
        """
        start_time = time.time()
        logging.info("Starting to find stitching pieces.")

        for puzzle_id in xrange(0, self._numb_puzzles):
            for segment_id in xrange(0, len(self._segments[puzzle_id])):
                self._segments[puzzle_id][segment_id].select_pieces_for_segment_stitching()

        logging.info("Completed finding stitching pieces.")
        print_elapsed_time(start_time, "finding stitching pieces")

    def _log_segment_information(self):
        """
        Helper function to log information regarding the segment.
        """
        # Print the number of segments per puzzle
        string_io = cStringIO.StringIO()
        print >> string_io, "Segments Per Output Puzzle"
        for puzzle_id in xrange(0, self._numb_puzzles):
            print >> string_io, "\tSolved Puzzle #%d:\t%d" % (puzzle_id, len(self._segments[puzzle_id]))
        print >> string_io, ""
        logging.info(string_io.getvalue())
        string_io.close()

    def _update_segment_neighbors(self):
        """
        A segment will have one or more neighbors.  This function updates the segments to reflect their neighboring
        segments.
        """
        # Iterate through all pieces
        for piece in self._pieces:

            # Ignore if placement is disallowed
            if piece.placement_disallowed:
                continue

            piece_puzzle_id = piece.puzzle_id
            piece_segment_id = piece.segment_number

            for (neighbor_loc, _) in piece.get_neighbor_puzzle_location_and_sides():
                # Verify the
                if self._is_slot_open(neighbor_loc):
                    continue
                # Get the neighbor piece
                neighbor_piece = self._get_piece_in_puzzle_location(neighbor_loc)

                # Verify the pieces are from the same puzzle
                if PaikinTalSolver._PERFORM_ASSERTION_CHECK:
                    assert piece_puzzle_id == neighbor_piece.puzzle_id
                    assert PuzzleLocation.are_adjacent(piece.puzzle_location, neighbor_loc)

                # If the pieces are from different segments, then mark them as adjacent
                neighbor_segment_id = neighbor_piece.segment_number
                if piece_segment_id != neighbor_segment_id:
                    self._segments[piece_puzzle_id][piece_segment_id].add_neighboring_segment(neighbor_segment_id)
                    self._segments[piece_puzzle_id][neighbor_segment_id].add_neighboring_segment(piece_segment_id)

    def color_segments(self):
        """
        This function colors each of the segments.  This allows for the generation of a visualization where
        no two adjacent segments have the same color.

        This function uses the Welsh-Powell Algorithm to color the graph.  For more information, see:

                 http://graphstream-project.org/doc/Algorithms/Welsh-Powell/
        """
        start_time = time.time()
        logging.info("Beginning coloring of segments.")

        for puzzle_id in xrange(0, self._numb_puzzles):
            # build a list that allows for sorting the segments by degree (i.e., number of neighbors)
            segment_degree_priority = []
            for i in xrange(0, len(self._segments[puzzle_id])):
                segment_degree_priority.append((i, self._segments[puzzle_id][i].neighbor_degree))
            # Perform an inplace sort using the degree of each segment.
            segment_degree_priority.sort(key=lambda x: x[1], reverse=True)  # Use reverse so descending

            # Get all the colors
            segment_colors = SegmentColor.get_all_colors()
            color_cnt = 0

            # Go through all the segment
            segment_cnt = 0
            while segment_cnt < len(segment_degree_priority):

                # Determine if next segment is colored.  If so, increment and retry
                next_segment_id = segment_degree_priority[segment_cnt][0]
                # if the segment is already colored, go to the next segment
                if self._segments[puzzle_id][next_segment_id].is_colored():
                    segment_cnt += 1
                    continue  # Use continue to prevent overflowing the list

                # Get the next color and then increment the color counter
                next_color = segment_colors[color_cnt]
                color_cnt += 1

                # Color the highest priority segment
                self._assign_color_to_segment(puzzle_id, next_segment_id, next_color)
                # Color any other segments that have no neighbor with the same color
                for other_segment_cnt in xrange(segment_cnt + 1, len(segment_degree_priority)):
                    other_segment_id = segment_degree_priority[other_segment_cnt][0]
                    # Check if not colored and does not have a neighbor with the specified color
                    if not self._segments[puzzle_id][other_segment_id].is_colored() \
                            and not self._segments[puzzle_id][other_segment_id].has_neighbor_color(next_color):
                        self._assign_color_to_segment(puzzle_id, other_segment_id, next_color)

        logging.info("Coloring of segments completed.")
        print_elapsed_time(start_time, "segment coloring")

    def _assign_color_to_segment(self, puzzle_id, segment_id, new_segment_color):
        """
        Colors a segment in the puzzle.  It also marks all neighbors of this segment that they have a neighbor
        with the specified color.

        The last stage in this function is to color the pieces that are part of the segment to be colored.

        Args:
            puzzle_id (int): Puzzle identification where the segment is located.
            segment_id (int): Number for the segment to be colored
            new_segment_color (SegmentColor): Color to set the segment
        """
        # Color the segment
        self._segments[puzzle_id][segment_id].color = new_segment_color

        # For all parts adjacent to this segment, add this as a neighbor color
        neighbor_segment_ids = self._segments[puzzle_id][segment_id].get_neighbor_segment_ids()
        for neighbor_id in neighbor_segment_ids:
            self._segments[puzzle_id][neighbor_id].add_neighbor_color(new_segment_color)

        # Color the pieces that belong to this segment.
        for piece_id in self._segments[puzzle_id][segment_id].get_piece_ids():
            self._pieces[piece_id].segment_color = self._segments[puzzle_id][segment_id].get_piece_color(piece_id)
            self._pieces[piece_id].is_stitching_piece = self._segments[puzzle_id][segment_id].is_piece_used_for_stitching(piece_id)

    def _is_pieces_best_buddies(self, first_piece, first_piece_side, second_piece, second_piece_side):
        """
        Checks whether two pieces pieces are best buddies on the specified sides.

        Args:
            first_piece (PuzzlePiece): A puzzle piece
            first_piece_side (PuzzlePieceSide):  The side of the first puzzle piece
            second_piece (PuzzlePiece): A second puzzle piece
            second_piece_side (PuzzlePieceSide): The side of the second piece being compared

        Returns (Bool): True if the two pieces are best buddies on their respective sides and False otherwise.
        """
        return self._inter_piece_distance.is_pieces_best_buddies(first_piece, first_piece_side,
                                                                 second_piece, second_piece_side)

    def get_piece_original_puzzle_id(self, piece_id):
        """
        Gets the puzzle number associated with the original puzzle associated with this piece.

        Args:
            piece_id (int): Puzzle identification number

        Returns (int): The associated piece's original (i.e. input) puzzle identification number
        """
        return self._pieces[piece_id].original_puzzle_id

    @property
    def puzzle_type(self):
        """
        Puzzle Type Accessor

        Gets whether the puzzle is type 1 or type 2

        Returns (PuzzleType):
            Type of the puzzle (either 1 or 2)
        """
        return self._puzzle_type
