"""Paikin Tal Solver Master Module

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""
import copy
import heapq
import pickle

import numpy

from hammoudeh_puzzle_solver.puzzle_importer import PuzzleType
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePieceRotation, PuzzlePieceSide
from paikin_tal_solver.inter_piece_distance import InterPieceDistance


class BestBuddyPoolInfo(object):
    """
    Used to encapulate best buddy objects in the pool of pieces to be placed.
    """
    def __init__(self, piece_id):
        self.piece_id = piece_id
        self._key = str(piece_id)

    @property
    def key(self):
        """

        Returns (int): Best Buddy Pool Info Key.
        """
        return self._key


class BestBuddyHeapInfo(object):
    """
    A heap is used to store the best buddy matches.  This class is used to encapsulate all the requisite
    data for the heap objects.

    It must implement the "__cmp__" method for sorting with the heap.  Note that cmp is used to create a
    maximum heap.
    """

    def __init__(self, bb_id, bb_side, neighbor_id, neighbor_side,
                 puzzle_id, location, mutual_compatibility):
        self.bb_id = bb_id
        self.bb_side = bb_side
        self.neighbor_id = neighbor_id
        self.neighbor_side = neighbor_side
        self.puzzle_id = puzzle_id
        self.location = location
        self.mutual_compatibility = mutual_compatibility

    def __cmp__(self, other):
        """
        Best Buddy Heap Comparison

        Used to organize information in the best buddy info heap.

        Args:
            other:

        Returns: Maximum heap so the piece with the higher mutual compatibility is given higher priority in the
        priority queue.
        """
        # Swapping to make a MAXIMUM heap
        return cmp(other.mutual_compatibility, self.mutual_compatibility)


class PuzzleOpenSlot(object):
    """
    As pieces are placed in the puzzle, invariably open slots on the puzzle board will be opened or closed.

    This data structure stores that information inside a Python dictionary.
    """

    def __init__(self, puzzle_id, (row, column), piece_id, open_side):
        self.puzzle_id = puzzle_id
        self.location = (row, column)
        self.piece_id = piece_id
        self.open_side = open_side
        self._key = str(puzzle_id) + "_" + str(row) + "_" + str(column) + "_" + str(open_side.value)

    @property
    def key(self):
        """
        Dictionary key for the an open slot in the dictionary.
        """
        return self._key


class PuzzleDimensions(object):
    """
    Stores the information regarding the puzzle dimensions including its top left and bottom right corners
    as well as its total size.
    """

    def __init__(self, puzzle_id, starting_point):
        self.puzzle_id = puzzle_id
        self.top_left = [starting_point[0], starting_point[1]]
        self.bottom_right = [starting_point[0], starting_point[1]]
        self.total_size = (1, 1)

    def update_dimensions(self):
        """
        Puzzle Dimensions Updater

        Updates the total dimensions of the puzzle.
        """
        self.total_size = (self.bottom_right[0] - self.top_left[0] + 1,
                           self.bottom_right[1] - self.top_left[1] + 1)


class NextPieceToPlace(object):
    """
    Contains all the information on the next piece in the puzzle to be placed.
    """

    def __init__(self, puzzle_id, open_slot_location, next_piece_id, next_piece_side,
                 neighbor_piece_id, neighbor_piece_side, compatibility, is_best_buddy):
        # Store the location of the open slot where the piece will be placed
        self.puzzle_id = puzzle_id
        self.open_slot_location = open_slot_location

        # Store the information on the next
        self.next_piece_id = next_piece_id
        self.next_piece_side = next_piece_side

        # Store the information about the neighbor piece
        self.neighbor_piece_id = neighbor_piece_id
        self.neighbor_piece_side = neighbor_piece_side

        # Store bookkeeping information
        self.mutual_compatibility = compatibility
        self.is_best_buddy = is_best_buddy

        # Store the information used to determine when to spawn a new board.
        self._numb_avg_placed_unplaced_links = 0
        self._total_placed_unplaced_compatibility_diff = 0


class PickleHelper(object):
    """
    The Pickle Helper class is used to simplify the importing and exporting of objects via the Python Pickle
    Library.
    """

    @staticmethod
    def importer(filename):
        """Generic Pickling Importer Method

        Helper method used to import any object from a Pickle file.

        ::Note::: This function does not support objects of type "Puzzle."  They should use the class' specialized
        Pickling functions.

        Args:
            filename (str): Pickle Filename

        Returns: The object serialized in the specified filename.

        """
        f = open(filename, 'r')
        obj = pickle.load(f)
        f.close()
        return obj

    @staticmethod
    def exporter(obj, filename):
        """Generic Pickling Exporter Method

        Helper method used to export any object to a Pickle file.

        ::Note::: This function does not support objects of type "Puzzle."  They should use the class' specialized
        Pickling functions.

        Args:
            obj:                Object to be exported to a specified Pickle file.
            filename (str):     Name of the Pickle file.

        """
        f = open(filename, 'w')
        pickle.dump(obj, f)
        f.close()


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

    _PRINT_PROGRESS_MESSAGES = True

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
        self._initialize_best_buddy_pool_and_heap()
        self._numb_puzzles = 0

        if PaikinTalSolver._PRINT_PROGRESS_MESSAGES:
            print "Starting to calculate inter-piece distances"

        # Calculate the inter-piece distances.
        self._inter_piece_distance = InterPieceDistance(self._pieces, self._distance_function, self._puzzle_type)

        if PaikinTalSolver._PRINT_PROGRESS_MESSAGES:
            print "Finished calculating inter-piece distances\n\n"

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

        # Place pieces until no pieces left to be placed.
        while self._numb_unplaced_pieces > 0:

            if PaikinTalSolver._PRINT_PROGRESS_MESSAGES and self._numb_unplaced_pieces % 50 == 0:
                print str(self._numb_unplaced_pieces) + " remain to be placed."

            # if len(self._best_buddies_pool) == 0:
            #     PickleHelper.exporter(self, "empty_best_buddy_pool.pk")
            #     return

            # Get the next piece to place
            next_piece = self._find_next_piece()

            if self._numb_puzzles < self._actual_numb_puzzles \
                    and next_piece.mutual_compatibility < PaikinTalSolver.DEFAULT_MINIMUM_MUTUAL_COMPATIBILITY_FOR_NEW_BOARD:
                # PickleHelper.exporter(self, "paikin_tal_board_spawn.pk")
                # return
                self._spawn_new_board()
            else:
                # Place the next piece
                self._place_normal_piece(next_piece)

        if PaikinTalSolver._PRINT_PROGRESS_MESSAGES:
            print "Placement complete.\n\n"

    def get_solved_puzzles(self):
        """
        Paikin and Tal Results Accessor

        Gets the results for the set of the Paikin and Tal solver.

        Returns ([[PuzzlePiece]]): Multiple puzzles each of which is a set of puzzle pieces.
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

        puzzle_id = next_piece_info.puzzle_id

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
        next_piece.location = next_piece_info.open_slot_location

        # Update the board dimensions
        self._updated_puzzle_dimensions(next_piece.puzzle_id, next_piece.location)

        # Update the data structures used for Paikin and Tal
        self._piece_locations[puzzle_id][next_piece.location] = True
        self._mark_piece_placed(next_piece.id_number)
        self._remove_open_slot(puzzle_id, next_piece.location)
        if next_piece_info.is_best_buddy:
            self._remove_best_buddy_from_pool(next_piece.id_number)

        self._add_best_buddies_to_pool(next_piece.id_number)
        self._update_open_slots(next_piece)

    def _remove_open_slot(self, puzzle_id, location):
        i = 0
        while i < len(self._open_locations):
            open_slot_info = self._open_locations[i]
            # If this open slot has the same location, remove it.
            # noinspection PyUnresolvedReferences
            if open_slot_info.puzzle_id == puzzle_id and open_slot_info.location == location:
                del self._open_locations[i]

            # If not the same location then go to the next open slot
            else:
                i += 1

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
            # Keep popping from the heap until a valid next piece is found.
            while next_piece is None:
                # Get the best next piece from the heap.
                heap_info = heapq.heappop(self._best_buddy_open_slot_heap)
                # Make sure the piece is not already placed and/or the slot not already filled.
                if not self._piece_placed[heap_info.bb_id] and self._is_slot_open(heap_info.puzzle_id,
                                                                                  heap_info.location):
                    next_piece = NextPieceToPlace(heap_info.puzzle_id, heap_info.location,
                                                  heap_info.bb_id, heap_info.bb_side,
                                                  heap_info.neighbor_id, heap_info.neighbor_side,
                                                  heap_info.mutual_compatibility, True)

            return next_piece

        else:
            print "\n\nNeed to recalculate the compatibilities.  Number of pieces left: " \
                  + str(self._numb_unplaced_pieces) + "\n\n"

            placed_and_open_pieces = copy.copy(self._piece_placed)
            for open_location in self._open_locations:
                placed_and_open_pieces[open_location.piece_id] = False
            # Recalculate the interpiece distances
            self._inter_piece_distance.recalculate_all_compatibilities_and_best_buddy_info(self._piece_placed,
                                                                                           placed_and_open_pieces)

            # Get all unplaced pieces
            unplaced_pieces = []
            for p_i in range(0, len(self._pieces)):
                # If the piece is not placed, then append to the list
                if not self._piece_placed[p_i]:
                    unplaced_pieces.append(p_i)
            # Use the unplaced pieces to determine the best location.
            return self._get_next_piece_from_pool(unplaced_pieces)

    def _is_slot_open(self, puzzle_id, location):
        """
        Open Slot Checker

        Checks whether the specified location is open in the associated puzzle.

        Args:
            puzzle_id (int): Puzzle identification number
            location ((int)): Tuple of the a locaiton of the puzzle which is row by column

        Returns: True of the location in the specified puzzle is open and false otherwise.
        """
        return not self._piece_locations[puzzle_id][location] and self._check_board_dimensions(puzzle_id, location)

    def _check_board_dimensions(self, puzzle_id, location):

        # If no puzzled dimensions, then slot is definitely open
        actual_dimensions = self._actual_puzzle_dimensions
        if actual_dimensions is None:
            return True
        else:
            puzzle_dimensions = self._placed_puzzle_dimensions[puzzle_id]
            for dim in xrange(0, len(actual_dimensions)):
                # Check if too from from upper left
                if location[dim] - puzzle_dimensions.top_left[dim] + 1 > actual_dimensions[dim]:
                    return False
                # Check if too from from upper left
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
        # heapq.heapify()

    def _get_next_piece_from_pool(self, unplaced_pieces):
        """
        When the best buddy pool is empty, pick the best piece from the unplaced pieces as the next
        piece to be placed.

        Args:
            unplaced_pieces ([BestBuddyPoolInfo]): Set of unplaced pieces

        Returns (NextPieceToPlace): Information on the piece that was selected as the best to be placed.
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
                if not self._is_slot_open(open_slot.puzzle_id, open_slot.location):
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
                        best_piece = NextPieceToPlace(open_slot.puzzle_id, open_slot.location,
                                                      next_piece_id, next_piece_side,
                                                      neighbor_piece_id, neighbor_side,
                                                      mutual_compat, is_best_buddy)
            # noinspection PyUnboundLocalVariable
            return best_piece

    def _initialize_open_slots(self):
        """
        Initalizes the set of open locations.
        """
        self._open_locations = []

    def _spawn_new_board(self):
        """
        New Board Spawner

        This function handles spawning a new board including any associated data structure resetting.
        """
        # Perform any post processing.
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

        if PaikinTalSolver._PRINT_PROGRESS_MESSAGES:
            print "\n\nBoard #" + str(self._numb_puzzles) + " was created.\n\n"

        # Get the first piece for the puzzle
        seed_piece_id = self._inter_piece_distance.next_starting_piece(self._piece_placed)
        seed = self._pieces[seed_piece_id]
        self._mark_piece_placed(seed_piece_id)

        shape = (len(self._pieces), len(self._pieces))
        self._piece_locations.append(numpy.empty(shape, numpy.bool))
        self._piece_locations[self._numb_puzzles - 1].fill(False)

        # Set the first piece's puzzle id
        seed.puzzle_id = self._numb_puzzles - 1
        board_center = (int(shape[0] / 2), int(shape[1]) / 2)
        seed.location = board_center
        seed.rotation = PuzzlePieceRotation.degree_0
        self._piece_locations[seed.puzzle_id][board_center] = True  # Note that this piece has been placed

        # Define new puzzle dimensions with the board center as the top left and bottom right
        puzzle_dimensions = PuzzleDimensions(seed.puzzle_id, board_center)
        self._placed_puzzle_dimensions.append(puzzle_dimensions)

        # # Add a new puzzle to the open locations tracker
        # self._open_locations.append([])

        # Add the placed piece's best buddies to the pool.
        self._add_best_buddies_to_pool(seed.id_number)
        self._update_open_slots(seed)

    def _updated_puzzle_dimensions(self, puzzle_id, placed_piece_location):
        """
        Puzzle Dimensions Updater
        Args:
            puzzle_id (int): Identification number of the puzzle
            placed_piece_location ([int]): Location of the newly placed piece.
        """
        board_dimensions = self._placed_puzzle_dimensions[puzzle_id]
        # Make sure the dimensions are somewhat plausible.
        if PaikinTalSolver._PERFORM_ASSERTION_CHECK:
            assert (board_dimensions.top_left[0] <= board_dimensions.bottom_right[0] and
                    board_dimensions.top_left[1] <= board_dimensions.bottom_right[1])

        # Store the puzzle dimensions.
        dimensions_changed = False
        for dim in range(0, len(board_dimensions.top_left)):
            if board_dimensions.top_left[dim] > placed_piece_location[dim]:
                board_dimensions.top_left[dim] = placed_piece_location[dim]
                dimensions_changed = True
            elif board_dimensions.bottom_right[dim] < placed_piece_location[dim]:
                board_dimensions.bottom_right[dim] = placed_piece_location[dim]
                dimensions_changed = True

        # If the dimensions changed, the update the board size and store it back in the array
        if dimensions_changed:
            board_dimensions.update_dimensions()
            self._placed_puzzle_dimensions[puzzle_id] = board_dimensions

    def _update_open_slots(self, placed_piece):
        """
        Open Slots Updater

        When a piece is placed, this function is run and updates the open slots that may have been created
        by that piece's placement.  For example, when the first piece in a puzzle is placed, this function, will
        open up four new slots.

        Whenever a new slot is opened, it must be compared against all best buddies in the pool and the pairing
        of that open slot and the best buddy added to the heap.

        Args:
            placed_piece (PuzzlePiece):
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
            if self._is_slot_open(puzzle_id, location):
                # noinspection PyTypeChecker
                self._open_locations.append(PuzzleOpenSlot(puzzle_id, location, piece_id, piece_side))

                # For each Best Buddy already in the pool, add an object to the heap.
                for bb_id in self._best_buddies_pool.values():

                    # Go through all valid best_buddy sides
                    valid_sides = InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type, piece_side)
                    for bb_side in valid_sides:
                        mutual_compat = self._inter_piece_distance.mutual_compatibility(piece_id, piece_side,
                                                                                        bb_id, bb_side)
                        # Create a heap info object and push it onto the heap.
                        heap_info = BestBuddyHeapInfo(bb_id, bb_side, piece_id, piece_side,
                                                      puzzle_id, location, mutual_compat)
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
                                                         open_slot_info.puzzle_id, open_slot_info.location,
                                                         mutual_compat)
                        # Push the best buddy onto the heap
                        heapq.heappush(self._best_buddy_open_slot_heap, bb_heap_info)

    @property
    def puzzle_type(self):
        """
        Puzzle Type Accessor

        Gets whether the puzzle is type 1 or type 2

        Returns (PuzzleType): Type of the puzzle (either 1 or 2)
        """
        return self._puzzle_type
