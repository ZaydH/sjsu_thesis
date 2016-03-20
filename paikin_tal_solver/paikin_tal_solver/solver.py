"""Paikin Tal Solver Master Module

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""
import numpy

from hammoudeh_puzzle_solver.puzzle_importer import PuzzleType
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePieceRotation, PuzzlePieceSide
from paikin_tal_solver.inter_piece_distance import InterPieceDistance


class BestBuddyPoolInfo(object):

    def __init__(self, piece_id):
        self.piece_id = piece_id

class PuzzleOpenSlot(object):

    def __init__(self, (row, column), piece_id, open_side):
        self.location = (row, column)
        self.piece_id = piece_id
        self.open_side = open_side


class PuzzleDimensions(object):

    def __init__(self, puzzle_id):
        self.puzzle_id = puzzle_id
        self.top_left = (0, 0)
        self.bottom_right = (0, 0)

class NextPieceToPlace(object):

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
                 new_board_mutual_compatibility=None):
        """
        Constructor for the Paikin and Tal solver.

        Args:
            numb_puzzles (int): Number of Puzzles to be solved.
            pieces ([PuzzlePiece])): List of puzzle pieces
            distance_function: Calculates the distance between two PuzzlePiece objects.
            puzzle_type (Optional PuzzleType): Type of Paikin Tal Puzzle
            puzzle_type (Optional Float): Minimum mutual compatibility when new boards are spawned
        """

        # Store the number of pieces.  Shuffle for good measure.
        self._pieces = pieces
        self._piece_placed = [False] * len(pieces)
        self._numb_unplaced_pieces = len(pieces)

        # Define the puzzle dimensions
        self._open_locations = [[]]
        self._piece_locations = []

        # Store the number of puzzles these collective set of pieces comprise.
        self._numb_puzzles = numb_puzzles

        # Store the function used to calculate piece to piece distances.
        self._distance_function = distance_function

        # Select the puzzle type.  If the user did not specify one, use the default.
        self._puzzle_type = puzzle_type if puzzle_type is not None else PaikinTalSolver.DEFAULT_PUZZLE_TYPE

        if new_board_mutual_compatibility is not None:
            self._new_board_mutual_compatibility = new_board_mutual_compatibility
        else:
            self._new_board_mutual_compatibility = PaikinTalSolver.DEFAULT_MINIMUM_MUTUAL_COMPATIBILITY_FOR_NEW_BOARD

        # Stores the best buddies which are prioritized for placement.
        self._best_buddies_pool = []
        self._numb_puzzles = 0

        if PaikinTalSolver._PRINT_PROGRESS_MESSAGES:
            print "Starting to calculate inter-piece distances"

        # Calculate the inter-piece distances.
        self._inter_piece_distance = InterPieceDistance(self._pieces, self._distance_function, self._puzzle_type)

        if PaikinTalSolver._PRINT_PROGRESS_MESSAGES:
            print "Finished calculating inter-piece distances"

    def run(self):
        """
        Runs the Paikin and Tal Solver.
        """

        # Reset the best buddies pool as a precaution.
        self._best_buddies_pool = []

        # Place the initial seed piece
        self._place_seed_piece()

        # Place pieces until no pieces left to be placed.
        while self._numb_unplaced_pieces > 0:

            if PaikinTalSolver._PRINT_PROGRESS_MESSAGES and self._numb_unplaced_pieces % 50 == 0:
                print str(self._numb_unplaced_pieces) + " remain to be placed."

            next_piece = self._find_next_piece()

            # TODO Remove special case when no next piece is selected
            if next_piece is None:
                return

            # TODO Include support for multiple boards
            if False and next_piece.mutual_compatibility < self._new_board_mutual_compatibility:
                self._spawn_new_board()
                # TODO make sure when a next piece is selected but not placed that nothing bad happens
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
        solved_puzzles = [[]] * self._numb_puzzles
        unassigned_pieces = []

        # Iterate through each piece and assign it to the array of pieces
        for piece in self._pieces:
            puzzle_id = piece.puzzle_id

            # If piece is not yet assigned, then group with other unassigned pieces
            if puzzle_id is None:
                unassigned_pieces.append(piece)
            # If piece is assigned, then put with other pieces from its puzzle
            else:
                solved_puzzles[puzzle_id - 1].append(piece)

        # Returns the set of solved puzzles
        return solved_puzzles, unassigned_pieces

    def _place_normal_piece(self, next_piece_info):
        # TODO write place next piece

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
        while i < len(self._open_locations[puzzle_id]):
            open_slot_info = self._open_locations[puzzle_id][i]
            # If this open slot has the same location, remove it.
            # noinspection PyUnresolvedReferences
            if open_slot_info.location == location:
                del self._open_locations[puzzle_id][i]

            # If not the same location then go to the next open slot
            else:
                i += 1

    def _remove_best_buddy_from_pool(self, piece_id):
        i = 0
        while i < len(self._best_buddies_pool):
            bb_info = self._best_buddies_pool[i]
            # If this open slot has the same location, remove it.
            if bb_info.piece_id == piece_id:
                del self._best_buddies_pool[i]
            # Not the same BB so go to the next one
            else:
                i += 1

    def _find_next_piece(self):
        # Prioritize placing from BB pool
        if len(self._best_buddies_pool) > 0:
            return self._get_next_piece_from_best_buddy()
        else:
            # TODO Determine what to do when BB pool is empty
            #assert False
            return None

    def _get_next_piece_from_best_buddy(self):
            best_piece = None
            is_best_buddy = True
            for bb_pool_obj in self._best_buddies_pool:
                # Get the piece id of the next piece to place
                next_piece_id = bb_pool_obj.piece_id

                # Iterate through each of the puzzles
                for puzzle_id in range(0, self._numb_puzzles):
                    # For each piece check each open slot
                    for open_slot in self._open_locations[puzzle_id]:
                        neighbor_piece_id = open_slot.piece_id
                        neighbor_side = open_slot.open_side

                        # Check the set of valid sides for each slot.
                        for next_piece_side in InterPieceDistance.get_valid_neighbor_sides(self._puzzle_type, neighbor_side):
                            mutual_compat = self._inter_piece_distance.mutual_compatibility(next_piece_id, next_piece_side,
                                                                                            neighbor_piece_id, neighbor_side)
                            # Check if need to update the best_piece
                            if best_piece is None or mutual_compat > best_piece.mutual_compatibility:
                                open_slot_location = open_slot.location

                                best_piece = NextPieceToPlace(puzzle_id, open_slot_location,
                                                              next_piece_id, next_piece_side,
                                                              neighbor_piece_id, neighbor_side,
                                                              mutual_compat, is_best_buddy)
            # noinspection PyUnboundLocalVariable
            return best_piece

    def _spawn_new_board(self):
        """
        New Board Spawner

        This function handles spawning a new board including any associated data structure resetting.
        """
        # Perform any cleanup needed.
        assert False

        # Place the next seed piece
        # noinspection PyUnreachableCode
        self._place_seed_piece()

        # Perform any post processing.
        assert False

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

        # Add the placed piece's best buddies to the pool.
        self._add_best_buddies_to_pool(seed.id_number)
        self._update_open_slots(seed)

    def _update_open_slots(self, placed_piece):
        """

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
            if self._piece_locations[puzzle_id][location] != True:
                # noinspection PyTypeChecker
                self._open_locations[puzzle_id].append(PuzzleOpenSlot(location, piece_id, piece_side))

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
                bb_pool_info = BestBuddyPoolInfo(bb[0])

                # If the best buddy is already placed or in the pool, skip it.
                if self._piece_placed[bb[0]] or bb_pool_info in self._best_buddies_pool:
                    continue

                # Add the best buddy to the pool
                self._best_buddies_pool.append(bb_pool_info)

