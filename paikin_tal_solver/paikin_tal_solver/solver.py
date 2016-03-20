import numpy

from hammoudeh_puzzle_solver.puzzle_importer import PuzzleType
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePieceRotation, PuzzlePieceSide
from paikin_tal_solver.inter_piece_distance import InterPieceDistance


class BestBuddyPoolInfo(object):

    def __init__(self, p_i, p_i_side, p_j, p_j_side):
        self.p_i = p_i
        self.p_i_side = p_i_side
        self.p_j = p_j
        self.p_j_side = p_j_side


class PuzzleOpenSlot(object):

    def __init__(self, (row, column), p_j, p_j_side):
        self.coord = (row, column)
        self.p_j = p_j
        self.p_j = p_j_side


class PuzzleDimensions(object):

    def __init__(self, puzzle_id):
        self.puzzle_id = puzzle_id
        self.top_left = (0, 0)
        self.bottom_right = (0, 0)

class PaikinTalSolver(object):
    """
    Paikin & Tal Solver
    """

    # stores the type of the puzzle to solve.
    DEFAULT_PUZZLE_TYPE = PuzzleType.type1

    # Used to simplify debugging without affecting test time by enabling assertion checks
    _PERFORM_ASSERTION_CHECK = True

    def __init__(self, numb_puzzles, pieces, distance_function, puzzle_type=None):
        """
        Constructor for the Paikin and Tal solver.

        Args:
            numb_puzzles (int): Number of Puzzles to be solved.
            pieces ([PuzzlePiece])): List of puzzle pieces
            distance_function: Calculates the distance between two PuzzlePiece objects.
            puzzle_type (Optional PuzzleType): Type of Paikin Tal Puzzle
        """

        # Store the number of pieces.  Shuffle for good measure.
        self._pieces = pieces
        self._pieces_placed = [False] * len(pieces)
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

        # Stores the best buddies which are prioritized for placement.
        self._best_buddies_pool = []
        self._numb_puzzles = 0

        # Calculate the inter-piece distances.
        self._inter_piece_distance = InterPieceDistance(self._pieces, self._distance_function, self._puzzle_type)

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
            if False:
                self._spawn_new_board()

            # Place the next piece
            self._place_normal_piece()

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

    def _place_normal_piece(self):
        pass

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
        first_piece_id = self._inter_piece_distance.next_starting_piece(self._pieces_placed)
        first_piece = self._pieces[first_piece_id]
        self._mark_piece_placed(first_piece_id)

        shape = (len(self._pieces), len(self._pieces))
        self._piece_locations.append(numpy.empty(shape, numpy.bool))

        # Set the first piece's puzzle id
        first_piece.puzzle_id = self._numb_puzzles - 1
        board_center = (int(shape[0] / 2), int(shape[1]) / 2)
        first_piece.location = board_center
        first_piece.rotation = PuzzlePieceRotation.degree_0
        self._piece_locations[first_piece.puzzle_id][board_center] = True  # Note that this piece has been placed

        # Add the placed piece's best buddies to the pool.
        self._add_best_buddies_to_pool(first_piece_id)
        self._update_open_slots(first_piece, first_piece_id)

    def _update_open_slots(self, placed_piece, piece_id):
        """

        Args:
            placed_piece (PuzzlePiece):
            piece_id (int):
        """

        puzzle_id = placed_piece.puzzle_id

        # Get the set of open location puzzle pieces and sides
        location_and_sides = placed_piece.get_neighbor_locations_and_sides()

        # TODO Open slot checker should be made far more efficient
        for location_side in location_and_sides:
                self._open_locations[puzzle_id].append(PuzzleOpenSlot(location_side[0], piece_id, location_side[1]))

    def _mark_piece_placed(self, piece_id):
        """
        Mark Puzzle Piece as Placed

        This function marks a puzzle piece as placed in the Paikin-Tal Puzzle Solver structure.

        Args:
            piece_id (int): Identification number for the puzzle piece
        """
        self._pieces_placed[piece_id] = True
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
                bb_pool_info = BestBuddyPoolInfo(piece_id, p_i_side, bb[0], bb[1])

                # If the best buddy is already placed or in the pool, skip it.
                if self._pieces_placed[bb[0]] or bb_pool_info in self._best_buddies_pool:
                    continue
                # Add the buddy to the
                self._best_buddies_pool.append(bb)

