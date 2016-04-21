"""
This module contains classes that would be helpful to any solver largely irrespective of the technique
used by the solver.
"""


class NextPieceToPlace(object):
    """
    Contains all the information on the next piece in the puzzle to be placed.
    """

    def __init__(self, open_slot_location, next_piece_id, next_piece_side,
                 neighbor_piece_id, neighbor_piece_side, compatibility, is_best_buddy, numb_best_buddies=None):
        # Store the location of the open slot where the piece will be placed
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

        # If not a best buddy, then cannot have a number of best buddies
        if not is_best_buddy and numb_best_buddies is not None:
            raise ValueError("The next piece to place was marked as not a best buddy but a best buddy count was specified.")
        self.numb_best_buddies = 0

    def __gt__(self, other):
        """
        Checks if the implicit piece should be prioritized over the specified piece.

        Args:
            other (NextPieceToPlace): Another next piece location to consider.

        Returns (bool): True if the implicit piece is a better next piece to place and

        """
        # If the implicit piece is None and the other is not, then the other is better
        if self.numb_best_buddies is None and other.numb_best_buddies is not None:
            return False
        # Opposite of above case.  Here the implicit piece has best buddies while the other does not
        elif self.numb_best_buddies is not None and other.numb_best_buddies is None:
            return True
        # If number best buddies equal or both None
        elif self.numb_best_buddies == other.numb_best_buddies:
            if self.mutual_compatibility > other.mutual_compatibility:
                return True
            else:
                return False
        # Finally rely on just the best buddy counts.
        if self.numb_best_buddies > other.numb_best_buddies:
            return True
        else:
            return False


class PuzzleLocation(object):
    """
    Structure for formalizing a puzzle location
    """

    def __init__(self, puzzle_id, row, column):
        self.puzzle_id = puzzle_id
        self.row = row
        self.column = column
        self._key = None

    @property
    def location(self):
        """
        Puzzle location as a tuple of (row, column)

        Returns (Tuple[int]): Tuple in the form (row, column)

        """
        return self.row, self.column

    @property
    def key(self):
        """
        Returns a unique key for a given puzzle location on a given board.

        Returns (str): Key for this puzzle location

        """
        if self._key is None:
            self._key = str(self.puzzle_id) + "_" + str(self.row) + "_" + str(self.column)
        return self._key


class NeighborSidePair(object):
    """
    Structure for storing information about a pairing of neighbor side and piece id
    """

    def __init__(self, neighbor_piece_id, neighbor_side):
        """
        Creates a container for storing information on neighbor identification numbers and sides

        Args:
            neighbor_piece_id (int): Identification number of the neighbor piece
            neighbor_side (PuzzlePieceSide): Side of the neighbor puzzle piece
        """
        self._neighbor_id = neighbor_piece_id
        self._neighbor_side = neighbor_side

    @property
    def id_number(self):
        """
        Gets the identification number of a neighbor piece in the neighbor side tuple.

        Returns (int): Identification number of a neighbor piece

        """
        return self._neighbor_id

    @property
    def side(self):
        """
        Gets the side of the neighbor piece of interest.

        Returns (PuzzlePieceSide): Side of the neighbor piece

        """
        return self._neighbor_side
