"""
This module contains classes that would be helpful to any solver largely irrespective of the technique
used by the solver.
"""
import logging

import time


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
        self.numb_best_buddies = numb_best_buddies

    def __gt__(self, other):
        """
        Checks if the implicit piece should be prioritized over the specified piece.

        Args:
            other (NextPieceToPlace): Another next piece location to consider.

        Returns (bool):
            True if the implicit piece is a better next piece to place and

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

    _PERFORM_ASSERT_CHECKS = True

    def __init__(self, puzzle_id, row, column):
        self.puzzle_id = puzzle_id
        self.row = row
        self.column = column
        self._key = None

    @property
    def location(self):
        """
        Puzzle location as a tuple of (row, column)

        Returns (Tuple[int]):
            Tuple in the form (row, column)

        """
        return self.row, self.column

    @property
    def key(self):
        """
        Returns a unique key for a given puzzle location on a given board.

        Returns (str):
            Key for this puzzle location

        """
        if self._key is None:
            self._key = str(self.puzzle_id) + "_" + str(self.row) + "_" + str(self.column)
        return self._key

    def is_adjacent_to(self, other):
        """
        Checks whether two the other puzzle location is adjacent to this PuzzleLocation.

        Args:
            other (PuzzleLocation): Puzzle location being compared for adjacency

        Returns (bool): True if the piece is adjacent to the other and False otherwise.
        """
        diff = abs(self.row - other.row) + abs(self.column - other.column)
        return diff == 1 and self.puzzle_id == other.puzzle_id

    @staticmethod
    def are_adjacent(first_loc, second_loc):
        """
        Checks for whether two puzzle locations are adjacent.

        Args:
            first_loc (PuzzleLocation): A puzzle location being checked for adjacency
            second_loc (PuzzleLocation): Other puzzle location being checked for adjacency

        Returns (bool): True if the two pieces are adjacent and False otherwise.
        """
        return first_loc.is_adjacent_to(second_loc)

    def get_adjacent_locations(self, board_dim=None):
        """
        Find the valid locations adjacent to a Puzzle Location.

        Args:
            board_dim (List[int]): Size of the puzzle in the format [number_of_rows, number_of_columns].  If a NumPy
                array is used to represent the board, this can be found using the "shape" method.  If this is not
                specified, then it is ignored.

        Returns (List[PuzzleLocation]): The valid puzzle locations in the order: top, right, bottom, left (if valid).

        """
        adjacent_locations = []
        # Make sure not off the edge of the board
        if self.row > 0:
            adjacent_locations.append(PuzzleLocation(self.puzzle_id, self.row - 1, self.column))

        # Make sure not off the edge of the board if a board dimension is passed
        if board_dim is None or self.column < board_dim[1] - 1:
            adjacent_locations.append(PuzzleLocation(self.puzzle_id, self.row, self.column + 1))

        # Make sure not off the edge of the board if a board dimension is passed
        if board_dim is None or self.row < board_dim[0] - 1:
            adjacent_locations.append(PuzzleLocation(self.puzzle_id, self.row + 1, self.column))

        # Make sure not off the edge of the board
        if self.column > 0:
            adjacent_locations.append(PuzzleLocation(self.puzzle_id, self.row, self.column - 1))

        # Verify the length of the adajcents makes sense
        if board_dim is None or (board_dim[0] > 1 and board_dim[1] > 1):
            assert len(adjacent_locations) >= 2

        return adjacent_locations

    def calculate_manhattan_distance_from(self, other):
        """
        Calculates the manhattan distance between two puzzle location
        Args:
            other (PuzzleLocation): Second puzzle location between compared

        Returns (int): Manhattan distance between the piece locations
        """
        if self.puzzle_id != other.puzzle_id:
            raise ValueError("To calculate manhattan distance between two puzzle locations, they must be in the same puzzle.")

        distance = abs(self.row - other.row) + abs(self.column - other.column)
        if self._PERFORM_ASSERT_CHECKS:
            assert distance >= 0
        return distance

    def __eq__(self, other):
        """
        Compares whether two puzzle pieces are equal

        Args:
            other (PuzzleLocation): Puzzle piece being compared against.

        Returns (True): If the pieces point to the same location and False otherwise.
        """
        return self.puzzle_id == other.puzzle_id and self.row == other.row and self.column == other.column


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

        Returns (int):
            Identification number of a neighbor piece

        """
        return self._neighbor_id

    @property
    def side(self):
        """
        Gets the side of the neighbor piece of interest.

        Returns (PuzzlePieceSide):
            Side of the neighbor piece

        """
        return self._neighbor_side


def print_elapsed_time(start_time, task_name):
    """
    Elapsed Time Printer

    Prints the elapsed time for a task in nice formatting.

    Args:
        start_time (int): Start time in seconds
        task_name (string): Name of the task that was performed

    """
    elapsed_time = time.time() - start_time

    # Print elapsed time and the current time.
    logging.info("The task \"%s\" took %d min %d sec." % (task_name, elapsed_time // 60, elapsed_time % 60))
