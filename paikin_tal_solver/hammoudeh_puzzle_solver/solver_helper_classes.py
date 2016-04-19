"""
This module contains classes that would be helpful to any solver largely irrespective of the technique
used by the solver.
"""



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
