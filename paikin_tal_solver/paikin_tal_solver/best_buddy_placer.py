"""
This contains classes that will be used by the best buddy placer technique developed by Zayd Hammoudeh
as an extension of Paikin and Tal's solver.
"""
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePieceSide


class MultisidePuzzleOpenSlot(object):
    """
    Represents a single open slot in a puzzle being solved.
    """

    def __init__(self, puzzle_location):
        """

        Args:
            puzzle_location (PuzzleLocation): Puzzle location information in the PuzzleLocation class format.
        """
        self._puzzle_location = puzzle_location
        self._neighbor_side_list = [None] * PuzzlePieceSide.get_numb_sides()
        self._numb_neighbors = 0

    def update_side_neighbor_info(self, side, neighbor_side_info):
        """
        Updates the side neighbor information for a piece.

        Args:
            side (PuzzlePieceSide): Side of the open slot of interest
            neighbor_side_info (NeighborSidePair): Information on the neighbor including the side of the neighbor
            that is adjacent to this piece.
        """
        # Can only update a side once for now.
        assert self._neighbor_side_list[side.value] is None
        # Update the neighbor list and increment the number of neighbors.
        self._neighbor_side_list[side.value] = neighbor_side_info
        self._numb_neighbors += 1

    def numb_neighbors(self):
        """
        Determines the number of ne

        Returns(int): Number neighbors for this open slot.

        """
        return self._numb_neighbors

    @property
    def key(self):
        """
        Gets the key of a MultisidePuzzleOpenSlot object.

        Returns (str): Unique key for a multisided open slot

        """
        return self._puzzle_location.key
