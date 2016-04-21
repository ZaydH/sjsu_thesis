"""
This contains classes that will be used by the best buddy placer technique developed by Zayd Hammoudeh
as an extension of Paikin and Tal's solver.
"""
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePieceSide


class BestBuddyPlacerCollection(object):

    def __init__(self):
        # Store the location of the open locations
        self._open_locations = {}
        # Depending on the number of best buddies, you are placed in a different tier of dictionary
        self._multiside_open_slots_lists = [{}] * PuzzlePieceSide.get_numb_sides()

    def update_open_slot(self, puzzle_location, side, neighbor_side_info):
        """
        When a piece is placed, some set of adjacent slots will need to be updated.  This performs that
        neighbor updating.

        Args:
            puzzle_location (PuzzleLocation): Unique location of the open slot
            side (PuzzlePieceSide): Side of the open slot where neighbor information is going to be added.
            neighbor_side_info (NeighborSidePair): Information regarding the piece neighbor

        """
        location_key = puzzle_location.key
        # Get the number of neighbors for the piece to be updated
        numb_neighbors = self._open_locations.get(location_key, None)

        # If this location does not exist, add it.
        if numb_neighbors is None:
            self._add_open_slot(puzzle_location, side, neighbor_side_info)
            return

        # Get the open slot and update the specified side
        # noinspection PyTypeChecker
        open_slot = self._multiside_open_slots_lists[numb_neighbors][location_key]
        open_slot.update_side_neighbor_info(side, neighbor_side_info)
        # Update the containers storing the open slot information
        self._put_slot_into_dictionaries(open_slot)

    def _put_slot_into_dictionaries(self, open_slot):
        """
        Puts an open slot into the dictionaries storing the slot information.

        Args:
            open_slot (MultisidePuzzleOpenSlot): Multiside slot information to add to the internal dictionary
              data structures.

        """
        loc_key = open_slot.location.key
        self._open_locations[loc_key] = open_slot.numb_neighbors
        self._multiside_open_slots_lists[open_slot.numb_neighbors][loc_key] = open_slot

    def _add_open_slot(self, puzzle_location, adjacent_side, neighbor_side_info):
        """
        Creates a new open slot and

        Args:
            puzzle_location (PuzzleLocation):
            neighbor_side_info (List[(side, NeighborSidePair)]): List of all neighbor sides.

        """
        # Create a new open slot
        new_open_slot = MultisidePuzzleOpenSlot(puzzle_location)
        new_open_slot.update_side_neighbor_info(adjacent_side, neighbor_side_info)

        # Add the open slot to the dictionaries
        self._put_slot_into_dictionaries(new_open_slot)

    def remove_open_slot(self, piece_location):
        """
        After a piece is placed, this function updates the data structures in teh best buddy placer to
        reflect this is no longer an open slot.

        Args:
            piece_location (PuzzleLocation): Location (both (row, column) and puzzle ID) where the last piece was
              placed

        """

        # Get the number of neighbors for the puzzle piece
        numb_neighbors = self._open_locations.get(piece_location.key, None)
        if numb_neighbors is None:
            return

        # Delete the piece from the diction
        del self._open_locations[piece_location.key]
        del self._multiside_open_slots_lists[piece_location.key]


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

    @property
    def location(self):
        """
        Returns the actual location of the open slot.

        Returns (PuzzleLocation): Location of the open slot

        """
        return self._puzzle_location

    @property
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
