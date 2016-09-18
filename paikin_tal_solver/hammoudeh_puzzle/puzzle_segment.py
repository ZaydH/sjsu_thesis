import sys

import numpy as np
from enum import Enum

from hammoudeh_puzzle.solver_helper_classes import PuzzleLocation


class PuzzleSegmentColor(Enum):
    """
    Valid segment colors.  These are represented in BGR format.
    """
    # Red = (0x0, 0x0, 0x0)
    Red = (0x22, 0x22, 0xB2)
    Blue = (0xFF, 0x0, 0x0)
    Green = (0x0, 0xFF, 0x0)
    Yellow = (0x00, 0xFF, 0xFF)
    Pink = (0x93, 0x14, 0xFF)

    @staticmethod
    def get_all_colors():
        """
        Accessor for all of the valid segment colors.

        Returns (List[SegmentColors]): All the valid segment colors
        """
        return [PuzzleSegmentColor.Red, PuzzleSegmentColor.Blue, PuzzleSegmentColor.Green,
                PuzzleSegmentColor.Yellow, PuzzleSegmentColor.Pink]

    def key(self):
        """
        Generates a key for the color for using the color in a dictionary.

        Returns (string): A string key for use of the segment color in a dictionary.

        """
        return str(self.value)


class PuzzleSegmentPieceInfo(object):

    DISTANCE_FROM_EDGE_DEFAULT_VALUE = sys.maxint

    def __init__(self, piece_id, location):
        """

        Args:
            piece_id (int): Identification number of the piece
            location (PuzzleLocation): Location of the puzzle piece
        """

        self._id_numb = piece_id
        self._location = location

        self._is_segment_similarity_piece = False
        self._key = PuzzleSegmentPieceInfo.create_key(piece_id)

        self._distance_from_edge = PuzzleSegmentPieceInfo.DISTANCE_FROM_EDGE_DEFAULT_VALUE

        # color information for the piece
        self._default_color = None
        self._final_color = None


    @property
    def id_number(self):
        """
        Accessor for the associated puzzle piece's identification number.

        Returns (int): Piece identification
        """
        return self._id_numb

    @property
    def key(self):
        """
        Dictionary key associated with this puzzle piece

        Returns (str): Puzzle piece key
        """
        return self._key

    @property
    def location(self):
        """
        Accesses the puzzle piece information's location

        Returns (PuzzleLocation): Puzzle piece's location
        """
        return self._location

    @property
    def is_segment_similarity_piece(self):
        """
        Returns whether the piece is a segment piece informaton

        Returns (bool): True if the piece is a segment similarity piece, and False otherwise

        """
        return self._is_segment_similarity_piece

    @property
    def default_color(self):
        """

        Returns (PuzzleSegmentColor): Default color assigned to all pieces in the segment
        """
        return self._default_color

    @default_color.setter
    def default_color(self, new_default_color):
        """
        Assigns the piece a default color

        Args:
            new_default_color (PuzzleSegmentColor): Default color assigned to all pieces in the segment to
                also be assigned to this piece.
        """
        self._default_color = new_default_color

    @property
    def distance_from_edge(self):
        """
        Access the distance (as measured using the Manhattan distance) of this piece to the edge of the segment

        Returns (int): Distance from the segment to the edge of the segment
        """
        return self._distance_from_edge

    @property
    def distance_from_edge(self):
        """
        Access the distance (as measured using the Manhattan distance) of this piece to the edge of the segment

        Returns (int): Distance from the segment to the edge of the segment
        """
        return self._distance_from_edge

    @distance_from_edge.setter
    def distance_from_edge(self, new_distance):
        """
        Updates the distance from the edge of the segment (as measured by Manhattan distance)

        Args:
            new_distance (int): Manhattan distance from the edge
        """
        self._distance_from_edge = new_distance

    @staticmethod
    def create_key(piece_id):
        """
        Creates a key for a PuzzleSegmentPieceInfo object.
        Args:
            piece_id (int): Puzzle piece identification number
        """
        return str(piece_id)


class PuzzleSegment(object):
    """
    This class is used to store a puzzle the information associated with a puzzle segment.
    """

    _PERFORM_ASSERT_CHECKS = True

    def __init__(self, puzzle_id, segment_id_number):
        """
        Puzzle segment constructor.

        Args:
            puzzle_id (int): Identification number of the solved puzzle
            segment_id_number (int): Unique identification number for the segment.
        """
        self._puzzle_id = puzzle_id
        self._segment_id_number = segment_id_number
        self._numb_pieces = 0
        self._pieces = {}
        self._seed_piece = None

        self._neighbor_segment_ids = {}
        self._neighbor_colors = {}  # This is used for coloring the graph to quickly determine neighbor colors

        self._color = None

    @property
    def puzzle_id(self):
        """
        Each segment is only associated with a single solved puzzle.  This property is used to access the puzzle
        identification number of the implicitly associated segment.

        Returns (int): Identification number of the puzzle associated with this segment.

        """
        return self._puzzle_id

    @property
    def id_number(self):
        """
        Property for return the identification number for a puzzle segment

        Returns (int): Segment ID number
        """
        return self._segment_id_number

    @property
    def numb_pieces(self):
        """
        Property that access the number of pieces in the puzzle

        Returns (int): Number of pieces in the segment (minimum one)
        """
        return self._numb_pieces

    def get_piece_ids(self):
        """
        Gets the identification number of the pieces that are in this segment.

        Returns (List[int]): The identification number of the pieces in this segment.
        """
        piece_ids = []
        for piece_info in self._pieces.values():
            piece_ids.append(piece_info.piece_id)
        return piece_ids

    def add_piece(self, piece):
        """
        Adds a puzzle piece to the puzzle segment.

        Args:
            piece (PuzzlePiece): Puzzle piece to be added
        """
        new_piece = PuzzleSegmentPieceInfo(piece.id_number, piece.puzzle_location)

        # Store the seed piece special
        if len(self._pieces) == 0:
            self._seed_piece = new_piece

        # Store all pieces in the piece dictionary
        self._pieces[new_piece.key] = new_piece

    def remove_piece(self, piece_id):
        """
        Removes a puzzle piece (as defined by the piece's identification number) from the puzzle segment.

        Args:
            piece_id (int): Identification if the puzzle piece to be removed from the segment
        """

        key = PuzzleSegmentPieceInfo.create_key(piece_id)
        # Optionally ensure the key exists before trying to remove it
        if PuzzleSegment._PERFORM_ASSERT_CHECKS:
            assert key in self._pieces          # verify the piece to be deleted actually exists
            assert key != self._seed_piece.key  # Cannot delete the seed piece

        del self._pieces[key]

    def add_neighboring_segment(self, neighbor_segment_id):
        """
        Adds a neighboring segment number for this segment.  Neighboring segment numbers are used to determine the
        degree of each segment and for coloring the visualization.

        Args:
            neighbor_segment_id (int): Identification number of the neighboring segment.
        """
        segment_key = PuzzleSegment._get_segment_key(neighbor_segment_id)

        # Verify that the neighbor ID does not match its own ID
        if PuzzleSegment._PERFORM_ASSERT_CHECKS:
            assert self._segment_id_number != neighbor_segment_id

        # If neighbor does not exist in the list, add the neighbor to the list
        if segment_key not in self._neighbor_segment_ids:
            self._neighbor_segment_ids[segment_key] = neighbor_segment_id

    def get_neighbor_segment_ids(self):
        """
        Accessor for the segments neighboring this segment.

        Returns (List[int]): The identification numbers of all segments that are adjacent to this segment.
        """
        # When getting the neighbor ids in the normal flow, this should not be blank
        if self._PERFORM_ASSERT_CHECKS:
            assert self.neighbor_degree >= 0

        # Convert to a list then return the list.
        neighbor_ids = []
        for val in self._neighbor_segment_ids.values():
            neighbor_ids.append(val)
        return neighbor_ids

    @property
    def neighbor_degree(self):
        """
        Property for getting the number of segments that are adjacent to this puzzle.

        Returns (int): Number of segments that are adjacent to this puzzle.
        """
        numb_neighbors = len(self._neighbor_segment_ids)

        # Verify the degree is greater than 0
        if PuzzleSegment._PERFORM_ASSERT_CHECKS:
            assert numb_neighbors >= 0

        return numb_neighbors

    def assign_distance_from_edge(self):
        """

        Returns:

        """

    def _build_segment_piece_map(self):

        # Find the two extreme corners of the
        [top_left, bottom_right] = self._find_top_left_and_bottom_right_corner_of_segment_map(border_padding=1)

        # Get the width and height of the board
        segment_map_width = bottom_right.col - top_left.col + 1
        segment_map_height = bottom_right.row - top_left.row + 1

        # Build the piece map
        piece_map = np.empty((segment_map_height, segment_map_width))

        # Put the pieces into their respective slot
        for piece_info in self._pieces:
            row = piece_info.location.row - top_left.row
            column = piece_info.location.column - top_left.column
            piece_map[row, column] = piece_info.piece_id

    def _find_top_left_and_bottom_right_corner_of_segment_map(self, border_padding=0):
        """
        Determines the puzzle width and length based on the information in the puzzle piece information objects.

        All information in this function is rebuilt when called so if the user plans to reuse this information they
        should store the values.

        Args:
            border_padding (int): Number of additional padding location around the segment to allow for a blank
                border

        Returns (Tuple[PuzzleLocation]): Tuple in the format (top_left, bottom_right) PuzzleLocation objects with
            the optionally specified padding.

        """

        if border_padding < 0:
            raise ValueError("Border padding must be greater than 0.  %d was specified." % border_padding)

        top_left_corner = PuzzleLocation(self._puzzle_id, sys.maxint, sys.maxint)
        bottom_right_corner = PuzzleLocation(self._puzzle_id, -sys.maxint, -sys.maxint)

        # Get the pieces in the puzzle
        for piece_info in self._pieces.values():
            # Get the piece location
            piece_loc = piece_info.location

            # Check top left
            if top_left_corner.row > piece_loc.row:
                top_left_corner.row = piece_loc.row
            if top_left_corner.column > piece_loc.col:
                top_left_corner.column = piece_loc.col

            # Check bottom right
            if bottom_right_corner.row < piece_loc.row:
                bottom_right_corner.row = piece_loc.row
            if bottom_right_corner.column < piece_loc.col:
                bottom_right_corner.column = piece_loc.col

        # Add the padding
        top_left_corner.row -= border_padding
        top_left_corner.column -= border_padding

        bottom_right_corner.row += border_padding
        bottom_right_corner.column += border_padding

        return [top_left_corner, bottom_right_corner]

    @staticmethod
    def _get_segment_key(segment_id):
        """
        Neighboring segment information is stored in a dictionary.  This function is used create that key identification
        number.

        Args:
            segment_id (int): Identification number of a segment

        Returns (String): Key associated with the puzzle segment identification number.
        """
        return str(segment_id)

    def add_neighbor_color(self, neighbor_color):
        """
        Adds a color for a single neighboring segment.  This allows for quick tracking of what colors the neighbors
        of this segment have.

        Args:
            neighbor_color (PuzzleSegmentColor): Color for a neighboring segment
        """
        # Probably not needed, but include just as a sanity check for now
        if PuzzleSegment._PERFORM_ASSERT_CHECKS:
            assert neighbor_color != self._color

        # If color does not already exist, then add it to the dictionary
        if not self.has_neighbor_color(neighbor_color):
            self._neighbor_colors[neighbor_color.key()] = neighbor_color

    def has_neighbor_color(self, neighbor_color):
        """
        Checks whether this segment already has a neighbor of the specified color.

        Args:
            neighbor_color (PuzzleSegmentColor): A possible color for a neighbor

        Returns (bool): True if this segment already has a neighbor with the specified color and False
           otherwise.
        """
        return neighbor_color.key() in self._neighbor_colors

    def is_colored(self):
        """
        Checks whether the segment is already colored.

        Returns (bool): True if the segment has been assigned a color and False otherwise.
        """
        return self._color is not None

    @property
    def color(self):
        """
        Gets the color assigned to this segment.

        Returns (SegmentColor): Color assigned to this segment
        """
        return self._color

    @color.setter
    def color(self, segment_color):
        """
        Updates the segment color

        Args:
            segment_color (PuzzleSegmentColor): New color for the segment

        """
        # A given segment should only be assigned to a color once
        if PuzzleSegment._PERFORM_ASSERT_CHECKS:
            assert self._color is None
            assert not self.has_neighbor_color(segment_color)  # Make sure no neighbor has this color
        self._color = segment_color

    @staticmethod
    def sort_by_degree(primary, other):
        """
        Sort puzzle segments ascending based off the degree (i.e. number of neighbors).

        Args:
            primary(PuzzleSegment): First piece to be compared.
            other(PuzzleSegment): Other piece to be compared

        Returns (int): -1 if primary has more neighboring segments than other.  0 if both segments have the same
            number of segments.  1 if other has more neighboring segments.
        """
        return cmp(other.neighbor_degree, primary.neighbor_degree)
