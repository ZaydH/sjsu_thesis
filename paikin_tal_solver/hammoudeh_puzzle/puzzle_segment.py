import sys
import functools

import cv2
import numpy as np
from enum import Enum

from hammoudeh_puzzle.puzzle_piece import PuzzlePiece
from hammoudeh_puzzle.solver_helper_classes import PuzzleLocation


class PuzzleSegmentColor(Enum):
    """
    Valid segment colors.  These are represented in BGR format.
    """
    # Red = (0x22, 0x22, 0xB2)
    Red = (0x0, 0x0, 0xFF)
    Blue = (0xFF, 0x0, 0x0)
    Green = (0x0, 0xFF, 0x0)
    Yellow = (0x00, 0xFF, 0xFF)
    # Pink = (0x93, 0x14, 0xFF)

    @staticmethod
    def get_all_colors():
        """
        Accessor for all of the valid segment colors.

        Returns (List[SegmentColors]): All the valid segment colors
        """
        return [PuzzleSegmentColor.Red, PuzzleSegmentColor.Blue, PuzzleSegmentColor.Green,
                PuzzleSegmentColor.Yellow]  # , PuzzleSegmentColor.Pink]

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

        self._distance_from_open_space = PuzzleSegmentPieceInfo.DISTANCE_FROM_EDGE_DEFAULT_VALUE

        # color information for the piece
        self._default_color = None
        self._distance_based_color = None

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
        Returns whether the piece is a segment piece information

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
        if new_default_color != self._default_color:
            # Need a new distance based color
            self._distance_based_color = None

        self._default_color = new_default_color

    @property
    def distance_from_open_space(self):
        """
        Access the distance (as measured using the Manhattan distance) of this piece to the edge of the segment

        Returns (int): Distance from the segment to the edge of the segment
        """
        return self._distance_from_open_space

    @property
    def distance_from_open_space(self):
        """
        Access the distance (as measured using the Manhattan distance) of this piece to the edge of the segment

        Returns (int): Distance from the segment to an open space
        """
        return self._distance_from_open_space

    @distance_from_open_space.setter
    def distance_from_open_space(self, new_distance):
        """
        Updates the distance from the edge of the segment (as measured by Manhattan distance)

        Args:
            new_distance (int): Manhattan distance from the edge
        """
        if new_distance != self.distance_from_open_space:
            # Need a new distance based color
            self._distance_based_color = None

        self.distance_from_open_space = new_distance

    def get_color_based_on_distance_from_an_open(self):
        """
        Returns a modified version of the puzzle segment's default color where the color is reflective of the
        distance the piece is from an open space.

        Returns (List[int]): Color of the piece with a lightening based off the piece's distance from an open space
        """

        # If the distance based color is already stored, then no need to recalculate
        if self._distance_based_color is not None:
            return self._distance_based_color

        max_lightness = 225  # Done to prevent just white sections in the image
        lightness_increase_per_distance_from_open = 5

        # Create an image to use the
        color_img = np.empty(1, 1, PuzzlePiece.NUMB_LAB_COLORSPACE_DIMENSIONS)
        for i in xrange(0, PuzzlePiece.NUMB_LAB_COLORSPACE_DIMENSIONS):
            color_img[0, 0, i] = self._default_color.value[i]

        # Convert color to lab
        lab_color = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)

        # Calculate a new lightness but do not let it be too light
        new_lightness = lab_color[0, 0, 0] + self._distance_from_open_space * lightness_increase_per_distance_from_open
        new_lightness = min(max_lightness, new_lightness)
        lab_color[0, 0, 0] = new_lightness

        # Calculate a new BGR that is lightened and then return it
        new_bgr = cv2.cvtColor(lab_color, cv2.COLOR_LAB2BGR)
        self._distance_based_color = [0, 0, 0]
        for i in PuzzlePiece.NUMB_LAB_COLORSPACE_DIMENSIONS:
            self._distance_based_color[i] = new_bgr[0, 0, i]
        return self._distance_based_color

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

    _EMPTY_PIECE_MAP_VALUE = -1

    _USE_DISTANCE_FROM_EDGE_BASED_COLORING = True

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

        self._piece_distance_from_open_space_up_to_date = False

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

        # Since a piece was removed, piece distance information no longer up to date
        self._piece_distance_from_open_space_up_to_date = False

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

        # Since a piece was removed, piece distance information no longer up to date
        self._piece_distance_from_open_space_up_to_date = False

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

    def _calculate_piece_distances_to_open_location(self):
        """
        Updates the segment's puzzle piece information to mark the number of pieces that are within a specified
        distance of an open segment location.
        """
        piece_map = self._build_segment_piece_map()

        # Build a list of unused pieces
        frontier_pieces = {}
        for piece in self._pieces:
            frontier_pieces[piece.key] = piece.id_number

        explored_pieces = {}  # As pieces' minimum distance is found, mark as explored
        previous_generation_locations = self._find_open_spaces_adjacent_to_piece(piece_map)

        distance_to_open = 0  # initialize consider the neighbors
        # Continue looping until all pieces have a distance found
        while len(frontier_pieces) > 0:
            # The current generation locations are used to find the distance for the next generation
            current_generation_locations = []
            for prev_gen_location in previous_generation_locations:
                # Find pieces adjacent to the previous location to increment distance by 1
                for adjacent_loc in prev_gen_location.get_adjacent_locations(piece_map.shape):

                    # Ignore locations off the piece map or that is empty
                    if not PuzzleSegment._is_piece_map_location_valid(piece_map, adjacent_loc.row, adjacent_loc.column) \
                            or PuzzleSegment._is_piece_map_location_empty(piece_map, adjacent_loc.row, adjacent_loc.column):
                        continue

                    # Get the piece in the specified location
                    piece_id = piece_map[adjacent_loc.row, adjacent_loc.column]
                    key = PuzzleSegmentPieceInfo.create_key(piece_id)
                    # The piece is in the frontier, mark its distance
                    if key in frontier_pieces:
                        # Delete piece from the frontier and add to the explored pieces
                        del frontier_pieces[key]
                        explored_pieces[key] = piece_id

                        # Update the piece's distance
                        self._update_piece_distance_to_open(piece_id, distance_to_open)
                        current_generation_locations.append(PuzzleLocation(puzzle_id=-1, row=adjacent_loc.row, column=adjacent_loc.col))

            # Store the previous generation locations
            previous_generation_locations = current_generation_locations
            distance_to_open += 1  # At the end of the

    def _update_piece_distance_to_open(self, piece_id, distance_to_open):
        """
        Updates the distance to open value for a piece

        Args:
            piece_id (int): Identification number of the piece to be updated
            distance_to_open (int): Distance between the piece and an open space in the board
        """
        key = PuzzleSegmentPieceInfo.create_key(piece_id)
        # Read the existing value, modify it, then write it back
        piece_info = self._pieces[key]
        piece_info.distance_from_open_space = distance_to_open
        self._pieces[key] = piece_info

    @staticmethod
    def _find_open_spaces_adjacent_to_piece(piece_map):
        """
        This function scans the mapping of puzzle pieces in the segment to their relative location and returns a list
        of those puzzle locations that have no piece in them but do have a piece directly adjacent to them.

        Args:
            piece_map (Numpy[int]): Relative mapping of piece locations to their place on the board.

        Returns (PuzzleLocation): Locations in the relative map that have no pieces but are ADJACENT to puzzle
            pieces
        """

        # Manually curried function to reduce code duplication
        is_map_loc_empty = functools.partial(PuzzleSegment._is_piece_map_location_empty, piece_map)

        # Get board size
        numb_rows, numb_cols = piece_map.shape

        open_locations_with_adjacent_piece = []

        # Find all the locations that are blank but have an adjacent piece
        for row in xrange(0, numb_rows):
            for col in xrange(0, numb_cols):
                # If the location is blank, then definitely empty
                if not is_map_loc_empty(piece_map, row, col):
                    continue

                # Check above, below, left, and right locations (in that order) for a piece
                if not is_map_loc_empty(row - 1, col) or not is_map_loc_empty(row + 1, col) \
                        or not is_map_loc_empty(row, col - 1) or not is_map_loc_empty(row, col + 1):

                    # Create a new PuzzleLocation and add to the list
                    open_locations_with_adjacent_piece.append(PuzzleLocation(puzzle_id=-1, row=row, column=col))

        # Since the piece map has a border around it, it should have at least 4 blank locations (one on top, one
        # on bottom, one on the left and one on the right)
        if PuzzleSegment._PERFORM_ASSERT_CHECKS:
            assert len(open_locations_with_adjacent_piece) >= 4

        return open_locations_with_adjacent_piece

    @staticmethod
    def _is_piece_map_location_valid(piece_map, row, col):
        """
        Checks whether the combination of the row and column return a valid location on the piece map matrix

        Args:
            piece_map (Numpy[int]): Relative of piece locations in the segment
            row (int): Row in the piece map to check
            col (int): Column in the piece map to check

        Returns (bool): True if the piece location exists in the piece map and False otherwise

        """
        if row < 0 or row >= piece_map.shape[0] or col < 0 or col >= piece_map.shape[1]:
            return False
        return True

    @staticmethod
    def _is_piece_map_location_empty(piece_map, row, col):
        """
        Checks whether the piece_map location is blank or is a populated with a piece.  If the specified location
        does not exist in the piece map (as it overflows the board), this is considered an empty location.

        Args:
            piece_map (Numpy[int]): Relative of piece locations in the segment
            row (int): Row in the piece map to check
            col (int): Column in the piece map to check

        Returns (bool): True if the specified location (i.e., row and column) has no piece and False otherwise.
        """
        if not PuzzleSegment._is_piece_map_location_valid(piece_map, row, col):
            return True

        return piece_map[row, col] == PuzzleSegment._EMPTY_PIECE_MAP_VALUE

    def _build_segment_piece_map(self):
        """
        Creates a relative mapping of puzzle pieces in the segment to each other.  THis is used for determining
        each piece's distance from an open location.

        Returns (Numpy[int]): Map of all puzzle pieces in their relative locations
        """
        # Find the two extreme corners of the
        [top_left, bottom_right] = self._find_top_left_and_bottom_right_corner_of_segment_map(border_padding=1)

        # Get the width and height of the board
        segment_map_width = bottom_right.col - top_left.col + 1
        segment_map_height = bottom_right.row - top_left.row + 1

        # Build the piece map
        piece_map = np.full((segment_map_height, segment_map_width), fill_value=PuzzleSegment._EMPTY_PIECE_MAP_VALUE,
                            dtype=np.int32)

        # Put the pieces into their respective slot
        for piece_info in self._pieces:
            row = piece_info.location.row - top_left.row
            column = piece_info.location.column - top_left.column
            piece_map[row, column] = piece_info.piece_id

        return piece_map

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

    def get_piece_color(self, piece_id):
        """
        Used to get the color of a puzzle piece based off the either the segment's default color or coloring based
        off the distance from an open space.

        Args:
            piece_id (int): Identification number of the puzzle piece

        Returns (List[int]): BGR representation of the color for the piece in the segment.
        """
        # If not using distance based color, no need to get the piece specific information
        if not PuzzleSegment._USE_DISTANCE_FROM_EDGE_BASED_COLORING:
            return self._color.value

        # If piece distance to open data is not valid, then calculate it.
        if not self._piece_distance_from_open_space_up_to_date:
            self._calculate_piece_distances_to_open_location()

        key = PuzzleSegmentPieceInfo.create_key(piece_id)
        piece_info = self._pieces[key]
        return piece_info.get_color_based_on_distance_from_an_open()
