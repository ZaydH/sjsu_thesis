import Queue
import sys
import functools

import cv2
import numpy as np
from enum import Enum
import math
import copy

from hammoudeh_puzzle import config
from hammoudeh_puzzle.puzzle_piece import PuzzlePiece
from hammoudeh_puzzle.solver_helper import PuzzleLocation


class SegmentColor(Enum):
    """
    Valid segment colors.  These are represented in BGR format.
    """
    # Dark_Red = (0x22, 0x22, 0xB2)
    Dark_Red = (0x0, 0x0, 0x80)
    Dark_Green = (0x0, 0x34, 0x0)
    Dark_Blue = (0x80, 0x0, 0x0)
    Dark_Yellow = (0x00, 0x33, 0x66)

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
        if PuzzleSegment.USE_DISTANCE_FROM_EDGE_BASED_COLORING:
            return [SegmentColor.Dark_Red, SegmentColor.Dark_Blue, SegmentColor.Dark_Green, SegmentColor.Dark_Yellow]
        else:
            return [SegmentColor.Red, SegmentColor.Blue, SegmentColor.Green, SegmentColor.Yellow]

    def key(self):
        """
        Generates a key for the color for using the color in a dictionary.

        Returns (string): A string key for use of the segment color in a dictionary.

        """
        return str(self.value)


class Stack(object):

    def __init__(self):
        self._contents = []

    def empty(self):
        """
        Gets whether the stack is empty.

        Returns (bool): True if the stack is empty and False otherwise.
        """
        return len(self._contents) == 0

    def push(self, value):
        """
        Puts a value on the top of the stack

        Args:
            value: Value to put at the top of the stack.
        """
        self._contents.append(value)

    def pop(self):
        """
        Removes the top of the stack and returns it.

        Returns:
            Object at the top of the stack.
        """
        return self._contents.pop()

    def peek(self):
        """
        Gets the top of the stack but does not remove it from the stack

        Returns:
            Object at the top of the stack.
        """
        last_index = len(self._contents) - 1
        return self._contents[last_index]


class DepthFirstSearchNode(object):

    def __init__(self, piece_id, puzzle_location, parent_id, depth):
        """
        Creates a Node for the Depth First Search of the Segment

        Args:
            piece_id (int): Identification number of the piece associated with the node
            puzzle_location (PuzzleLocation): Location of the piece.
            parent_id (int): Identification number of this node's parent.
            depth (int): Depth of the node in the tree.
        """
        self.piece_id = piece_id
        self.key = str(self.piece_id)

        self.location = puzzle_location

        self.is_articulation_point = False

        self._child_count = 0
        self._parent_id = parent_id

        self._last_child_id = None
        self.reset_child_id()

        self._depth = depth
        self._lowpoint = depth

    def create_child(self, piece):
        """
        Creates a child node from a parent node.

        Args:
            piece (SegmentPieceInfo): Puzzle piece associated with the child node

        Returns (DepthFirstSearchNode): Child node
        """
        self._increment_child_count()
        return DepthFirstSearchNode(piece.piece_id, piece.location,
                                    parent_id=self.piece_id, depth=self._depth + 1)

    @staticmethod
    def define_root(piece):
        """
        Creates the root of the DFS tree.

        Args:
            piece (SegmentPieceInfo): Puzzle piece associated with the root node

        Returns (DepthFirstSearchNode):
            Root node using the specified piece as the seed.
        """
        # noinspection PyTypeChecker
        return DepthFirstSearchNode(piece.piece_id, piece.location, parent_id=None, depth=0)

    def update_lowpoint(self, adjacent_node, check_is_articulation):
        """
        Updates the low point of the node.

        If appropriate, this function will also update if the point is an articulation point.

        Args:
            adjacent_node (DepthFirstSearchNode): Node adjacent to the implicit node
            check_is_articulation (bool): If True, then the function checks (and if necessary updates) whether the
               implicit piece is an articulation piece.
        """
        if check_is_articulation:
            if not self.is_root():  # Root's condition for articulation is more than one child node.
                self.is_articulation_point = self.is_articulation_point or adjacent_node._lowpoint >= self._depth
            self._lowpoint = min(self._lowpoint, adjacent_node._lowpoint)
        else:
            self._lowpoint = min(self._lowpoint, adjacent_node._depth)

    def _increment_child_count(self):
        """
        Increments the child count for node.

        Also, will set the root as articulation if appropriate.
        """
        self._child_count += 1
        if self._parent_id is None and self._child_count > 1:
            self.is_articulation_point = True

    def is_root(self):
        """
        Checks if the implicit node is the root of the DFS tree.

        Returns (bool): True if the piece is the root of the tree and False otherwise.
        """
        return self._parent_id is None

    def has_parent(self, piece_id):
        """
        Checks whether the specified piece is this node's parent.

        Args:
            piece_id (int): Identification number of a piece to check if it is a parent.

        Returns (bool): True if the specified piece identification number corresponds to the parent and False
        otherwise.
        """
        if self._parent_id is None:
            return False

        return self._parent_id == piece_id

    @staticmethod
    def create_key(piece_id):
        """
        Creates a key object for Depth First Search node.

        Args:
            piece_id (int): Puzzle piece identification for the node.

        Returns (str): Key for a node.
        """
        return str(piece_id)

    @property
    def last_child_id(self):
        """
        Gets the last identification number of the last child piece.

        This is used in the iterative version of the depth first search for articulation pieces.

        Returns (int): Identification number of the last child.  If there is no last child, this returns "None".
        """
        return self._last_child_id

    @last_child_id.setter
    def last_child_id(self, child_piece_id):
        """
        Sets the identification of the last child piece.

        This is used in the iterative version of the depth first search for articulation pieces.

        Args:
            child_piece_id (int): Identification number of the last child piece.
        """
        self._last_child_id = child_piece_id

    def has_last_child_id(self):
        """
        Checks whether the node has a last child node.

        This is used in the iterative version of the depth first search for articulation pieces.

        Returns (bool): True if the node has a last child and false otherwise.
        """
        return self._last_child_id is not None

    def reset_child_id(self):
        """
        Clears the last child identification number,

        This is used in the iterative version of the depth first search for articulation pieces.
        """
        self._last_child_id = None

    def is_last_child_id(self, piece_id):
        """
        Checks whether the specified piece identification number corresponds with this node last child identification
        number.

        This is used in the iterative version of the depth first search for articulation pieces.

        Args:
            piece_id (int): Identification number of the piece that may be the child

        Returns (bool): "True" if the specified piece identification number corresponds with the last child and
            "False" otherwise.
        """
        return self._last_child_id == piece_id


class SegmentGridLocation(object):
    def __init__(self, grid_row, grid_column):
        """
        Simple structure for representing a segment grid cell

        Args:
            grid_row (int): Row where the segment grid cell is located
            grid_column (int): Column where the segment grid cell is located
        """
        self.grid_row = grid_row
        self.grid_column = grid_column


class SegmentGridCell(object):
    _PERFORM_ASSERT_CHECKS = True
    
    def __init__(self, grid_location, grid_center):
        """
        Grid cell information is used
        Args:
            grid_location (SegmentGridLocation): Location of this cell in the segment
            grid_center (PuzzleLocation): Center of the grid with respect to the solved puzzle
        """
        self._grid_location = grid_location

        self._grid_center = grid_center

        self._has_cell_next_to_open = False

        self._has_stitching_piece = False

        self._piece_id_at_distance = []

    @property
    def grid_location(self):
        """
        Gets the location of this grid cell in the segment's grid.  This is used when determining which piece
        to use for the neighbor solver.

        Returns (SegmentGridLocation): Location of the this cell in the segment grid
        """
        return self._grid_location

    @property
    def has_cell_next_to_open(self):
        """
        Stores whether the grid cell has a puzzle piece next to a blank space

        Returns (bool): True if the grid cell has a puzzle piece next to an open space and False otherwise
        """
        return self._has_cell_next_to_open

    @has_cell_next_to_open.setter
    def has_cell_next_to_open(self, value):
        """
        Updates whether the grid cell has a puzzle piece next to a blank space

        Args:
            value (bool): True if the grid cell has a puzzle piece next to an open space and False otherwise
        """
        if not isinstance(value, bool):
            raise ValueError("Value passed must be a boolean")
        self._has_cell_next_to_open = value

    @staticmethod
    def calculate_grid_cell_center(piece_map_top_left, grid_location, piece_map_shape, grid_size):
        """
        Calculates the center of the grid cell relative to the solved puzzle.  This allows for calculation of the
        distance of each piece to the center of the segment cell.

        Args:
            piece_map_top_left (PuzzleLocation):
            grid_location (SegmentGridLocation): Grid cell location in the segment
            piece_map_shape (Tuple[int]): Dimension of the piece map (piece rows by piece columns)
            grid_size (Tuple[int]): Dimension of the segment grid (cell rows by cell columns)

        Returns (PuzzleLocation): Puzzle location at the center of the grid cell
        """
        top_left_dim = (piece_map_top_left.row, piece_map_top_left.column)
        grid_dim = (grid_location.grid_row, grid_location.grid_column)

        # Calculate row center first then column center second
        center_values = []
        for i in xrange(0, len(grid_dim)):

            if grid_size[i] - 1 == grid_dim[i]:
                # Find actual center if dimension is smaller than normal width
                length_grid = (piece_map_shape[i] - 2 * PuzzleSegment.PIECE_MAP_BORDER_WIDTH)
                length_grid -= grid_dim[i] * PuzzleSegment.NEIGHBOR_SEGMENT_GRID_CELL_WIDTH
                center = int(length_grid / 2)
            else:
                # Use middle of the dimension
                center = int(0.5 * PuzzleSegment.NEIGHBOR_SEGMENT_GRID_CELL_WIDTH)

            # Add the top left offset and the offset for the location in the grid
            center += top_left_dim[i] + PuzzleSegment.PIECE_MAP_BORDER_WIDTH
            center += (grid_dim[i] * PuzzleSegment.NEIGHBOR_SEGMENT_GRID_CELL_WIDTH)
            center_values.append(center)

        return PuzzleLocation(piece_map_top_left.puzzle_id, center_values[0], center_values[1])

    @property
    def has_stitching_piece(self):
        """
        Gets whether the specified piece is a stitching piece

        Returns (bool): True if the associated piece is a stitching piece and False otherwise.
        """
        return self._has_stitching_piece

    @has_stitching_piece.setter
    def has_stitching_piece(self, value):
        """
        Sets whether the specified piece is a stitching piece

        Args:
            value (bool): True if the associated piece is a stitching piece and False otherwise.
        """
        if not isinstance(value, bool):
            raise ValueError("value must be of type boolean.")
        self._has_stitching_piece = value

    def add_piece_at_distance_from_open(self, distance_from_open, piece_id, piece_location):
        """
        Adds a piece at the
        Args:
            distance_from_open (int): Distance the piece is from the nearest open space
            piece_id (int): Identification number of piece
            piece_location (PuzzleLocation): Location of the specified piece
        """
        if not isinstance(distance_from_open, int):
            raise ValueError("Distance from an open space must be an integer.")
        if not isinstance(piece_id, int):
            raise ValueError("Piece id must be an integer.")

        # Mark that the segment has a piece now next to an open
        if distance_from_open == 1:
            self.has_cell_next_to_open = True

        # Add blank cells at the specified distance (if needed)
        while len(self._piece_id_at_distance) < distance_from_open + 1:  # Add 1 since this is the index to use
            self._piece_id_at_distance.append([])

        self._piece_id_at_distance[distance_from_open].append((piece_id, piece_location))

    def determine_stitching_piece_id(self):
        """
        Determines the identification number of the piece to be used as the stitching piece for this grid cell.  It uses
        the maximum distance from the border specified in the PuzzleSegment class as the ideal distance.  If no pieces
        are at that distance, it then moves in one level.

        If there are multiple pieces at the same distance from the edge, then the piece that is closest to the center
        of the grid cell is used as the stitching piece.

        Returns (int): Identification number of the piece to be used as the stitching piece for this grid cell
        """
        if SegmentGridCell._PERFORM_ASSERT_CHECKS:
            assert self.has_cell_next_to_open

        # Find the distance that will be used as the baseline for the stitching piece
        starting_dist = min(PuzzleSegment.MAXIMUM_DISTANCE_FROM_OPEN_FOR_STITCHING_PIECE,
                            len(self._piece_id_at_distance) - 1)
        for stitching_dist in xrange(starting_dist, 0, -1):
            if len(self._piece_id_at_distance[stitching_dist]) > 0:
                break

        # Select the piece closest to the center of the grid cell
        best_distance = None
        stitching_piece = None
        # noinspection PyUnboundLocalVariable
        for piece_id, piece_loc in self._piece_id_at_distance[stitching_dist]:
            piece_dist = self._grid_center.calculate_manhattan_distance_from(piece_loc)
            if best_distance is None or best_distance > piece_dist:
                stitching_piece = piece_id
                best_distance = piece_dist

        if self._PERFORM_ASSERT_CHECKS:
            assert stitching_piece is not None
        return stitching_piece


class SegmentPieceInfo(object):
    _PERFORM_ASSERT_CHECKS = True

    DISTANCE_FROM_EDGE_DEFAULT_VALUE = sys.maxint

    MAXIMUM_SEGMENT_COLOR_LIGHTNESS = 225  # Done to prevent just white sections in the image

    LIGHTNESS_INCREASE_PER_PIECE_AWAY_FROM_OPEN = 35

    def __init__(self, piece_id, location):
        """
        Constructor for SegmentPieceInfo Objects

        Args:
            piece_id (int): Identification number of the piece
            location (PuzzleLocation): Location of the puzzle piece
        """

        self._piece_id = piece_id
        self._location = location

        self._is_stitching_piece = False
        self._key = SegmentPieceInfo.create_key(piece_id)

        self._distance_from_open_space = SegmentPieceInfo.DISTANCE_FROM_EDGE_DEFAULT_VALUE

        # color information for the piece
        self._default_color = None
        self._distance_based_color = None

        # Used for building the stitching pieces
        self._stitching_grid_cell_loc = None

    @property
    def piece_id(self):
        """
        Accessor for the associated puzzle piece's identification number.

        Returns (int): Piece identification
        """
        return self._piece_id

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

    def reset_puzzle_id(self):
        """
        Accesses the puzzle piece information's location

        Returns (PuzzleLocation): Puzzle piece's location
        """
        self._location.puzzle_id = None

    @property
    def is_stitching_piece(self):
        """
        Gets whether the piece is a stitching piece for its segment

        Returns (bool): True if the piece is a stitching piece, and False otherwise

        """
        return self._is_stitching_piece

    @is_stitching_piece.setter
    def is_stitching_piece(self, stitching_piece_value):
        """
        Sets whether the piece is a stitching piece for its segment

        Args:
            stitching_piece_value (bool): Indicates whether the piece is a stitching piece
        """
        if not isinstance(stitching_piece_value, bool):
            raise ValueError("stitching_piece_value must be of type bool")
        self._is_stitching_piece = stitching_piece_value

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
            new_default_color (SegmentColor): Default color assigned to all pieces in the segment to
                also be assigned to this piece.
        """
        if SegmentPieceInfo._PERFORM_ASSERT_CHECKS:
            assert new_default_color is not None

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

        self._distance_from_open_space = new_distance

    @property
    def stitching_grid_cell_location(self):
        """
        When determining segment similarity,

        Returns (SegmentGridLocation): The grid location for the piece in the segmentation map
        """
        if SegmentPieceInfo._PERFORM_ASSERT_CHECKS:
            assert self._stitching_grid_cell_loc is not None
        return self._stitching_grid_cell_loc

    @stitching_grid_cell_location.setter
    def stitching_grid_cell_location(self, new_grid_location):
        """
        Updates the grid cell location for this puzzle piece.

        Args:
            new_grid_location (SegmentGridLocation): New grid location for the piece in the segment
        """
        self._stitching_grid_cell_loc = new_grid_location

    def distance_based_color(self):
        """
        Gets the calculated distance based color for the puzzle piece information.

        Returns (List[int]): Color of the piece with a lightening based off the piece's distance from an open space
        """
        if self._distance_based_color is not None:
            self.determine_color_based_on_distance_from_an_open()
        return self._distance_based_color

    def determine_color_based_on_distance_from_an_open(self):
        """
        Returns a modified version of the puzzle segment's default color where the color is reflective of the
        distance the piece is from an open space.

        Returns (List[int]): Color of the piece with a lightening based off the piece's distance from an open space
        """

        # If the distance based color is already stored, then no need to recalculate
        if self._distance_based_color is not None:
            return self._distance_based_color

        # Create an image to use the
        color_img = np.uint8([[self._default_color.value]])

        # Convert color to lab
        lab_color = cv2.cvtColor(color_img, cv2.COLOR_BGR2HLS)

        # Subtract one since each piece is a minimum of 1 away from open (otherwise it is open)
        lightness_increase = (self._distance_from_open_space - 1) * SegmentPieceInfo.LIGHTNESS_INCREASE_PER_PIECE_AWAY_FROM_OPEN

        lightness_index = 1  # Index in the NumPy array corresponding to the lightness dimension
        new_lightness = lab_color[0, 0, lightness_index] + lightness_increase

        # Calculate a new lightness but do not let it be too light
        new_lightness = min(SegmentPieceInfo.MAXIMUM_SEGMENT_COLOR_LIGHTNESS, new_lightness)
        lab_color[0, 0, lightness_index] = np.uint8(new_lightness)

        # Calculate a new BGR that is lightened and then return it
        new_bgr = cv2.cvtColor(lab_color, cv2.COLOR_HLS2BGR)

        # Change color from a NumPy matrix to a tuple of integers so it plays nice with OpenCV
        output_color = []
        for i in xrange(0, new_bgr.shape[2]):
            output_color.append(int(new_bgr[0, 0, i]))
        self._distance_based_color = tuple(output_color)

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

    _PERFORM_ASSERT_CHECKS = config.PERFORM_ASSERT_CHECKS

    _EMPTY_PIECE_MAP_VALUE = -1

    USE_DISTANCE_FROM_EDGE_BASED_COLORING = True

    PIECE_MAP_BORDER_WIDTH = 1

    NEIGHBOR_SEGMENT_GRID_CELL_WIDTH = 10

    MAXIMUM_DISTANCE_FROM_OPEN_FOR_STITCHING_PIECE = 3

    def __init__(self, puzzle_id, segment_id_number):
        """
        Puzzle segment constructor.

        Args:
            puzzle_id (int): Identification number of the solved puzzle
            segment_id_number (int): Unique identification number for the segment.
        """
        self._puzzle_id = puzzle_id
        self._segment_id_number = segment_id_number
        self._pieces = {}
        self._removed_pieces = {}
        self._piece_id_list = None
        self._seed_piece = None

        self._reset_neighbor_info()

        self._color = None  # Needed to prevent PyCharm __init__ error.  Not actually needed.
        self._reset_color()

        # Used to determine the stitching pieces to be used
        self._piece_map = None
        self._top_left = None
        self._piece_map_border_width = PuzzleSegment.PIECE_MAP_BORDER_WIDTH
        self._piece_groupings_by_distance_from_open = None
        self._neighbor_grid_cell_info = None

        self._piece_distance_from_open_space_up_to_date = False
        self._stitching_piece_ids = []

    @property
    def puzzle_id(self):
        """
        Each segment is only associated with a single solved puzzle.  This property is used to access the puzzle
        identification number of the implicitly associated segment.

        Returns (int): Identification number of the puzzle associated with this segment.

        """
        if PuzzleSegment._PERFORM_ASSERT_CHECKS:
            assert self._puzzle_id is not None
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
        return len(self._pieces)

    def get_piece_ids(self):
        """
        Gets the identification number of the pieces that are in this segment.

        Returns (List[int]): The identification number of the pieces in this segment.
        """
        # Rebuild piece ID list if needed
        if self._piece_id_list is None:
            self._piece_id_list = []
            for piece_info in self._pieces.values():
                self._piece_id_list.append(piece_info.piece_id)

        return self._piece_id_list

    def _reset_neighbor_info(self):
        """
        Re-initializes all data structures associated with the neighbor segments.
        """
        self._neighbor_segment_ids = {}
        self._neighbor_colors = {}  # This is used for coloring the graph to quickly determine neighbor colors

    def _reset_color(self):
        """
        Resets the color attribute of the segment to the default fault.
        """
        self._color = None

    def update_segment_for_multipuzzle_solver(self, new_segment_id):
        """
        This function is used as part of the multi-puzzle solver.

        It reinitializes/resets the data structures associated with the puzzle segment when it is treated as an
        independent unit and no longer part of a solved puzzle.

        One of the primary things this entails is that the segment identification number may be changed.

        Args:
            new_segment_id (int): New identification number for the segment.
        """
        self._reset_neighbor_info()

        self._puzzle_id = None
        self._segment_id_number = new_segment_id

        self._reset_color()

        # Iterate through all the pieces and reset the puzzle location to None
        for key in self._pieces.keys():
            piece = self._pieces[key]
            piece.reset_puzzle_id()
            self._pieces[key] = piece

    def add_piece(self, piece):
        """
        Adds a puzzle piece to the puzzle segment.

        Args:
            piece (PuzzlePiece): Puzzle piece to be added
        """
        new_piece = SegmentPieceInfo(piece.id_number, piece.puzzle_location)

        # Store the seed piece special
        if len(self._pieces) == 0:
            self._seed_piece = new_piece

        # Since a piece was added, piece distance information no longer up to date
        self._piece_distance_from_open_space_up_to_date = False
        self._piece_id_list = None

        # Store all pieces in the piece dictionary
        self._pieces[new_piece.key] = new_piece

    def remove_piece(self, piece_id):
        """
        Removes a puzzle piece (as defined by the piece's identification number) from the puzzle segment.

        Args:
            piece_id (int): Identification if the puzzle piece to be removed from the segment
        """

        key = SegmentPieceInfo.create_key(piece_id)
        # Optionally ensure the key exists before trying to remove it
        if PuzzleSegment._PERFORM_ASSERT_CHECKS:
            assert key in self._pieces  # verify the piece to be deleted actually exists
            assert piece_id != self._seed_piece.piece_id   # Cannot delete the seed piece

        piece_location = self._pieces[key].location

        # Since a piece was removed, piece distance information no longer up to date
        self._piece_distance_from_open_space_up_to_date = False
        self._piece_id_list = None
        del self._pieces[key]

        # Update the list of removed pieces.
        if PuzzleSegment._PERFORM_ASSERT_CHECKS:
            assert key not in self._removed_pieces
        self._removed_pieces[key] = piece_id

        self._mark_piece_map_location_open(piece_location)

    def remove_articulation_points_and_disconnected_pieces(self, is_pieces_best_buddies_func=None):
        """
        Removes from the segment any articulation points as well as those pieces that become disconnected from the seed
        upon removal of the articulation points.

        Note that if the seed is an articulation point, all other pieces except the seed are removed from the
        segment.

        Args:
            is_pieces_best_buddies_func: Function used to check if two pieces are best buddies.

        Returns (List[int]): List of all pieces removed from the segment.
        """

        self._build_piece_map()

        articulation_points = self._find_articulation_points(is_pieces_best_buddies_func)

        # If no articulation pieces, this can be removed.
        if not articulation_points:
            return self._build_removed_pieces_list()

        # Remove the articulation pieces
        for node in articulation_points:
            # Cannot remove the seed piece so it becomes its own segment.
            if node.piece_id == self._seed_piece.piece_id:
                return self._remove_all_pieces_except_seed()
            self.remove_piece(node.piece_id)

        self._remove_disconnected_pieces(is_pieces_best_buddies_func)

        return self._build_removed_pieces_list()

    def _remove_disconnected_pieces(self, is_pieces_best_buddies_func=None):
        """
        Removes from the segment any pieces that are not reachable from the seed.

        Args:
            is_pieces_best_buddies_func: Function used to check if two pieces are best buddies.
        """

        # Anything in the unexplored set at the end is disconnected
        unexplored_set = {}
        for piece in self._pieces.values():
            if piece.piece_id != self._seed_piece.piece_id:
                unexplored_set[piece.key] = piece.piece_id

        # Run until no connected pool is empty
        connected_queue = Queue.Queue()
        connected_queue.put(self._seed_piece.location)
        while not connected_queue.empty():

            piece_location = connected_queue.get()
            current_piece_id = self._get_piece_at_segment_location(piece_location)

            # If the piece is adjacent and unexplored, remove from unexplored list
            for adjacent_id in self._get_location_adjacency_list(piece_location):

                if is_pieces_best_buddies_func is not None \
                        and not is_pieces_best_buddies_func(current_piece_id, adjacent_id):
                    continue

                adjacent_key = PuzzlePiece.create_key(adjacent_id)
                if adjacent_key in unexplored_set:
                    del unexplored_set[adjacent_key]
                    connected_queue.put(self._pieces[adjacent_key].location)

        # Remove the unexplored pieces
        for unexplored_piece_id in unexplored_set.values():
            self.remove_piece(unexplored_piece_id)

    def _find_articulation_points(self, is_pieces_best_buddies_func=None):
        """
        Gets the list of articulation points for a segment.

        Args:
            is_pieces_best_buddies_func: Function used to check if two pieces are best buddies.

        Returns (DepthFirstSearchNode): List of the nodes that are articulation points.
        """

        # Build the root of the tree
        tree_root = DepthFirstSearchNode.define_root(self._seed_piece)
        dfs_tree = {tree_root.key: tree_root}

        # Perform depth first search to find the articulation points.
        PuzzleSegment._depth_first_search_for_articulation_points_iterative(self, dfs_tree, tree_root,
                                                                            is_pieces_best_buddies_func)

        # Build and return the list of articulation points.
        return [node for node in dfs_tree.values() if node.is_articulation_point]

    @staticmethod
    def _depth_first_search_for_articulation_points_recursive(puzzle_segment, dfs_tree, current_node,
                                                              is_pieces_best_buddies_func=None):
        """
        Performs depth first search to find the articulation points (if any).

        Based off the code here: https://en.wikipedia.org/wiki/Biconnected_component

        Args:
            puzzle_segment (PuzzleSegment): Puzzle segment being analyzed
            dfs_tree (dict): Representation of DFS tree as a dictionary
            current_node (DepthFirstSearchNode): Current node in the DFS tree
            is_pieces_best_buddies_func: Function used to check if two pieces are best buddies.
        """
        if PuzzleSegment._PERFORM_ASSERT_CHECKS:
            assert current_node.piece_id == puzzle_segment._get_piece_at_segment_location(current_node.location)

        for adjacent_id in puzzle_segment._get_location_adjacency_list(current_node.location):
            
            if is_pieces_best_buddies_func is not None \
                    and not is_pieces_best_buddies_func(current_node.piece_id, adjacent_id):
                continue
            # # Always skip the parent.
            # if current_node.has_parent(adjacent_id):
            #     continue

            adjacent_piece_key = DepthFirstSearchNode.create_key(adjacent_id)
            # Check if the piece is visited
            try:
                adjacent_node = dfs_tree[adjacent_piece_key]
                current_node.update_lowpoint(adjacent_node, check_is_articulation=False)

            # Piece not visited
            except KeyError:
                child_node = current_node.create_child(puzzle_segment._pieces[adjacent_piece_key])
                dfs_tree[child_node.key] = child_node
                PuzzleSegment._depth_first_search_for_articulation_points_recursive(puzzle_segment,
                                                                                    dfs_tree,
                                                                                    child_node,
                                                                                    is_pieces_best_buddies_func)
                current_node.update_lowpoint(child_node, check_is_articulation=True)

    def _depth_first_search_for_articulation_points_iterative(self, dfs_tree, root_node,
                                                              is_pieces_best_buddies_func=None):
        """
        Performs depth first search to find the articulation points (if any).

        Based off the code here: https://en.wikipedia.org/wiki/Biconnected_component

        Args:
            dfs_tree (dict): Representation of DFS tree as a dictionary
            root_node (DepthFirstSearchNode): Root of the DFS tree
            is_pieces_best_buddies_func: Function used to check if two pieces are best buddies.
        """

        explored_piece_count = 0

        # Create a stack and push the current key onto it.
        piece_key_stack = Stack()
        piece_key_stack.push(root_node.key)

        # Run the solver until all pieces reached (i.e. stack is empty)
        while not piece_key_stack.empty():

            # Get the node off the stop of the stack.
            current_piece_key = piece_key_stack.peek()
            current_node = dfs_tree[current_piece_key]
            # Clear the recursive call if applicable.
            recursive_call_simulation = False

            # Ensure the location is valid
            if PuzzleSegment._PERFORM_ASSERT_CHECKS:
                assert current_node.piece_id == self._get_piece_at_segment_location(current_node.location)

            # Iterate through all neighbors of the piece
            for adjacent_id in self._get_location_adjacency_list(current_node.location):

                # Only consider best buddies
                if is_pieces_best_buddies_func is not None \
                        and not is_pieces_best_buddies_func(current_node.piece_id, adjacent_id):
                    continue

                # Simulate what would have been done when returning from the recursive call.
                if current_node.has_last_child_id():
                    # Update the low point
                    if current_node.is_last_child_id(adjacent_id):
                        child_node = dfs_tree[PuzzlePiece.create_key(current_node.last_child_id)]
                        current_node.update_lowpoint(child_node, check_is_articulation=True)
                        current_node.reset_child_id()
                    continue

                adjacent_piece_key = DepthFirstSearchNode.create_key(adjacent_id)
                # Check if the piece is visited
                if adjacent_piece_key in dfs_tree:
                    adjacent_node = dfs_tree[adjacent_piece_key]
                    current_node.update_lowpoint(adjacent_node, check_is_articulation=False)

                # Piece not visited
                else:
                    current_node.last_child_id = adjacent_id

                    # Create the child node
                    child_node = current_node.create_child(self._pieces[adjacent_piece_key])
                    dfs_tree[child_node.key] = child_node

                    # Simulate making the recursive call
                    piece_key_stack.push(child_node.key)
                    recursive_call_simulation = True
                    break

            # Pop off the stack
            if not recursive_call_simulation:
                piece_key_stack.pop()
                explored_piece_count += 1

        # Verify all pieces explored.
        if PuzzleSegment._PERFORM_ASSERT_CHECKS:
            assert explored_piece_count == self.numb_pieces

    def _remove_all_pieces_except_seed(self):
        """
        Updates the segment and removes all non-seed pieces from the segment.

        Returns (List[str]): List of the identification numbers of the pieces removed from the segment.
        """
        # Use a copy of the piece list since it is going to be updated as the list runs
        all_piece_ids = copy.copy(self.get_piece_ids())

        # Delete everything but the seed piece
        for piece_id in all_piece_ids:
            if self._seed_piece.piece_id == piece_id:
                continue
            self.remove_piece(piece_id)
        return self._build_removed_pieces_list()

    def _build_removed_pieces_list(self):
        """
        Builds a list of the pieces removed from the list.

        Returns (List[int]): List of the identification numbers of the pieces removed from the segment.
        """
        return [piece_id for piece_id in self._removed_pieces.values()]

    def _get_location_adjacency_list(self, puzzle_location):
        """
        Gets all of the puzzle pieces at locations adjacent to the specified location.

        Args:
            puzzle_location (PuzzleLocation): Location in the segment

        Returns (List[int]): Identification number (if any) of the adjacent puzzle pieces.
        """
        adjacency_list = []
        for adjacent_location in puzzle_location.get_adjacent_locations():
            if self._does_location_have_piece(adjacent_location):
                adjacency_list.append(self._get_piece_at_segment_location(adjacent_location))

        return adjacency_list

    def _get_piece_at_segment_location(self, puzzle_location):
        """
        Gets the puzzle piece at the specified location

        Args:
            puzzle_location (PuzzleLocation): Location inside the puzzle segment

        Returns (int): Identification number of the piece at the specified location.
        """
        row, col = self._get_location_piece_map_row_and_column(puzzle_location)

        piece_id = self._piece_map[row, col]
        if config.PERFORM_ASSERT_CHECKS:
            assert piece_id != PuzzleSegment._EMPTY_PIECE_MAP_VALUE
        return piece_id

    def _get_location_piece_map_row_and_column(self, puzzle_location):
        """
        Normalizes the specified puzzle location based off the top left row and column

        Args:
            puzzle_location ():

        Returns (Tuple[int]): Two element Tuple in the form (row, column)
        """
        # Get the adjusted row and column
        row = puzzle_location.row - self._top_left.row
        if PuzzleSegment._PERFORM_ASSERT_CHECKS:
            assert row >= 0

        col = puzzle_location.column - self._top_left.column
        if PuzzleSegment._PERFORM_ASSERT_CHECKS:
            assert col >= 0

        return row, col

    def _update_piece_map_location(self, puzzle_location, piece_id):
        """
        Updates the segment's piece map by updating the piece identification number at the specified puzzle location.

        Args:
            puzzle_location (PuzzleLocation): Location in the segment
            piece_id (int): Identification number of the piece in the specified segment location
        """
        # Get the adjust row and column
        row, col = self._get_location_piece_map_row_and_column(puzzle_location)
        self._piece_map[row, col] = piece_id

    def _mark_piece_map_location_open(self, puzzle_location):
        """
        Updates the piece map for the implicit segment at the specified location to be open.

        Args:
            puzzle_location (PuzzleLocation): Segment location to be marked as open.
        """
        if PuzzleSegment._PERFORM_ASSERT_CHECKS:
            assert self._does_location_have_piece(puzzle_location)

        self._update_piece_map_location(puzzle_location, PuzzleSegment._EMPTY_PIECE_MAP_VALUE)

    def _does_location_have_piece(self, puzzle_location):
        """
        Checks whether the specified location has a puzzle piece.

        Args:
            puzzle_location (PuzzleLocation): Location in the puzzle.

        Returns (bool):
            True if the location has a piece and False otherwise.
        """
        # Calculate the offset row and column
        row, col = self._get_location_piece_map_row_and_column(puzzle_location)
        if not PuzzleSegment._is_piece_map_location_valid(self._piece_map, row, col) \
                or PuzzleSegment._is_piece_map_location_empty(self._piece_map, row, col):
            return False
        return True

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

    def _determine_piece_distances_to_open_location(self):
        """
        Determines the distance each piece in the segment is from an open space.

        Once this distance is found for each piece, the SegmentPieceInfo of each is updated to store this distance
        for future analysis.
        """
        self._build_piece_map()

        explored_pieces = {}  # As pieces' minimum distance is found, mark as explored
        open_locations_next_to_piece = self._find_open_spaces_adjacent_to_piece(self._piece_map)
        self._piece_groupings_by_distance_from_open = [open_locations_next_to_piece]

        distance_to_open = 1  # If not a blank space, minimum of distance 1 away
        # Continue looping until all pieces have a distance found
        while len(explored_pieces) < len(self._pieces):
            current_generation_locations = []
            # The previous generation locations are used to find the distance for the next generation
            for prev_gen_location in self._piece_groupings_by_distance_from_open[distance_to_open - 1]:
                # Find pieces adjacent to the previous location to increment distance by 1
                for adjacent_loc in prev_gen_location.get_adjacent_locations(self._piece_map.shape):

                    # Ignore locations off the piece map or that is empty
                    if not PuzzleSegment._is_piece_map_location_valid(self._piece_map, adjacent_loc.row,
                                                                      adjacent_loc.column) \
                            or PuzzleSegment._is_piece_map_location_empty(self._piece_map, adjacent_loc.row,
                                                                          adjacent_loc.column):
                        continue

                    # Get the piece in the specified location
                    piece_id = self._piece_map[adjacent_loc.row, adjacent_loc.column]
                    key = SegmentPieceInfo.create_key(piece_id)
                    # The piece is in the frontier, mark its distance
                    if key not in explored_pieces:
                        # Delete piece from the frontier and add to the explored pieces
                        explored_pieces[key] = piece_id

                        # Update the piece's distance
                        self._update_piece_distance_to_open(piece_id, distance_to_open)
                        current_generation_locations.append(PuzzleLocation(puzzle_id=-1, row=adjacent_loc.row,
                                                                           column=adjacent_loc.column))

            # Store the previous generation locations
            self._piece_groupings_by_distance_from_open.append(current_generation_locations)
            distance_to_open += 1

        # Distance to open up to date
        self._piece_distance_from_open_space_up_to_date = True
        self._stitching_piece_ids = None  # Reset the stitching piece info

    def _update_piece_distance_to_open(self, piece_id, distance_to_open):
        """
        Updates the distance to open value for a piece

        Args:
            piece_id (int): Identification number of the piece to be updated
            distance_to_open (int): Distance between the piece and an open space in the board
        """
        key = SegmentPieceInfo.create_key(piece_id)
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
                if not is_map_loc_empty(row, col):
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

    def _build_piece_map(self):
        """
        Creates a relative mapping of puzzle pieces in the segment to each other.  This is used for determining
        each piece's distance from an open location.

        Returns (Numpy[int]): Map of all puzzle pieces in their relative locations
        """
        # Find the two extreme corners of the piece map
        [self._top_left, bottom_right] = self._find_top_left_and_bottom_right_corner_of_segment_map()

        # Get the width and height of the board
        segment_map_width = bottom_right.column - self._top_left.column + 1
        segment_map_height = bottom_right.row - self._top_left.row + 1

        # Build the piece map
        piece_map = np.full((segment_map_height, segment_map_width), fill_value=PuzzleSegment._EMPTY_PIECE_MAP_VALUE,
                            dtype=np.int32)

        # Put the pieces into their respective slot
        for piece_info in self._pieces.values():
            row = piece_info.location.row - self._top_left.row
            column = piece_info.location.column - self._top_left.column
            piece_map[row, column] = piece_info.piece_id

        self._piece_map = piece_map

    def _find_top_left_and_bottom_right_corner_of_segment_map(self):
        """
        Determines the puzzle width and length based on the information in the puzzle piece information objects.

        All information in this function is rebuilt when called so if the user plans to reuse this information they
        should store the values.

        Returns (Tuple[PuzzleLocation]): Tuple in the format (top_left, bottom_right) PuzzleLocation objects with
            the optionally specified padding.

        """
        if self._piece_map_border_width < 0:
            raise ValueError("Border padding must be greater than or equal to 0.  %d was specified."
                             % self._piece_map_border_width)

        top_left_corner = PuzzleLocation(self._puzzle_id, sys.maxint, sys.maxint)
        bottom_right_corner = PuzzleLocation(self._puzzle_id, -sys.maxint, -sys.maxint)

        # Get the pieces in the puzzle
        for piece_info in self._pieces.values():
            # Get the piece location
            piece_loc = piece_info.location

            # Check top left
            if top_left_corner.row > piece_loc.row:
                top_left_corner.row = piece_loc.row
            if top_left_corner.column > piece_loc.column:
                top_left_corner.column = piece_loc.column

            # Check bottom right
            if bottom_right_corner.row < piece_loc.row:
                bottom_right_corner.row = piece_loc.row
            if bottom_right_corner.column < piece_loc.column:
                bottom_right_corner.column = piece_loc.column

        # Add the padding
        top_left_corner.row -= self._piece_map_border_width
        top_left_corner.column -= self._piece_map_border_width

        bottom_right_corner.row += self._piece_map_border_width
        bottom_right_corner.column += self._piece_map_border_width

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
            neighbor_color (SegmentColor): Color for a neighboring segment
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
            neighbor_color (SegmentColor): A possible color for a neighbor

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
            segment_color (SegmentColor): New color for the segment

        """
        # A given segment should only be assigned to a color once
        if PuzzleSegment._PERFORM_ASSERT_CHECKS:
            assert self._color is None
            assert not self.has_neighbor_color(segment_color)  # Make sure no neighbor has this color

        self._color = segment_color

        # Modify in place the piece colors
        for key in self._pieces.keys():
            self._pieces[key].default_color = self._color

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
        if not PuzzleSegment.USE_DISTANCE_FROM_EDGE_BASED_COLORING:
            return self._color.value

        # If piece distance to open data is not valid, then calculate it.
        if not self._piece_distance_from_open_space_up_to_date:
            self._determine_piece_distances_to_open_location()

        key = SegmentPieceInfo.create_key(piece_id)
        return self._pieces[key].determine_color_based_on_distance_from_an_open()

    def is_piece_used_for_stitching(self, piece_id):
        """
        Gets whether the specified piece is used for stitching in this segment

        Args:
            piece_id (int): Identification number of the piece

        Returns (bool): True if it is a stitching piece and False otherwise.
        """
        key = SegmentPieceInfo.create_key(piece_id)
        return self._pieces[key].is_stitching_piece

    def select_pieces_for_segment_stitching(self):
        """
        This function returns the pieces to be used as stitching pieces for this segment.

        Returns (List[int]): Identification number of the pieces in this segment to be used as stitching pieces.
        """

        # If piece distance to open data is not valid, then calculate it.
        if self._piece_distance_from_open_space_up_to_date and self._stitching_piece_ids:
            return self._stitching_piece_ids

        if not self._piece_distance_from_open_space_up_to_date:
            self._determine_piece_distances_to_open_location()

        self._determine_pieces_grid_locations()

        # Calculate the grid size
        grid_size = []
        for i in xrange(0, len(self._piece_map.shape)):
            numb_cells_in_dim = 1.0 * (self._piece_map.shape[i] - 2 * PuzzleSegment.PIECE_MAP_BORDER_WIDTH)
            numb_cells_in_dim /= PuzzleSegment.NEIGHBOR_SEGMENT_GRID_CELL_WIDTH
            grid_size.append(int(math.ceil(numb_cells_in_dim)))
        row_index = 0
        column_index = 1

        # Build the grid cell information
        self._neighbor_grid_cell_info = []
        for grid_row in xrange(0, grid_size[row_index]):
            self._neighbor_grid_cell_info.append([])
            for grid_col in xrange(0, grid_size[column_index]):
                grid_cell_location = SegmentGridLocation(grid_row, grid_col)
                grid_cell_center = SegmentGridCell.calculate_grid_cell_center(self._top_left, grid_cell_location,
                                                                              self._piece_map.shape, grid_size)
                self._neighbor_grid_cell_info[grid_row].append(SegmentGridCell(grid_cell_location, grid_cell_center))

        # Assign each piece to a grid cell
        for dist in xrange(1, len(self._piece_groupings_by_distance_from_open)):  # Start from 1 since 0 is open spaces
            for relative_loc in self._piece_groupings_by_distance_from_open[dist]:
                piece_id = self._piece_map[relative_loc.row, relative_loc.column]
                # Get the key for the piece
                key = SegmentPieceInfo.create_key(piece_id)
                self._add_piece_to_grid_cell(dist, self._pieces[key])

        # Gets all of the stitching pieces
        self._stitching_piece_ids = []
        for grid_row in xrange(0, grid_size[row_index]):
            for grid_col in xrange(0, grid_size[column_index]):

                if self._neighbor_grid_cell_info[grid_row][grid_col].has_cell_next_to_open:
                    stitching_piece_id = self._neighbor_grid_cell_info[grid_row][grid_col].determine_stitching_piece_id()
                    self._stitching_piece_ids.append(stitching_piece_id)

                    # Mark the piece as a stitching piece
                    key = SegmentPieceInfo.create_key(stitching_piece_id)
                    self._pieces[key].is_stitching_piece = True

        return self._stitching_piece_ids

    def _add_piece_to_grid_cell(self, distance_from_open, piece_info):
        """
        Adds a piece to the grid cell.  Interface is a bit cumbersome so this makes a single method handle it

        Args:
            distance_from_open (int): Distance of the piece to the nearest open space
            piece_info (SegmentPieceInfo): Information on the piece to be added to the grid cell.  It also
              encapsulates the grid cell location
        """
        grid_cell_loc = piece_info.stitching_grid_cell_location

        grid_row = grid_cell_loc.grid_row
        grid_col = grid_cell_loc.grid_column

        piece_id = piece_info.piece_id
        piece_loc = piece_info.location
        self._neighbor_grid_cell_info[grid_row][grid_col].add_piece_at_distance_from_open(distance_from_open,
                                                                                          piece_id,
                                                                                          piece_loc)

    def _determine_pieces_grid_locations(self):
        """
        Calculates the grid locations for each piece in the segment.
        """
        # Iterate through all pieces and determine their grid cell lo
        for key in self._pieces.keys():
            piece_info = self._pieces[key]
            # Calculate the grid location
            piece_info.stitching_grid_cell_location = PuzzleSegment._calculate_piece_grid_location(piece_info.location,
                                                                                                   self._top_left)
            # Add the piece back to the dictionary
            self._pieces[key] = piece_info

    @staticmethod
    def _calculate_piece_grid_location(piece_location, top_left_location):
        """
        Helper function for calculating the grid location of a piece

        Args:
            piece_location (PuzzleLocation): Location of the piece to be placed in a grid
            top_left_location (PuzzleLocation): Relative location of the top corner of the piece map

        Returns (SegmentGridLocation): Grid location where the puzzle piece is located
        """
        # Calculate grid row and column
        relative_row = piece_location.row - top_left_location.row - PuzzleSegment.PIECE_MAP_BORDER_WIDTH
        grid_row = int(math.floor(relative_row / PuzzleSegment.NEIGHBOR_SEGMENT_GRID_CELL_WIDTH))

        relative_col = piece_location.column - top_left_location.column - PuzzleSegment.PIECE_MAP_BORDER_WIDTH
        grid_col = int(math.floor(relative_col / PuzzleSegment.NEIGHBOR_SEGMENT_GRID_CELL_WIDTH))

        return SegmentGridLocation(grid_row, grid_col)
