"""
Created by Zayd Hammoudeh (zayd.hammoudeh@sjsu.edu)
"""
import random

from enum import Enum
import numpy as np
import cv2  # OpenCV

from hammoudeh_puzzle.solver_helper_classes import PuzzleLocation


class Location(object):
    """
    Location Object

    Used to represent any two dimensional location in matrix row/column notation.
    """

    def __init__(self, (row, column)):
        self.row = row
        self.column = column


class PuzzlePieceRotation(Enum):
    """Puzzle Piece PieceRotation

    Enumerated type for representing the amount of rotation for a puzzle piece.

    Note:
        Pieces can only be rotated in 90 degree increments.

    """

    degree_0 = 0      # No rotation
    degree_90 = 90    # 90 degree rotation
    degree_180 = 180  # 180 degree rotation
    degree_270 = 270  # 270 degree rotation
    degree_360 = 360

    @staticmethod
    def all_rotations():
        """
        All Rotation Accessor

        Gets a list of all supported rotations for a puzzle piece.  The list is ascending from 0 degrees to 270
        degrees increasing.

        Returns ([PuzzlePieceRotation]):
        List of all puzzle rotations.
        """
        return [PuzzlePieceRotation.degree_0, PuzzlePieceRotation.degree_90,
                PuzzlePieceRotation.degree_180, PuzzlePieceRotation.degree_270]

    @staticmethod
    def random_rotation():
        """
        Random Rotation

        Generates and returns a random rotation.

        Returns (PuzzlePieceRotation):
        A random puzzle piece rotation
        """
        return random.choice(PuzzlePieceRotation.all_rotations())


class PuzzlePieceSide(Enum):
    """Puzzle Piece Side

    Enumerated type for representing the four sides of the a puzzle piece.

    Note:
        Pieces can only be rotated in 90 degree increments.

    """

    top = 0
    right = 1
    bottom = 2
    left = 3

    @staticmethod
    def get_numb_sides():
        """
        Accessor for the number of sizes for a puzzle piece.

        Returns (int):
            Since these are rectangular pieces, it returns size 4.  This is currently fixed.
        """
        return 4

    @staticmethod
    def get_all_sides():
        """
        Static method to extract all the sides of a piece.

        Returns ([PuzzlePieceSide]):
            List of all sides of a puzzle piece starting at the top and moving clockwise.
        """
        return [PuzzlePieceSide.top, PuzzlePieceSide.right, PuzzlePieceSide.bottom, PuzzlePieceSide.left]

    @property
    def complementary_side(self):
        """
        Determines and returns the complementary side of this implicit side parameter.  For example, if this side
        is "left" then the function returns "right" and vice versa.

        Returns (PuzzlePieceSide):
            Complementary side of the piece.
        """
        if self == PuzzlePieceSide.top:
            return PuzzlePieceSide.bottom

        if self == PuzzlePieceSide.right:
            return PuzzlePieceSide.left

        if self == PuzzlePieceSide.bottom:
            return PuzzlePieceSide.top

        if self == PuzzlePieceSide.left:
            return PuzzlePieceSide.right

    @property
    def side_name(self):
        """
        Gets the name of a puzzle piece side without the class name

        Returns (str):
            The name of the side as a string
        """
        return str(self).split(".")[1]


class SolidColor(Enum):
    """
    Solid color in Blue, Green, Red (BGR) format.
    """
    black = (0, 0, 0)
    white = (255, 255, 255)


class PuzzlePiece(object):
    """
    Puzzle Piece Object.  It is a very simple object that stores the puzzle piece's pixel information in a
    NumPY array.  It also stores the piece's original information (e.g. X/Y location and puzzle ID) along with
    what was determined by the solver.
    """

    # Represents L-A-B dimensions in the LAB color space
    NUMB_LAB_COLORSPACE_DIMENSIONS = 3

    _PERFORM_ASSERT_CHECKS = True

    # Use predicted values for edge borders for speed up
    _USE_STORED_PREDICTED_VALUE_SPEED_UP = True

    # When drawing the image results, optionally draw a border around the pieces to
    # make piece differences more evident.
    _ADD_RESULTS_IMAGE_BORDER = True
    _WHITE_BORDER_THICKNESS = 2  # pixels

    def __init__(self, puzzle_id, location, lab_img, piece_id=None, puzzle_grid_size=None):
        """
        Puzzle Piece Constructor.

        Args:
            puzzle_id (int): Puzzle identification number
            location ([int]): (row, column) location of this piece.
            lab_img: Image data in the form of a NumPy array.
            piece_id (int): Piece identification number.
            puzzle_grid_size ([int]): Grid size of the puzzle
        """

        # Verify the piece id information
        if piece_id is None and puzzle_grid_size is not None:
            raise ValueError("Using the puzzle grid size is not supported if piece id is \"None\".")

        # Piece ID is left to the solver to set
        self._orig_piece_id = piece_id
        self._assigned_piece_id = None

        self._orig_puzzle_id = puzzle_id
        self._assigned_puzzle_id = None

        # Store the original location of the puzzle piece and initialize a placeholder x/y location.
        self._orig_loc = location
        self._assigned_loc = None

        # Store the segment information
        self._segment_id_numb = None
        self._segment_color = None
        self._is_stitching_piece = None

        # Optionally calculate the identification numbers of the piece neighbors
        self._actual_neighbor_ids = None
        if puzzle_grid_size is not None:
            self.calculate_actual_neighbor_id_numbers(puzzle_grid_size)

        # Store the image data
        self._img = lab_img
        (length, width, dim) = self._img.shape
        if width != length:
            raise ValueError("Only square puzzle pieces are supported at this time.")
        if dim != PuzzlePiece.NUMB_LAB_COLORSPACE_DIMENSIONS:
            raise ValueError("This image does not appear to be in the LAB colorspace as it does not have 3 dimensions")
        self._width = width
        # For some debug images, we may want to see a solid image instead of the original image.
        # This property stores that color.
        self._results_image_coloring = None

        # Used to speed up piece to piece calculations
        self._border_average_color = None
        self._predicted_border_values = [None] * PuzzlePieceSide.get_numb_sides()
        self._calculate_border_color_average()

        # Rotation gets set later.
        self._rotation = None

    def calculate_actual_neighbor_id_numbers(self, puzzle_grid_size):
        """
        Neighbor ID Calculator

        Given a grid size, this function calculates the identification number of this piece's neighbors.  If a piece
        has no neighbor, then location associated with that puzzle piece is filled with "None".

        Args:
            puzzle_grid_size ([int]): Grid size (number of rows, number of columns) for this piece's puzzle.
        """

        # Only need to calculate the actual neighbor id information once
        if self._actual_neighbor_ids is not None:
            return
        # Initialize actual neighbor id information
        self._actual_neighbor_ids = []

        # Extract the information on the puzzle grid size
        (numb_rows, numb_cols) = puzzle_grid_size

        # Check the top location first
        # If the row is 0, then it has no top neighbor
        if self._orig_loc[0] == 0:
            neighbor_id = None
        else:
            neighbor_id = self._orig_piece_id - numb_cols
        self._actual_neighbor_ids.append((neighbor_id, PuzzlePieceSide.top))

        # Check the right side
        # If in the last column, it has no right neighbor
        if self._orig_loc[1] + 1 == numb_cols:
            neighbor_id = None
        else:
            neighbor_id = self._orig_piece_id + 1
        self._actual_neighbor_ids.append((neighbor_id, PuzzlePieceSide.right))

        # Check the bottom side
        # If in the last column, it has no right neighbor
        if self._orig_loc[0] + 1 == numb_rows:
            neighbor_id = None
        else:
            neighbor_id = self._orig_piece_id + numb_cols
        self._actual_neighbor_ids.append((neighbor_id, PuzzlePieceSide.bottom))

        # Check the right side
        # If in the last column, it has no left neighbor
        if self._orig_loc[1] == 0:
            neighbor_id = None
        else:
            neighbor_id = self._orig_piece_id - 1
        self._actual_neighbor_ids.append((neighbor_id, PuzzlePieceSide.left))

        # Convert the list to a tuple since it is immutable
        self._actual_neighbor_ids = tuple(self._actual_neighbor_ids)

    def _calculate_border_color_average(self):
        """
        Calculate the average color for each border to expedite calculations of puzzle piece side
        """
        # Top side border sum
        # noinspection PyListCreation
        border_color = [np.sum(self.get_row_pixels(0))]
        # Right side
        border_color.append(np.sum(self.get_column_pixels(self._width - 1)))
        # Bottom side
        border_color.append(np.sum(self.get_row_pixels(self._width - 1)))
        # Left side
        border_color.append(np.sum(self.get_column_pixels(0)))

        # convert to average
        for i in xrange(0, len(border_color)):
            border_color[i] = 1.0 * border_color[i] / (self.width * PuzzlePiece.NUMB_LAB_COLORSPACE_DIMENSIONS)

        # Convert to a tuple
        self._border_average_color = tuple(border_color)

    def border_average_color(self, side):
        """
        Border Average Color Accessor

        Gets the average color for a border piece.

        Args:
            side (PuzzlePieceSide): Side of the puzzle piece whose average value will be returned.

        Returns (float):
            The average pixel value for the puzzle piece border.
        """
        return self._border_average_color[side.value]

    @property
    def original_neighbor_id_numbers_and_sides(self):
        """
        Neighbor Identification Number Property

        In a puzzle, each piece has up to four neighbors.  This function access that identification number information.

        Returns (List[int, PuzzlePieceSide]):
            Identification number for the puzzle piece on the specified side of the original object.

        """
        # Verify that the array containing the neighbor id numbers is not none
        if PuzzlePiece._PERFORM_ASSERT_CHECKS:
            assert self._actual_neighbor_ids is not None

        # Return the piece's neighbor identification numbers
        return self._actual_neighbor_ids

    @property
    def width(self):
        """
        Gets the size of the square puzzle piece.  Since it is square, width its width equals its length.

        Returns (int):
            Width of the puzzle piece in pixels.

        """
        return self._width

    @property
    def original_puzzle_location(self):
        """
        Accessor for the Original Puzzle Location

        Returns (PuzzleLocation): Original puzzle location using the PuzzleLocation class.
        """
        return PuzzleLocation(self.original_puzzle_id, self._orig_loc[0], self._orig_loc[1])

    @property
    def puzzle_location(self):
        """
        Property to access the Solved Puzzle Location

        Returns (PuzzleLocation): Puzzle location using the PuzzleLocation class.  This is the location in the solved
           puzzle.
        """
        return PuzzleLocation(self.puzzle_id, self._assigned_loc[0], self._assigned_loc[1])

    @property
    def location(self):
        """
        Gets the location of the puzzle piece on the board.

        Returns ([int]):
            Tuple of the (row, column)

        """
        return self._assigned_loc

    @location.setter
    def location(self, new_loc):
        """
        Updates the puzzle piece location.

        Args:
            new_loc (List[int]): New puzzle piece location.

        """
        if len(new_loc) != 2:
            raise ValueError("Location of a puzzle piece must be a two dimensional tuple")
        self._assigned_loc = new_loc

    @property
    def original_puzzle_id(self):
        """
        Accessor for the puzzle piece's original puzzle identification number.

        Returns (int): Identification number of the puzzle when it was created.
        """
        return self._orig_puzzle_id

    @property
    def puzzle_id(self):
        """
        Assigned Puzzle Identification Number

        Gets the location of the puzzle piece on the board.

        Returns (int):
            Assigned Puzzle ID number.

        """
        return self._assigned_puzzle_id

    @puzzle_id.setter
    def puzzle_id(self, new_puzzle_id):
        """
        Updates the puzzle ID number for the puzzle piece.

        Returns (int):
            Board identification number

        """
        self._assigned_puzzle_id = new_puzzle_id

    @property
    def original_piece_id(self):
        """
        Original Piece ID Number

        Gets the original (i.e., correct) piece identification number

        Returns (int):
            Original identification number assigned to the piece at its creation.  Should be globally unique.
        """
        return self._orig_piece_id

    @property
    def id_number(self):
        """
        Puzzle Piece ID Getter

        Gets the identification number for a puzzle piece.

        Returns (int):
            Puzzle piece identification number
        """
        # Check whether the assigned piece ID is not none
        if PuzzlePiece._PERFORM_ASSERT_CHECKS:
            assert self._assigned_piece_id is not None
        # Return the piece id number
        return self._assigned_piece_id

    @id_number.setter
    def id_number(self, new_piece_id):
        """
        Piece ID Setter

        Sets the puzzle piece's identification number.

        Args:
            new_piece_id (int): Puzzle piece identification number
        """
        self._assigned_piece_id = new_piece_id

    @property
    def segment_number(self):
        """
        Puzzle Piece Segment Number Accessor

        Gets the identification number of the segment this puzzle is assigned to.

        Returns (int):
            Identification number of the segment the puzzle piece is assigned to.
        """
        # Check whether the assigned piece ID is not none
        if PuzzlePiece._PERFORM_ASSERT_CHECKS:
            assert self._segment_id_numb is not None
        # Return the piece id number
        return self._segment_id_numb

    @segment_number.setter
    def segment_number(self, new_segment_id):
        """
        Puzzle Piece Segment Number Setter

        Sets the segment number the puzzle piece is assigned to.

        Args:
            new_segment_id (int): New segment number the puzzle piece is assigned to
        """
        self._segment_id_numb = new_segment_id

        # Reset the segment color
        self._segment_color = None

    def has_segment_color(self):
        """
        Checks whether the puzzle piece has a segment color.

        Returns (bool): True if the piece has a segment color and False otherwise.
        """
        return self._segment_color is not None

    @property
    def segment_color(self):
        """
        Puzzle Piece Segment Color Accessor

        Each segment is assigned a color.  This is stored as part of the individual puzzle pieces.  This property
        access the color assigned to each puzzle piece.

        Returns (PuzzleSegmentColor): Segment color for this puzzle piece.
        """
        # Check whether the assigned piece ID is not none
        if PuzzlePiece._PERFORM_ASSERT_CHECKS:
            assert self._segment_color is not None

        # Return the piece id number
        return self._segment_color

    @segment_color.setter
    def segment_color(self, new_segment_color):
        """
        Puzzle Piece Segment Color Setter

        Sets the color for the segment this piece belongs to.

        Args:
            new_segment_color (PuzzleSegmentColor): New segment color assigned to this piece.
        """

        # For a given segment assignment, the puzzle piece should only have a color set once.
        if PuzzlePiece._PERFORM_ASSERT_CHECKS:
            assert self._segment_id_numb is not None

        self._segment_color = new_segment_color

    @property
    def is_stitching_piece(self):
        """
        Gets whether the piece is a stitching piece

        Returns (bool): True if the piece is a stitching piece and False otherwise.
        """
        if PuzzlePiece._PERFORM_ASSERT_CHECKS:
            assert self._is_stitching_piece is not None
        return self._is_stitching_piece

    @is_stitching_piece.setter
    def is_stitching_piece(self, value):
        """
        Sets whether the piece is a stitching piece

        Args:
            value (bool): True if the piece is a stitching piece and False otherwise.
        """
        if not isinstance(value, bool):
            raise ValueError("value must be of type bool.")
        self._is_stitching_piece = value

    @property
    def lab_image(self):
        """
        Get's a puzzle piece's image in the LAB colorspace.

        Returns (Numpy[int]):
            Numpy array of the piece's lab image.
        """
        return self._img

    @property
    def rotation(self):
        """
        Rotation Accessor

        Gets the puzzle piece's rotation.

        Returns (PuzzlePieceRotation):
            The puzzle piece's rotation
        """
        return self._rotation

    @rotation.setter
    def rotation(self, new_rotation):
        """
        Puzzle Piece Rotation Setter

        Updates a puzzle piece's rotation.

        Args:
            new_rotation (PuzzlePieceRotation): New rotation for the puzzle piece.
        """
        self._rotation = new_rotation

    def side_adjacent_to_location(self, location):
        """
        Given an adjacent puzzle location, this function returns the side that is touching that adjacent location.

        Args:
            location : A puzzle piece location adjacent to this piece.  This can either be a Tuple[Int]
              or PuzzleLocation.

        Returns (PuzzlePieceSide):
            Side of this piece that is touching the adjacent location

        """
        # Support multiple types either PuzzleLocation or Tuple
        if type(location) is PuzzleLocation:
            loc_and_side = self.get_neighbor_puzzle_location_and_sides()
        else:
            loc_and_side = self.get_neighbor_locations_and_sides()

        # Iterate through the possibilities and return if the location matches
        for (loc, side) in loc_and_side:
            if loc == location:
                return side
        # If you reached here, something went wrong
        err = "Specified Location: \"(%s,%s)\" is not adjacent this piece's location \"(%s, %s)\"" % (location[0],
                                                                                                      location[1],
                                                                                                      self.location[0],
                                                                                                      self.location[1])
        raise ValueError(err)

    @property
    def results_image_coloring(self):
        """
        Gets the results color image for the piece.

        Returns(List):
            Either a single BGR integer list when a solid color is used.  If it is using polygon print, then the
            return is a List[(List[int], PuzzlePieceSide)].

        """
        return self._results_image_coloring

    @results_image_coloring.setter
    def results_image_coloring(self, color):
        """
        Sets the image coloring when only a single color is needed.

        Args:
            color (List[int]): Color of the image in BGR format

        """
        self._results_image_coloring = color

    def reset_image_coloring_for_polygons(self):
        """
        Sets up the results image coloring for
        """
        self._results_image_coloring = []

    def results_image_polygon_coloring(self, side, color):
        """
        Sets the image coloring when only a single color is needed.

        Args:
            side (PuzzlePieceSide): Side of the piece that will be assigned a color.
            color (List[int]): Color of the image in BGR format
        """
        self._results_image_coloring.append((side, color.value))

    def get_neighbor_puzzle_location_and_sides(self):
        """
        Neighbor Puzzle Location and Sides

        This function replaces the previously used "get_neighbor_locations_and_sides".  Rather than returning the
        location as a list, it now uses the PuzzleLocation class.

        Given a puzzle piece, this function returns the four surrounding coordinates/location and the sides of THIS
        puzzle piece that corresponds to those locations so that it can be added to the open slot list.

        Returns (List[Tuple(PuzzleLocation, PuzzlePieceSide)]: A list of puzzle piece locations and their
           accompanying side.
        """

        location_side_tuple = self.get_neighbor_locations_and_sides()

        # Build the structure to use puzzle locations
        output_loc_and_side = []
        for (temp_loc, side) in location_side_tuple:
            output_loc_and_side.append((PuzzleLocation(self._assigned_puzzle_id, temp_loc[0], temp_loc[1]), side))
        return output_loc_and_side

    def get_neighbor_locations_and_sides(self):
        """
        Neighbor Locations and Sides

        Given a puzzle piece, this function returns the four surrounding coordinates/location and the sides of THIS
        puzzle piece that corresponds to those locations so that it can be added to the open slot list.

        DEPRECATED.

        Returns ([([int], PuzzlePieceSide)]):
            Valid puzzle piece locations and the respective puzzle piece side.
        """

        if PuzzlePiece._PERFORM_ASSERT_CHECKS:
            assert self.location is not None
            assert self.rotation is not None

        # TODO this approach does not account for missing pieces.
        return PuzzlePiece._get_neighbor_locations_and_sides(self.location, self.rotation)

    @staticmethod
    def _get_neighbor_locations_and_sides(piece_loc, piece_rotation):
        """
        Neighbor Locations and Sides

        Static method that given a piece location and rotation, it returns the four surrounding coordinates/location
        and the puzzle piece side that aligns with it so that it can be added to the open slot list.

        Args:
            piece_loc ([int]):
            piece_rotation (PuzzlePieceRotation):

        Returns ([([int], PuzzlePieceSide)]):
            Valid puzzle piece locations and the respective puzzle piece side.
        """
        # Get the top location and respective side
        top_loc = (piece_loc[0] - 1, piece_loc[1])
        # noinspection PyTypeChecker
        location_piece_side_tuples = [(top_loc, PuzzlePiece._determine_unrotated_side(piece_rotation,
                                                                                      PuzzlePieceSide.top))]
        # Get the right location and respective side
        right_loc = (piece_loc[0], piece_loc[1] + 1)
        # noinspection PyTypeChecker
        location_piece_side_tuples.append((right_loc, PuzzlePiece._determine_unrotated_side(piece_rotation,
                                                                                            PuzzlePieceSide.right)))
        # Get the bottom location and its respective side
        bottom_loc = (piece_loc[0] + 1, piece_loc[1])
        # noinspection PyTypeChecker
        location_piece_side_tuples.append((bottom_loc, PuzzlePiece._determine_unrotated_side(piece_rotation,
                                                                                             PuzzlePieceSide.bottom)))
        # Get the right location and respective side
        left_loc = (piece_loc[0], piece_loc[1] - 1)
        # noinspection PyTypeChecker
        location_piece_side_tuples.append((left_loc, PuzzlePiece._determine_unrotated_side(piece_rotation,
                                                                                           PuzzlePieceSide.left)))
        # Return the location/piece side tuples
        return location_piece_side_tuples

    def bgr_image(self):
        """
        Get's a puzzle piece's image in the BGR colorspace.

        Returns (Numpy[int]):
            Numpy array of the piece's BGR image.
        """
        return cv2.cvtColor(self._img, cv2.COLOR_LAB2BGR)

    def get_row_pixels(self, row_numb, reverse=False):
        """
        Extracts a row of pixels from a puzzle piece.

        Args:
            row_numb (int): Pixel row in the image.  Must be between 0 and the width of the piece - 1 (inclusive).
            reverse (Optional bool): Select whether to reverse the pixel information.

        Returns (Numpy[int]):
            A vector of 3-dimensional pixel values for a row in the image.
        """
        if row_numb < 0:
            raise ValueError("Row number for a piece must be greater than or equal to zero.")
        if row_numb >= self._width:
            raise ValueError("Row number for a piece must be less than the puzzle's pieces width")

        if reverse:
            return self._img[row_numb, ::-1, :]
        else:
            return self._img[row_numb, :, :]

    def get_column_pixels(self, col_numb, reverse=False):
        """
        Extracts a row of pixels from a puzzle piece.

        Args:
            col_numb (int): Pixel column in the image.  Must be between 0 and the width of the piece - 1 (inclusive).
            reverse (Optional bool): Select whether to reverse the pixel information.

        Returns (Numpy[int]):
            A vector of 3-dimensional pixel values for a column in the image.
        """
        if col_numb < 0:
            raise ValueError("Column number for a piece must be greater than or equal to zero.")
        if col_numb >= self._width:
            raise ValueError("Column number for a piece must be less than the puzzle's pieces width")
        # If you reverse, change the order of the pixels.
        if reverse:
            return self._img[::-1, col_numb, :]
        else:
            return self._img[:, col_numb, :]

    def _assign_to_original_location(self):
        """
        Loopback Location Assigner

        Test Method Only.  Correctly assigns a piece to its original location.
        """
        self._assigned_loc = self._orig_loc

    def _set_id_number_to_original_id(self):
        """
        Loopback ID Number

        Test Method Only.  Sets the assigned and original piece id number to the same value.
        """
        self._assigned_piece_id = self._orig_piece_id

    @staticmethod
    def calculate_asymmetric_distance(piece_i, piece_i_side, piece_j, piece_j_side):
        """
        Uses the Asymmetric Distance function to calculate the distance between two puzzle pieces.

        Args:
            piece_i (PuzzlePiece):
            piece_i_side (PuzzlePieceSide):
            piece_j (PuzzlePiece):
            piece_j_side (PuzzlePieceSide): Side of piece j that is adjacent to piece i.

        Returns (double):
            Distance between the sides of two puzzle pieces.
        """

        # Get the border information for p_i if not pre-calculated
        i_border = None
        i_second_to_last = None
        if piece_i._predicted_border_values[piece_i_side.value] is None or not PuzzlePiece._USE_STORED_PREDICTED_VALUE_SPEED_UP:
            # Get the border and second to last ROW on the TOP side of piece i
            if piece_i_side == PuzzlePieceSide.top:
                i_border = piece_i.get_row_pixels(0)
                i_second_to_last = piece_i.get_row_pixels(1)

            # Get the border and second to last COLUMN on the RIGHT side of piece i
            elif piece_i_side == PuzzlePieceSide.right:
                i_border = piece_i.get_column_pixels(piece_i.width - 1)
                i_second_to_last = piece_i.get_column_pixels(piece_i.width - 2)

            # Get the border and second to last ROW on the BOTTOM side of piece i
            elif piece_i_side == PuzzlePieceSide.bottom:
                i_border = piece_i.get_row_pixels(piece_i.width - 1)
                i_second_to_last = piece_i.get_row_pixels(piece_i.width - 2)

            # Get the border and second to last COLUMN on the LEFT side of piece i
            elif piece_i_side == PuzzlePieceSide.left:
                i_border = piece_i.get_column_pixels(0)
                i_second_to_last = piece_i.get_column_pixels(1)
            else:
                raise ValueError("Invalid edge for piece i")

        # If rotation is allowed need to reverse pixel order in some cases.
        reverse = False  # By default do not reverse
        # Always need to reverse when they are the same side
        if piece_i_side == piece_j_side:
            reverse = True
        # Get the pixels along the TOP of piece_j
        if piece_j_side == PuzzlePieceSide.top:
            if piece_i_side == PuzzlePieceSide.right:
                reverse = True
            j_border = piece_j.get_row_pixels(0, reverse)

        # Get the pixels along the RIGHT of piece_j
        elif piece_j_side == PuzzlePieceSide.right:
            if piece_i_side == PuzzlePieceSide.top:
                reverse = True
            j_border = piece_j.get_column_pixels(piece_i.width - 1, reverse)

        # Get the pixels along the BOTTOM of piece_j
        elif piece_j_side == PuzzlePieceSide.bottom:
            if piece_i_side == PuzzlePieceSide.left:
                reverse = True
            j_border = piece_j.get_row_pixels(piece_i.width - 1, reverse)

        # Get the pixels along the RIGHT of piece_j
        elif piece_j_side == PuzzlePieceSide.left:
            if piece_i_side == PuzzlePieceSide.bottom:
                reverse = True
            j_border = piece_j.get_column_pixels(0, reverse)
        else:
            raise ValueError("Invalid edge for piece i")

        # If needed, recalculate the side value.
        if piece_i._predicted_border_values[piece_i_side.value] is None or not PuzzlePiece._USE_STORED_PREDICTED_VALUE_SPEED_UP:
            # Calculate the value of pixels on piece j's edge.
            piece_i._predicted_border_values[piece_i_side.value] = (2 * (i_border.astype(np.int32))
                                                                    - i_second_to_last.astype(np.int32))
        # Get the predicated stored value
        predicted_j = piece_i._predicted_border_values[piece_i_side.value]

        # noinspection PyUnresolvedReferences
        pixel_diff = predicted_j.astype(np.int32) - j_border.astype(np.int32)

        # Return the sum of the absolute values.
        pixel_diff = np.absolute(pixel_diff)
        return np.sum(pixel_diff, dtype=np.int32)

    def set_placed_piece_rotation(self, placed_side, neighbor_piece_side, neighbor_piece_rotation):
        """
        Placed Piece Rotation Setter

        Given an already placed neighbor piece's adjacent side and rotation, this function sets the rotation
        of some newly placed piece that is put adjacent to that neighbor piece.

        Args:
            placed_side (PuzzlePieceSide): Side of the placed puzzle piece that is adjacent to the neighbor piece

            neighbor_piece_side (PuzzlePieceSide): Side of the neighbor piece that is adjacent to the newly
            placed piece.

            neighbor_piece_rotation (PuzzlePieceRotation): Rotation of the already placed neighbor piece
        """
        # Calculate the placed piece's new rotation
        self.rotation = PuzzlePiece._calculate_placed_piece_rotation(placed_side, neighbor_piece_side,
                                                                     neighbor_piece_rotation)

    @staticmethod
    def _calculate_placed_piece_rotation(placed_piece_side, neighbor_piece_side, neighbor_piece_rotation):
        """
        Placed Piece Rotation Calculator

        Given an already placed neighbor piece, this function determines the correct rotation for a newly placed
        piece.

        Args:
            placed_piece_side (PuzzlePieceSide): Side of the placed puzzle piece adjacent to the existing piece
            neighbor_piece_side (PuzzlePieceSide): Side of the neighbor of the placed piece that is touching
            neighbor_piece_rotation (PuzzlePieceRotation): Rotation of the neighbor piece

        Returns (PuzzlePieceRotation):
            Rotation of the placed puzzle piece given the rotation and side of the neighbor piece.
        """
        # Get the neighbor piece rotation
        unrotated_complement = neighbor_piece_side.complementary_side

        placed_rotation_val = int(neighbor_piece_rotation.value)
        # noinspection PyUnresolvedReferences
        placed_rotation_val += 90 * (PuzzlePieceRotation.degree_360.value + (unrotated_complement.value
                                                                             - placed_piece_side.value))
        # Calculate the normalized rotation
        # noinspection PyUnresolvedReferences
        placed_rotation_val %= PuzzlePieceRotation.degree_360.value
        # Check if a valid rotation value.
        if PuzzlePiece._PERFORM_ASSERT_CHECKS:
            assert placed_rotation_val % 90 == 0
        # noinspection PyUnresolvedReferences
        return PuzzlePieceRotation(placed_rotation_val % PuzzlePieceRotation.degree_360.value)

    @staticmethod
    def _determine_unrotated_side(piece_rotation, rotated_side):
        """
        Unrotated Side Determiner

        Given a piece's rotation and the side of the piece (from the reference of the puzzle), find its actual
        (i.e. unrotated) side.

        Args:
            piece_rotation (PuzzlePieceRotation): Specified rotation for a puzzle piece.
            rotated_side (PuzzlePieceSide): From a Puzzle perspective, this is the exposed side

        Returns(PuzzlePieceSide):
            Actual side of the puzzle piece
        """
        rotated_side_val = rotated_side.value
        # Get the number of 90 degree rotations
        numb_90_degree_rotations = int(piece_rotation.value / 90)

        # Get the unrotated side
        unrotated_side = (rotated_side_val + (PuzzlePieceSide.get_numb_sides() - numb_90_degree_rotations))
        unrotated_side %= PuzzlePieceSide.get_numb_sides()

        # Return the actual side
        return PuzzlePieceSide(unrotated_side)

    @staticmethod
    def _get_neighbor_piece_rotated_side(placed_piece_loc, neighbor_piece_loc):
        """

        Args:
            placed_piece_loc ([int]): Location of the newly placed piece
            neighbor_piece_loc ([int): Location of the neighbor of the newly placed piece

        Returns (PuzzlePieceSide): Side of the newly placed piece where the placed piece is now location.

        ::Note:: This does not take into account any rotation of the neighbor piece.  That is why this function is
        referred has "rotated side" in its name.
        """
        # Calculate the row and column distances
        row_dist = placed_piece_loc[0] - neighbor_piece_loc[0]
        col_dist = placed_piece_loc[1] - neighbor_piece_loc[1]

        # Perform some checking on the pieces
        if PuzzlePiece._PERFORM_ASSERT_CHECKS:
            # Verify the pieces are in the same puzzle
            assert abs(row_dist) + abs(col_dist) == 1

        # Determine the relative side of the placed piece
        if row_dist == -1:
            return PuzzlePieceSide.top
        elif row_dist == 1:
            return PuzzlePieceSide.bottom
        elif col_dist == -1:
            return PuzzlePieceSide.left
        else:
            return PuzzlePieceSide.right

    @staticmethod
    def create_solid_image(bgr_color, width, height=None):
        """
        Create a solid image for displaying in output images.

        Args:
            bgr_color (Tuple[int]): Color  in BLUE, GREEN, RED notation.  Each element for blue, green, or red
              must be between 0 and 255 inclusive.
            width (int): Width of the image in pixels.
            height (Optional int): Height of the image in number of pixels.  If it is not specified, then the image
              is a square.

        Returns (Numpy[int]):
            Image in the form of a NumPy matrix of size: (length by width by 3)

        """
        # Handle the case when no height is specified.
        if height is None:
            height = width
        # Create a black image
        image = np.zeros((height, width, PuzzlePiece.NUMB_LAB_COLORSPACE_DIMENSIONS), np.uint8)
        # Fill with the bgr color
        if isinstance(bgr_color, Enum):
            image[:] = bgr_color.value
        else:
            image[:] = bgr_color
        # Optionally add a border around the pieces before returning
        if PuzzlePiece._ADD_RESULTS_IMAGE_BORDER:
            return PuzzlePiece.add_results_image_border(image)
        else:
            return image

    @staticmethod
    def add_results_image_border(image):
        """
        Optionally add an image border around the piece image.  This is primarily intended for use
        with the solid results images.

        Args:
            image (Numpy[int]): Piece image with no border

        Returns (Numpy[int]):
            Piece image with a border around the solid image.
        """
        (height, width, _) = image.shape
        # noinspection PyUnresolvedReferences
        cv2.rectangle(image, (0, 0), (width, height), SolidColor.white.value,
                      thickness=PuzzlePiece._WHITE_BORDER_THICKNESS)
        return image

    def key(self):
        """
        Standardized method for using puzzle pieces as keys to a dictionary.

        Returns (string): Object key to be used for dictionaries
        """
        return str(self.id_number)

    @staticmethod
    def create_side_polygon_image(bgr_color_by_side, width, height=None):
        """
        Create a solid image for displaying in output images.

        Args:
            bgr_color_by_side (Tuple[(Tuple[int], PuzzlePieceSide)]): Color in BLUE, GREEN, RED notation for each
              specified puzzle piece side.  Draws four triangles based off the
            width (int): Width of the image in pixels.
            height (Optional int): Height of the image in number of pixels.  If it is not specified, then the image
              is a square.

        Returns (Numpy[int]):
            Image in the form of a NumPy matrix of size: (length by width by 3)

        """
        # Handle the case when no height is specified.
        if height is None:
            height = width

        # Verify each side is accounted for.
        if PuzzlePiece._PERFORM_ASSERT_CHECKS:
            assert len(bgr_color_by_side) == PuzzlePieceSide.get_numb_sides()

        # Define the center point of the image.
        center_point = [height / 2, width / 2]

        # Used for assertion checking.
        sides_drawn = []

        # Define the other four coordinates for the polygon.
        top_left = [0, 0]
        top_right = [width - 1, 0]
        bottom_left = [0, height - 1]
        bottom_right = [height - 1, width - 1]

        # Create a black image
        image = np.zeros((width, height, PuzzlePiece.NUMB_LAB_COLORSPACE_DIMENSIONS), np.uint8)
        # For each side, fill with a polygon.
        for (side, color) in bgr_color_by_side:

            # Build the points in the polygon vector
            if side == PuzzlePieceSide.top:
                vector_points = [top_left, top_right]
            elif side == PuzzlePieceSide.right:
                vector_points = [top_right, bottom_right]
            elif side == PuzzlePieceSide.bottom:
                vector_points = [bottom_left, bottom_right]
            else:
                vector_points = [top_left, bottom_left]
            vector_points.append(center_point)

            # Add drawn sides to the assertion checks
            if PuzzlePiece._PERFORM_ASSERT_CHECKS:
                # Ensure no side is drawn twice
                assert side not in sides_drawn
                # Add the side tot he list.
                sides_drawn.append(side)

            # Build a polygon
            polygon = np.array([vector_points], np.int32)
            cv2.fillConvexPoly(image, polygon, color)

        # Verify that all sides are drawn
        if PuzzlePiece._PERFORM_ASSERT_CHECKS:
            assert len(sides_drawn) == PuzzlePieceSide.get_numb_sides()

        # Draw an "X" to clearly demarcate the triangles
        # noinspection PyUnresolvedReferences
        cv2.line(image, tuple(top_left), tuple(bottom_right), SolidColor.black.value, thickness=1)
        # noinspection PyUnresolvedReferences
        cv2.line(image, tuple(top_right), tuple(bottom_left), SolidColor.black.value, thickness=1)

        # Optionally add a border around the pieces before returning
        if PuzzlePiece._ADD_RESULTS_IMAGE_BORDER:
            return PuzzlePiece.add_results_image_border(image)
        else:
            return image

    def is_correctly_placed(self, puzzle_offset_upper_left_location):
        """
        Piece Placement Checker

        Checks whether the puzzle piece is correctly placed.

        Args:
            puzzle_offset_upper_left_location (Tuple[int]): Modified location for the origin of the puzzle

        Returns (bool):
            True if the puzzle piece is in the correct location and False otherwise.
        """

        # Verify all dimensions
        for i in xrange(0, len(self._orig_loc)):
            # If for the current dimension
            if self._assigned_loc[i] - self._orig_loc[i] - puzzle_offset_upper_left_location[i] != 0:
                return False
        # Mark as correctly placed
        return True


def top_level_calculate_asymmetric_distance(piece_i, piece_i_side, piece_j, piece_j_side):
    """
    NOTE: This function merely calls the static method: <b>PuzzlePiece.calculate_asymmetric_distance</b>.  This
    wrapper function is used because Python module "multiprocessing" requires that functions be pickle-able
    which means they need to be visible at the top level and <b>not</b> static methods.

    Uses the Asymmetric Distance function to calculate the distance between two puzzle pieces.

    Args:
        piece_i (PuzzlePiece):
        piece_i_side (PuzzlePieceSide):
        piece_j (PuzzlePiece):
        piece_j_side (PuzzlePieceSide): Side of piece j that is adjacent to piece i.

    Returns (double):
        Distance between the sides of two puzzle pieces.
    """
    return PuzzlePiece.calculate_asymmetric_distance(piece_i, piece_i_side, piece_j, piece_j_side)
