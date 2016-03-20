"""
Created by Zayd Hammoudeh (zayd.hammoudeh@sjsu.edu)
"""
import random

from enum import Enum
import numpy
import cv2  # Open CV


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


class PuzzlePiece(object):
    """
    Puzzle Piece Object.  It is a very simple object that stores the puzzle piece's pixel information in a
    NumPY array.  It also stores the piece's original information (e.g. X/Y location and puzzle ID) along with
    what was determined by the solver.
    """

    NUMB_LAB_COLORSPACE_DIMENSIONS = 3

    _PERFORM_ASSERTION_CHECKS = True

    def __init__(self, puzzle_id, location, lab_img):
        """
        Puzzle Piece Constructor.

        Args:
            puzzle_id (int): Puzzle identification number
            location ([int]): (row, column) location of this piece.
            lab_img: Image data in the form of a numpy array.

        """

        # Piece ID is left to the solver to set
        self._piece_id = None

        self._orig_puzzle_id = puzzle_id
        self._assigned_puzzle_id = None

        # Store the original location of the puzzle piece and initialize a placeholder x/y location.
        self._orig_loc = location
        self._assigned_loc = None

        # Store the image data
        self._img = lab_img
        (length, width, dim) = self._img.shape
        if width != length:
            raise ValueError("Only square puzzle pieces are supported at this time.")
        if dim != PuzzlePiece.NUMB_LAB_COLORSPACE_DIMENSIONS:
            raise ValueError("This image does not appear to be in the LAB colorspace as it does not have 3 dimensions")
        self._width = width

        # Rotation gets set later.
        self._rotation = None

    @property
    def width(self):
        """
        Gets the size of the square puzzle piece.  Since it is square, width its width equals its length.

        Returns (int): Width of the puzzle piece in pixels.

        """
        return self._width

    @property
    def location(self):
        """
        Gets the location of the puzzle piece on the board.

        Returns ([int]): Tuple of the (row, column)

        """
        return self._assigned_loc

    @location.setter
    def location(self, new_loc):
        """
        Updates the puzzle piece location.

        Args:
            new_loc ([int]): New puzzle piece location.

        """
        if len(new_loc) != 2:
            raise ValueError("Location of a puzzle piece must be a two dimensional tuple")
        self._assigned_loc = new_loc

    @property
    def puzzle_id(self):
        """
        Gets the location of the puzzle piece on the board.

        Returns (int): Assigned Puzzle ID number.

        """
        return self._assigned_puzzle_id

    @puzzle_id.setter
    def puzzle_id(self, new_puzzle_id):
        """
        Updates the puzzle ID number for the puzzle piece.

        Returns (int): Board identification number

        """
        self._assigned_puzzle_id = new_puzzle_id

    @property
    def id_number(self):
        """
        Puzzle Piece ID Getter

        Gets the identification number for a puzzle piece.

        Returns (int): Puzzle piece indentification number
        """
        return self._piece_id

    @id_number.setter
    def id_number(self, new_piece_id):
        """
        Piece ID Setter

        Sets the puzzle piece's identification number.

        Args:
            new_piece_id (int): Puzzle piece identification number
        """
        self._piece_id = new_piece_id

    @property
    def lab_image(self):
        """
        Get's a puzzle piece's image in the LAB colorspace.

        Returns:
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

    def get_neighbor_locations_and_sides(self):

        if PuzzlePiece._PERFORM_ASSERTION_CHECKS:
            assert self.location is not None
            assert self.rotation is not None

    def bgr_image(self):
        """
        Get's a puzzle piece's image in the BGR colorspace.

        Returns:
        Numpy array of the piece's BGR image.
        """
        return cv2.cvtColor(self._img, cv2.COLOR_LAB2BGR)

    def get_row_pixels(self, row_numb, reverse=False):
        """
        Extracts a row of pixels from a puzzle piece.

        Args:
            row_numb (int): Pixel row in the image.  Must be between 0 and the width of the piece - 1 (inclusive).
            reverse (Optional bool): Select whether to reverse the pixel information.

        Returns:

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

        Returns:

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
        """Loopback Assigner

        Test Method Only.  Correctly assigns a piece to its original location.
        """
        self._assigned_loc = self._orig_loc

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
            Distance between
        """

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

        # Calculate the value of pixels on piece j's edge.
        predicted_j = 2 * (i_border.astype(numpy.int16)) - i_second_to_last.astype(numpy.int16)
        # noinspection PyUnresolvedReferences
        pixel_diff = predicted_j.astype(numpy.int16) - j_border.astype(numpy.int16)

        # Return the sum of the absolute values.
        pixel_diff = numpy.absolute(pixel_diff)
        return numpy.sum(pixel_diff, dtype=numpy.int32)

    def set_placed_piece_rotation(self, placed_side, neighbor_piece):
        """

        Args:
            placed_side (PuzzlePieceSide): Side of the placed puzzle piece.
            neighbor_piece (PuzzlePiece): Neighbor Puzzle Piece
        """
        # Perform some checking on the pieces
        if PuzzlePiece._PERFORM_ASSERTION_CHECKS:
            # Verify the pieces are in the same puzzle
            assert self._assigned_puzzle_id == neighbor_piece.puzzle_id
            # Verify the neighbor piece has a rotation setting
            assert neighbor_piece.rotation is not None

        # Calculate the placed piece's new rotation
        self.rotation = PuzzlePiece._calculate_placed_piece_rotation(self.location, placed_side,
                                                                     neighbor_piece.location, neighbor_piece.rotation)

    @staticmethod
    def _calculate_placed_piece_rotation(placed_piece_location, placed_piece_side,
                                         neighbor_piece_location, neighbor_piece_rotation):

        # Get the neighbor piece rotation
        neighbor_rotated_side = PuzzlePiece.get_neighbor_piece_rotated_side(placed_piece_location,
                                                                            neighbor_piece_location)
        neighbor_unrotated_side = PuzzlePiece._determine_unrotated_side(neighbor_piece_rotation, neighbor_rotated_side)
        unrotated_complement = neighbor_unrotated_side.complementary_side

        placed_rotation_val = int(neighbor_piece_rotation.value)
        placed_rotation_val += 90 * (PuzzlePieceRotation.degree_360.value + (unrotated_complement.value
                                                                             - placed_piece_side.value))
        # Calculate the normalized rotation
        placed_rotation_val %= PuzzlePieceRotation.degree_360.value
        # Check if a valid rotation value.
        if PuzzlePiece._PERFORM_ASSERTION_CHECKS:
            assert placed_rotation_val % 90 == 0
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

        Returns(PuzzlePieceSide): Actual side of the puzzle piece
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
    def get_neighbor_piece_rotated_side(placed_piece_loc, neighbor_piece_loc):
        """

        Args:
            placed_piece_loc ([int]): Location of the newly placed piece
            neighbor_piece_loc ([int): Location of the neighbor of the newly placed piece

        Returns (PuzzlePieceSide): Side of the newly placed piece where the placed piece is now location.

        Notes: This does not take into account any rotation of the neighbor piece.  That is why this function is
        referred has "rotated side" in its name.
        """
        # Calculate the row and column distances
        row_dist = placed_piece_loc[0] - neighbor_piece_loc[0]
        col_dist = placed_piece_loc[1] - neighbor_piece_loc[1]

        # Perform some checking on the pieces
        if PuzzlePiece._PERFORM_ASSERTION_CHECKS:
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



