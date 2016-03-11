from enum import Enum
import numpy


class PuzzlePieceRotation(Enum):
    """Puzzle Piece PieceRotation

    Enumerated type for representing the amount of rotation for a puzzle piece.

    Note:
        Pieces can only be rotated in 90 degree increments.

    """

    degree_0 = 0        # No rotation
    degree_90 = 90      # 90 degree rotation
    degree_180 = 180    # 180 degree rotation
    degree_270 = 270    # 270 degree rotation

    @staticmethod
    def _calculate_xy_rotation(width, rotation, x, y):
        """
        To simplify the implementation, the puzzle data is never rotated.  Rather, for a given set of X/Y coordinates
        this function rotates them and then returns the rotated coordinates.

        Args:
            width (int):    Width of a puzzle piece in number of pixels.
            rotation(PuzzlePieceRotation): Specified rotation of the puzzle piece.
            x (int):        Unrotated X coordinate
            y (int):        Unrotated Y coordinate

        Returns ((int,int)):
            Rotated X/Y coordinates.

        """
        if rotation == PuzzlePieceRotation.degree_0:
            return x, y
        if rotation == PuzzlePieceRotation.degree_90:
            return width - 1 - y, x
        if rotation == PuzzlePieceRotation.degree_180:
            return width - 1 - x, width - 1 - y
        if rotation == PuzzlePieceRotation.degree_270:
            return y, width - 1 - x


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

    def __init__(self, puzzle_id, location, img):
        """
        Puzzle Piece Constructor.

        Args:
            puzzle_id (int): Puzzle identification number
            location ([int]): XY location of this piece.
            img: Image data in the form of a numpy array.

        """

        self._orig_puzzle_id = puzzle_id
        self._assigned_puzzle_id = None

        # Store the original location of the puzzle piece and initialize a placeholder x/y location.
        self._orig_loc = location
        self._assigned_loc = None

        # Store the image data
        self._img = img
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

        Returns ([int]): Tuple of the (x_location, y_location)

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

        Returns ([int]): Tuple of the (x_location, y_location)

        """
        return self._assigned_puzzle_id

    @puzzle_id.setter
    def puzzle_id(self, new_puzzle_id):
        """
        Updates the puzzle ID number for the puzzle piece.

        Returns (int): Board identification number

        """
        self._assigned_puzzle_id = new_puzzle_id

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
            return self._img[::-1, row_numb, :]
        else:
            return self._img[:, row_numb, :]

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
            return self._img[col_numb, ::-1, :]
        else:
            return self._img[col_numb, :, :]

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
            i_second_to_last = 2*i_border - piece_i.get_row_pixels(1)

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
        reverse = False # By default do not reverse
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
        predicted_j = 2*i_border - i_second_to_last
        pixel_diff = predicted_j - j_border

        # Return the sum of the absolute values.
        pixel_diff = numpy.absolute(pixel_diff).sum
        return numpy.sum(pixel_diff)

