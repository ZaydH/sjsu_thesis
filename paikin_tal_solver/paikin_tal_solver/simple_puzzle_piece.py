import copy

from enum import Enum


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

class SimplePuzzlePiece(object):

    # When running debug tests, extra more time consuming error checking is done.
    RUN_DEBUG_TESTS = True

    # Number of dimensions in the LAB colorspace.
    NUMB_LAB_DIMENSIONS = 3

    # Define the minimum and maximum values for a LAB pixel.  Helps double check
    # and catch errors.
    PIXEL_LAB_MINIMUM_VALUE = 0
    PIXEL_LAB_MAXIMUM_VALUE = 255

    def __init__(self, id_numb, lab_pixel_data):
        """
        Constructor for the SimplePuzzlePiece.

        Args:
            id_numb (int):          Puzzle Piece ID number
            lab_pixel_data ([int]:  3D Matrix containing the pixel data.  Matrix size is (width x width x 3) where 3 is
                                    the number of dimensions in the LAB color space.

        """

        # Store the information on the piece.
        self._id = id_numb
        self._width = len(lab_pixel_data)
        self._pixel_data = copy.deepcopy(lab_pixel_data)

        # In debug mode, check
        if SimplePuzzlePiece.RUN_DEBUG_TESTS:
            self._check_piece_dimensions()

    def _check_piece_dimensions(self):
        """
        Checks if the puzzle piece dimensions and LAB values are valid.

        Should only be called as part of debug based self test.

        """
        for x in range(0, self._width):
            # Check if the all the rows have the same length as the width.
            assert(len(self._pixel_data[x]) == self._width)
            for y in range(0, self._width):
                assert(len(self._pixel_data[x][y]) == SimplePuzzlePiece.NUMB_LAB_DIMENSIONS)
                for d in range(0, SimplePuzzlePiece.NUMB_LAB_DIMENSIONS):
                    # noinspection PyChainedComparisons
                    assert(SimplePuzzlePiece.PIXEL_LAB_MINIMUM_VALUE <= self._pixel_data[x][y][d] and
                           self._pixel_data[x][y][d] <= SimplePuzzlePiece.PIXEL_LAB_MAXIMUM_VALUE)

    def get_pixel(self, rotation, x, y):
        """
        Returns a pixel value for a desired pixel.  It will take the x/y coordinates and adjusts them automatically
        based off the rotation.

        Args:
            rotation (PuzzlePieceRotation): Rotation of the possible piece in 90degree increments.
            x (int):        Pixel x coordinate
            y (int):        Pixel y coordinate

        Returns ([int]):
        Pixel value in the LAB colorspace.

        """
        (x_rot, y_rot) = SimplePuzzlePiece._calculate_xy_rotation(self._width, rotation, x, y)
        return self._pixel_data[x_rot][y_rot]

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
