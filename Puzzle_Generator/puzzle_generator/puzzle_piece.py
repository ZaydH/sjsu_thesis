from PIL import Image
from enum import Enum
import random
from PIL import Image
from pickle import dumps, loads


class Rotation(Enum):
    """
    Enumerated type for representing the amount of rotation for
    a puzzle piece.
    **Note:** Pieces can only be rotated in 90 degree increments.
    """

    degree_0 = 0        # No rotation
    degree_90 = 90      # 90 degree rotation
    degree_180 = 180    # 180 degree rotation
    degree_270 = 270    # 270 degree rotation
    _degree_360 = 360   # Internal use only.  Same as 0 degree rotation.

    @staticmethod
    def get_all_rotations():

        """
        Gets a list of all valid values of Rotation
        :return: List of rotation values in increasing order.
        """

        return (Rotation.degree_0, Rotation.degree_90, Rotation.degree_180,
            Rotation.degree_270)


class PuzzlePiece:

    # Minimum width for a puzzle piece.
    MINIMUM_WIDTH = 10

    # May want to disable rotation so have a check for that.
    rotation_enabled = True

    def __init__(self, width, image=None, start_x=None, start_y=None):
        """
        Constructur of an empty puzzle piece object
        :param width: int  Width of the puzzle piece in pixels
        """
        # Number of pixels in the width of the piece
        if width < PuzzlePiece.MINIMUM_WIDTH:
            raise ValueError("Specified width is less than the minimum width of %d" % {PuzzlePiece.MINIMUM_WIDTH})
        self._width = width
        # Create the matrix for storing the pixel information
        # noinspection PyUnusedLocal
        self._pixels = Image.new("RGB", (width, width), "white")
        if image is None and (start_x is not None or start_y is not None):
            raise ValueError("Argument image cannot be null if argument start_x or start_y are specified")
        if image is not None and (start_x is None or start_y is None):
            raise ValueError("If argument image is specified, neither start_x or start_y can be None")
        # If a valid pixel array is passed, then copy the contents to the piece.
        if image is not None:
            # Extract a subimage
            box = (start_x, start_y, start_x + width, start_y + width)
            self._pixels = image.crop(box)
            assert(self._pixels.size == (width, width))

        # Set a random rotation
        self._rotation = None # Create a reference to prevent compiler warnings
        self.set_rotation(Rotation.degree_0)



    def _get_unrotated_coordinates(self, rotated_x, rotated_y):
        """
        For a given puzzle piece with its own rotation and width and already
        rotated x/y coordinates, this function calculates unrotated x/y coordinates.

        This function is needed since the puzzle piece's pixels are stored unrotated.
        Hence, we need to rotate rotated x/y coordinates into their unrotated
        equivalents.

        :param rotated_x: int An already rotated x coordinate
        :param rotated_y: int An already rotated y coordinate

        :return: Tuple in the format (unrotated_x, unrotated_y)

        """

        # Number of rotations is equal to the number of rotations required to make the
        # rotated x/y values full circle.  Hence, 0degree rotation takes no 90 degrees.
        # 90degree rotation takes 3 rotations, 180degrees takes 2, and 270 takes 1.
        # noinspection PyProtectedMember,PyUnresolvedReferences
        numb_90_degree_rotations = (Rotation._degree_360.value - self._rotation.value)
        # noinspection PyProtectedMember
        numb_90_degree_rotations %= Rotation._degree_360.value
        numb_90_degree_rotations /= Rotation.degree_90.value
        # Handle case where loop is not run.
        (unrotated_x, unrotated_y) = (rotated_x, rotated_y)
        for i in range(0, numb_90_degree_rotations):

            (unrotated_x, unrotated_y) = self._get_rotated_coordinates(rotated_x, rotated_y, Rotation.degree_90 )
            # Updated rotated values in case need to rotated again
            (rotated_x, rotated_y) = (unrotated_x, unrotated_y)

        # Return the unrotated coordinates
        return unrotated_x, unrotated_y

    def _get_rotated_coordinates(self, unrotated_x, unrotated_y, rotation=None):
        """
        Given a specified puzzle piece with its own rotation and width,
        this function calculates rotated x and y values.  This keeps the
        object data unchanged during a rotation.

        :param unrotated_x:   x coordinate to rotate
        :param unrotated_y:   y coordinate to rotate
        :param rotation:      Number of degrees to rotate.  Uses Rotation enum class.  If no rotation is specied, uses
            the specified object's rotation.
        :return:    Tuple in format (rotated_x, rotated_y)

        """

        if rotation is None:  # If no rotation i specified then use the specified object's rotation
            rotation = self._rotation

        # Calculate number of 90 degree rotations to perform.
        numb_90_degree_rotations = rotation.value / Rotation.degree_90.value
        # Each iteration of the loop rotates the x, y coordinates 90 degrees
        # Each a 180 degree rotation is two 90 degree rotations
        (rotated_x, rotated_y) = (unrotated_x, unrotated_y)  # In case loop is never run
        for i in range(0, numb_90_degree_rotations):

            # Calculate the rotated x and y
            rotated_x = self.get_width() - 1 - unrotated_y
            rotated_y = unrotated_x
            # update the unrotated x/y values in case need to rotate again
            (unrotated_x, unrotated_y) = (rotated_x, rotated_y)

        # Return the rotated x and y coordinates
        return rotated_x, rotated_y


    def putpixel(self, x, y, pixel):
        """
        Update's an image's pixel value.
        :param x:       Pixel's x coordinate
        :param y:       Pixel's y coordinate
        :param pixel:   New pixel value.  A Tuple in the format (Red, Green, Blue)
        """

        if x >= self._width:
            raise ValueError("Pixel's \"x\" coordinate must be between 0 and width - 1")
        if y >= self._width:
            raise ValueError("Pixel's \"y\" coordinate must be between 0 and width - 1")

        # Correct for any rotation
        (unrotated_x, unrotated_y) = self._get_unrotated_coordinates(x, y)
        # Updated the pixel value using the unrotated x and y coordinates
        self._pixels.putpixel((unrotated_x, unrotated_y), pixel)

    def getpixel(self, x, y):
        """
        Get a pixel from a puzzle piece.
        :param x: int   X coordinate for the pixel to get
        :param y: int   Y coordinate for the pixel to get
        :return:        Pixel at the piece's specified pixel.
        """
        if x >= self._width:
            raise ValueError("Pixel's \"x\" coordinate must be between 0 and width - 1")
        if y >= self._width:
            raise ValueError("Pixel's \"y\" coordinate must be between 0 and width - 1")

        return self._pixels.getpixel((x, y))

    def set_rotation(self, rotation):
        """
        Sets the puzzle piece's rotation.
        :type rotation: Rotation
        :param rotation: The rotation of the specified puzzle piece.
        """
        if rotation != Rotation.degree_0:
            PuzzlePiece.assert_rotation_enabled()

        if rotation.value % Rotation.degree_90.value != 0:
            raise ValueError("Invalid rotation value.")

        self._rotation = rotation

    @staticmethod
    def assert_rotation_enabled():
        if not PuzzlePiece.rotation_enabled:
            raise AttributeError("Cannot set rotation.  Rotation is disabled for puzzle pieces.")

    def get_rotation(self):
        """
        Returns the puzzle piece's rotation.
        :return: Rotation Puzzle piece's rotation
        """

        return self._rotation

    def randomize_rotation(self):
        """
        Randomly sets the rotation of a piece.
        """

        PuzzlePiece.assert_rotation_enabled()
        # Get the list of rotations
        all_rotations = Rotation.get_all_rotations()
        # Set the rotation to a randomly selected value
        i = random.randint(0, len(all_rotations) - 1)
        self.set_rotation(all_rotations[i])


    def get_width(self):
        """
        Gets the width of the puzzle piece in pixels
        :return: Number of pixels wide the image is
        """

        return self._width
