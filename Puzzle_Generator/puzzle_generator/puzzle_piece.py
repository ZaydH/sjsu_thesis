from PIL import Image
from enum import Enum
import random

class PuzzlePiece:

    class Rotation(Enum):
        degree_0 = 0
        degree_90 = 90
        degree_180 = 180
        degree_270 = 270
        _degree_360 = 360  # Internal use only

        @staticmethod
        def get_all_rotations():
            return [PuzzlePiece.Rotation.degree_0, PuzzlePiece.Rotation.degree_90,
                    PuzzlePiece.Rotation.degree_180, PuzzlePiece.Rotation.degree_270]


    def __init__(self, width):
        # Number of pixels in the width of the piece
        self._width = width
        # Create the matrix for storing the pixel information
        self._pixels = [ [ -1 for x in range(0, width)] for x in range(0,width)]
        # Set a random rotation
        self._rotation = -1 # Create a reference to prevent compiler warnings
        self.set_rotation(PuzzlePiece.Rotation.degree_0)

    def _get_unrotated_coordinates(self, rotated_x, rotated_y):
        """
        For a given puzzle piece with its own rotation and width and already
        rotated x/y coordinates, this function calculates unrotated x/y coordinates.

        This function is needed since the puzzle piece's pixels are stored unrotated.
        Hence, we need to rotate rotated x/y coordinates into their unrotated
        equivalents.

        :param rotated_x: An already rotated x coordinate
        :param rotated_y: An already rotated y coordinate

        :return: Tuple in the format (unrotated_x, unrotated_y)
        """
        # Number of rotations is equal to the number of rotations required to make the
        # rotated x/y values full circle.  Hence, 0degree rotation takes no 90 degrees.
        # 90degree rotation takes 3 rotations, 180degrees takes 2, and 270 takes 1.
        # noinspection PyProtectedMember,PyUnresolvedReferences
        numb_90_degree_rotations = (PuzzlePiece.Rotation._degree_360.value - self._rotation.value) \
                                   % PuzzlePiece.Rotation._degree_360.value
        numb_90_degree_rotations /= PuzzlePiece.Rotation.degree_90.value
        # Handle case where loop is not run.
        (unrotated_x, unrotated_y) = (rotated_x, rotated_y)
        for i in range(0, numb_90_degree_rotations):
            (unrotated_x, unrotated_y) = self._get_rotated_coordinates(rotated_x,
                                                                       rotated_y,
                                                                       PuzzlePiece.Rotation.degree_90 )
            # Updated rotated values in case need to rotated again
            (rotated_x, rotated_y) = (unrotated_x, unrotated_y)
        # Return the unrotated coordinates
        return unrotated_x, unrotated_y

    def _get_rotated_coordinates(self, unrotated_x, unrotated_y, rotation=-1):
        """
        Given a specified puzzle piece with its own rotation and width,
        this function calculates rotated x and y values.  This keeps the
        object data unchanged during a rotation.

        :param unrotated_x:   x coordinate to rotate
        :param unrotated_y:   y coordinate to rotate
        :param rotation:      Number of degrees to rotate.  Uses Rotation enum class.  If
                              no rotation is specied, uses the specified object's rotation.
        :return:    Tuple in format (rotated_x, rotated_y)
        """
        # If no rotation i specified then use the specified object's rotation
        if rotation == -1:
            rotation = self._rotation
        # Calculate number of 90 degree rotations to perform.
        numb_90_degree_rotations = rotation.value / PuzzlePiece.Rotation.degree_90.value
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


    def setpixel(self, x, y, pixel):
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
        self._pixels[unrotated_x][unrotated_y] = pixel

    def getpixel(self, x, y, pixel):
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
        self._pixels = pixel

    def set_rotation(self, rotation):
        """
        Sets the puzzle piece's rotation.

        :param rotation: Type Rotation.  The rotation of this
                         puzzle piece.
        """
        if rotation.value % self.Rotation.degree_90.value != 0:
            raise ValueError("Invalid rotation value.")
        self._rotation = rotation

    def get_rotation(self):
        """
        Returns the puzzle piece's rotation.
        :return: The
        """
        return self._rotation

    def get_width(self):
        """
        Gets the width of the puzzle piece in pixels

        :return: Number of pixels wide the image is
        """
        return self._width