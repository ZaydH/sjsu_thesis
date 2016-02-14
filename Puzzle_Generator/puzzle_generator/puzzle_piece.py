"""Jigsaw Puzzle Piece Class

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""

from enum import Enum
import random
from PIL import Image


class Rotation(Enum):
    """Puzzle Piece Rotation

    Enumerated type for representing the amount of rotation for a puzzle piece.

    Note:
        Pieces can only be rotated in 90 degree increments.

    """
    degree_0 = 0        # No rotation
    degree_90 = 90      # 90 degree rotation
    degree_180 = 180    # 180 degree rotation
    degree_270 = 270    # 270 degree rotation
    _degree_360 = 360   # Internal use only.  Same as 0 degree rotation.

    @staticmethod
    def degrees(n):
        """
        Factory method for getting the rotation enum based solely on the number of degrees of rotation.

        Args:
            n (int): Rotation in degrees.  Must be in 90 degree increments (e.g. 0, 90, 180, 270 only)

        Returns (Rotation): Rotation enum for the specified number of degrees.

        """
        # noinspection PyUnresolvedReferences
        assert n % Rotation.degree_90.value == 0 and n < Rotation._degree_360.value
        rot_cnt = n / 90
        all_rotations = [Rotation.degree_0, Rotation.degree_90, Rotation.degree_180, Rotation.degree_270]
        return all_rotations[rot_cnt]

    @staticmethod
    def get_all_rotations():
        """
        Gets a list of all valid values of Rotation

        Returns:
            List[Rotation]: List of all valid rotation values in increasing order.

        """

        return [Rotation.degree_0, Rotation.degree_90, Rotation.degree_180, Rotation.degree_270]

    # noinspection PyUnresolvedReferences
    def get_numb_90_rotations_to_other(self, other):
        """
        Determines the number of 90 degree rotations from the current rotation to another rotation.

        Args:
            other (Rotation: Another rotation

        Returns (int): Number of 90 degree rotations required.  It is bounded between 0 and 3 as that is the minimum
        and maximum of clockwise rotations to move between any 90 degree angles without doing a complete revolution.
        """
        degree_difference = other.value - self.value
        if degree_difference < 0:
            degree_difference += Rotation._degree_360.value
        return degree_difference / Rotation.degree_90.value


class PieceSide(Enum):
    top_side = 0
    right_side = 1
    bottom_side = 2
    left_side = 3

    @staticmethod
    def get_all_sides():
        """
        Gets all possible sides of a puzzle piece.

        Returns ([PieceSide]): An list of puzzle pieces sides in the order: [top_side, right_side, bottom_side,
        left_side].  Hence, the sides are listed clockwise starting from the top side.

        """
        return [PieceSide.top_side, PieceSide.right_side, PieceSide.bottom_side, PieceSide.left_side]

    @property
    def paired_edge(self):
        """
        For a given side of a piece, this function returns the corresponding
        piece of the neighbor that would be on that side.

        Returns: Corresponding edge for the neighboring piece that would be
        on the side of this piece.  For example, the right side of this piece
        is adjacent to the left side of neighbor and vice versa.

        """
        if self == PieceSide.top_side:
            return PieceSide.bottom_side

        elif self == PieceSide.bottom_side:
            return PieceSide.top_side

        elif self == PieceSide.left_side:
            return PieceSide.right_side

        elif self == PieceSide.right_side:
            return PieceSide.bottom_side

        else:
            assert False


class PuzzlePiece(object):

    # Minimum width for a puzzle piece.
    MINIMUM_WIDTH = 10

    # May want to disable rotation so have a check for that.
    rotation_enabled = True

    def __init__(self, width=None, actual_location=None, image=None, start_x=None, start_y=None):
        """
        Constructor of an empty puzzle piece object

        Args:
            width (int): The width (and length) of the puzzle piece in number of pixels.
            actual_location ([int, int]): Tuple of two ints showing the actual location of the image in the
                                          original image.
            image (Optional[Image]): The second parameter. Defaults to None.
            start_x (Optional[int]): Number of pieces along the image **width**. Defaults to None.  Used to extract
                                     the pixels from the passed in image.
            start_y (Optional[int]): Number of pieces along the image **height**. Defaults to None.  Used to extract
                                     the pixels from the passed in image.

        Returns:
            New puzzle piece

        """

        # Number of pixels in the width of the piece
        if width < PuzzlePiece.MINIMUM_WIDTH:
            raise ValueError("Specified width is less than the minimum width of %d" % PuzzlePiece.MINIMUM_WIDTH)
        self._width = width

        # Create the matrix for storing the pixel information
        # noinspection PyUnusedLocal
        self._pixels = Image.new("RGB", (width, width), "white")
        if image is None and (actual_location is not None or start_x is not None or start_y is not None):
            raise ValueError("Argument image cannot be null if argument actual_location, start_x, or start_y " +
                             "are specified")
        if image is not None and (actual_location is None or start_x is None or start_y is None):
            raise ValueError('If argument image is specified, neither actual_location, start_x, or start_y can be None')
        # If a valid pixel array is passed, then copy the contents to the piece.
        if image is not None:
            # Extract a subimage
            self._actual_location = actual_location
            # noinspection PyTypeChecker
            box = (start_x, start_y, start_x + width, start_y + width)
            self._pixels = image.crop(box)
            assert(self._pixels.size == (width, width))
        else:
            self._actual_location = None

        # Add information for when this piece appears in the puzzle
        self._assigned_location = None
        # Initialize empty neighbor list.
        # noinspection PyUnusedLocal
        self._assigned_sides = [None for x in PieceSide.get_all_sides()]

        # Set a random rotation
        self._rotation = None  # Create a reference to prevent compiler warnings
        self.rotation = Rotation.degree_0

    def _get_unrotated_coordinates(self, rotated_x, rotated_y):
        """X-Y Coordinate **Un-**Rotator

        For a given puzzle piece with its own rotation and width and already
        rotated x/y coordinates, this function calculates unrotated x/y coordinates.

        This function is needed since the puzzle piece's pixels are stored unrotated.
        Hence, we need to rotate rotated x/y coordinates into their unrotated
        equivalents.

        Args:
            rotated_x (int): An already rotated x coordinate
            rotated_y (int): An already rotated y coordinate

        Returns:
            (int, int): Tuple in the format (unrotated_x, unrotated_y)

        """

        assert self.rotation is not None

        # Number of rotations is equal to the number of rotations required to make the
        # rotated x/y values full circle.  Hence, 0degree rotation takes no 90 degrees.
        # 90degree rotation takes 3 rotations, 180degrees takes 2, and 270 takes 1.
        # noinspection PyRedundantParentheses,PyTypeChecker
        numb_90_degree_rotations = (self.rotation).get_numb_90_rotations_to_other(Rotation.degrees(0))
        # Handle case where loop is not run.
        (unrotated_x, unrotated_y) = (rotated_x, rotated_y)
        for i in range(0, numb_90_degree_rotations):

            # noinspection PyTypeChecker
            (unrotated_x, unrotated_y) = self._get_rotated_coordinates(rotated_x, rotated_y, Rotation.degree_90)
            # Updated rotated values in case need to rotated again
            (rotated_x, rotated_y) = (unrotated_x, unrotated_y)

        # Return the unrotated coordinates
        return unrotated_x, unrotated_y

    def _get_rotated_coordinates(self, unrotated_x, unrotated_y, rotation=None):
        """ XY Coordinator **Rotator**

        Given a specified puzzle piece with its own rotation and width,
        this function calculates rotated x and y values.  This keeps the
        object data unchanged during a rotation.

        Args:
            unrotated_x (int): X coordinate to rotate
            unrotated_y (int): Y coordinate to rotate
            rotation (Rotation): Number of degrees to rotate.  Uses Rotation enum class.  If no rotation is
                specified, then the function the specified object's rotation.

        Returns:
            (int, int): Tuple of rotated XY coordinates in the format (rotated_x, rotated_y)

        """

        if rotation is None:  # If no rotation i specified then use the specified object's rotation
            rotation = self._rotation

        # Calculate number of 90 degree rotations to perform.
        # noinspection PyUnresolvedReferences
        numb_90_degree_rotations = (Rotation.degrees(0)).get_numb_90_rotations_to_other(rotation)
        # Each iteration of the loop rotates the x, y coordinates 90 degrees
        # Each a 180 degree rotation is two 90 degree rotations
        (rotated_x, rotated_y) = (unrotated_x, unrotated_y)  # In case loop is never run
        for i in range(0, numb_90_degree_rotations):

            # Calculate the rotated x and y
            rotated_x = self.width - 1 - unrotated_y
            rotated_y = unrotated_x
            # update the unrotated x/y values in case need to rotate again
            (unrotated_x, unrotated_y) = (rotated_x, rotated_y)

        # Return the rotated x and y coordinates
        return rotated_x, rotated_y

    @property
    def assigned_location(self):
        return self._assigned_location

    @assigned_location.setter
    def assigned_location(self, assigned_location):
        """
        Updates the assigned location of a piece in the puzzle.

        Args:
            assigned_location ([int, int]): Location where the piece is assigned in the X/Y grid.
        """
        assert len(assigned_location) == 2
        self._assigned_location = assigned_location

    @property
    def image(self):
        """
        Gets the image for a particular piece.  It does appropriately

        Returns:
            (Image): Piece's image with appropriate rotation

        """
        return self._pixels.rotate(self._rotation.value)

    def get_neighbor_coordinate(self, piece_side):
        """
        For the implicit piece, this function returns the x/y coordinates of the neighbor piece on the specified
        side.

        Args:
            piece_side (PieceSide): Side of the piece where the neighbor
                                    is located.

        Returns (int, int): Tuple in the form of (x_coordinate, y_coordinate) for the neighbor piece.

        """

        # Verify the piece has an actual location.
        assert self._assigned_location is not None
        assigned_x, assigned_y = self._assigned_location

        # Depending on the piece side, select the appropriate piece
        if piece_side == PieceSide.top_side:
            assert assigned_y > 0  # Verify not off the board
            return assigned_x, assigned_y - 1

        elif piece_side == PieceSide.bottom_side:
            return assigned_x, assigned_y + 1

        elif piece_side == PieceSide.left_side:
            assert assigned_x > 0  # Verify not off the board
            return assigned_x - 1, assigned_y

        elif piece_side == PieceSide.right_side:
            return assigned_x + 1, assigned_y
        else:
            assert False

    # noinspection SpellCheckingInspection
    def putpixel(self, x, y, pixel):
        """Pixel Updater

        Update's an image's pixel value.

        Args:
            x (int):  Pixel's x coordinate
            y (int):  Pixel's y coordinate
            pixel: New pixel value.  A Tuple in the format (Red, Green, Blue)

        Raises:
            ValueError: x and y must be between 0 and (puzzle piece width - 1)

        """
        if x >= self._width:
            raise ValueError("Pixel's \"x\" coordinate must be between 0 and width - 1")
        if y >= self._width:
            raise ValueError("Pixel's \"y\" coordinate must be between 0 and width - 1")

        # Correct for any rotation
        (unrotated_x, unrotated_y) = self._get_unrotated_coordinates(x, y)
        # Updated the pixel value using the unrotated x and y coordinates
        self._pixels.putpixel((unrotated_x, unrotated_y), pixel)

    # noinspection SpellCheckingInspection
    def getpixel(self, x, y):
        """Pixel Accessor

        Gets a pixel from a puzzle piece.

        Args:
            x (int): X coordinate for the pixel to get
            y (int): Y coordinate for the pixel to get

        Returns:
            Pixel at the piece's specified pixel.

        Raises:
            ValueError: x and y must be between 0 and (puzzle piece width - 1)

        """
        if x >= self._width:
            raise ValueError("Pixel's \"x\" coordinate must be between 0 and width - 1")
        if y >= self._width:
            raise ValueError("Pixel's \"y\" coordinate must be between 0 and width - 1")

        return self._pixels.getpixel((x, y))

    def get_edge_start_corner_coordinate_and_pixel_step(self, piece_side):
        """

        Args:
            piece_side (PieceSide): Side of the piece whose starting corner pixel is being returned.

        Returns ([[int], [int]]:

        """
        assert piece_side in PieceSide.get_all_sides()

        # Handle the dimension that is changing first.
        # For a top to bottom pair, it is the x dimension
        if piece_side == PieceSide.top_side or piece_side == PieceSide.left_side:
            start_coord = (0, 0)
        elif piece_side == PieceSide.right_side:
            start_coord = (self._width - 1, 0)
        # For a left to right pair, it is the y dimension
        else:
            start_coord = (0, self._width - 1)

        # Handle the dimension for each side that is unchanging
        if piece_side == PieceSide.top_side or piece_side == PieceSide.bottom_side:
            offset_step = (1, 0)
        else:
            offset_step = (0, 1)

        # Return the start and end x/y coordinates
        return start_coord, offset_step

    @property
    def rotation(self):
        """Puzzle Piece Rotation Accessor

        Get's a puzzle piece's rotation setting.  In most cases, a user should not need to call this
        function.  Once 'rotation' is called, the return X-Y coordinates are returned
        rotated.

        Returns:
            Rotation: Puzzle piece's rotation

        """
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        """Piece Rotation Modifier

        Sets the puzzle piece's rotation.

        Args:
            rotation (Rotation): The rotation of the specified puzzle piece.

        """
        if rotation != Rotation.degree_0:
            PuzzlePiece.assert_rotation_enabled()

        # noinspection PyUnresolvedReferences
        if rotation.value % Rotation.degree_90.value != 0:
            raise ValueError("Invalid rotation value.")

        self._rotation = rotation

    @staticmethod
    def assert_rotation_enabled():
        """Illegal Rotation Checker

        Checks if there is an attempt to rotate a puzzle piece while piece rotation is disabled.

        Raises:
            AttributeError: Variable 'rotation_enabled' is set to false.

        """
        if not PuzzlePiece.rotation_enabled:
            raise AttributeError("Cannot set rotation.  Rotation is disabled for puzzle pieces.")

    def randomize_rotation(self):
        """Puzzle Piece Rotation Randomizer

        Randomly sets the puzzle piece's rotation.

        """
        PuzzlePiece.assert_rotation_enabled()
        # Get the list of rotations
        all_rotations = Rotation.get_all_rotations()
        # Set the rotation to a randomly selected value
        i = random.randint(0, len(all_rotations) - 1)
        self.rotation = all_rotations[i]

    @property
    def width(self):
        """Puzzle Piece Width Accessor

        Gets the width of the puzzle piece in pixels

        Returns:
            int: The width in number of pixels of the puzzle.

        """
        return self._width

    @staticmethod
    def calculate_pieces_edge_distance(piece1, piece1_side, piece2):
        """

        Args:
            piece1 (PuzzlePiece): A single puzzle piece
            piece1_side (PieceSide): The side of piece1 where piece2 will be placed.
            piece2 (PuzzlePiece): Another puzzle piece that will be placed adjacent
            to piece1.

        Returns (int): Sum of the squared difference between the RGB values of the pixels
        along the the specified side of piece1 and the corresponding side of piece2.

        """

        # Verify the two pieces have the same width
        assert piece1.width == piece2.width

        # Get piece1's coordinate information
        piece1_start_coord, piece1_pixel_step = piece1.get_edge_start_corner_coordinate_and_pixel_step(piece1_side)

        # Get the piece2 coordinate information
        piece2_side = piece1_side.paired_edge
        piece2_start_coord, piece2_pixel_step = piece2.get_edge_start_corner_coordinate_and_pixel_step(piece2_side)

        pixel_sum = 0
        for i in range(0, piece1.width):
            # Get the pixel for piece1
            piece1_pixel_coord = [start + i * offset for start, offset in zip(piece1_start_coord, piece1_pixel_step)]
            piece1_pixel = (piece1_pixel_coord[0] + i * offset[0], piece1_pixel_coord[1] + i * offset[1])

            # Get the pixel for piece2
            piece2_pixel_coord = [start + i * offset for start, offset in zip(piece2_start_coord, piece2_pixel_step)]
            piece2_pixel = (piece2_pixel_coord[0] + i * offset[0], piece2_pixel_coord[1] + i * offset[1])

            # For this pixel pair, add the sum of their respective RGB differences
            pixel_sum += sum([(pixel1_rgb - pixel2_rgb) ^ 2 for pixel1_rgb, pixel2_rgb in zip(piece1_pixel,
                                                                                              piece2_pixel)])
        return pixel_sum
