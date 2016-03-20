"""Jigsaw Puzzle Object

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""
import copy
import os
import math
import random

import numpy
import cv2  # OpenCV
from enum import Enum

from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePiece, PuzzlePieceRotation


class PuzzleType(Enum):
    """
    Type of the puzzle to solve.  Type 1 has no piece rotation while type 2 allows piece rotation.
    """

    type1 = 1
    type2 = 2


class ImageColor(Enum):
        """
        Used to create solid color images for base images and for image manipulation.
        """
        black = 1


class Puzzle(object):
    """
    Puzzle Object represents a single Jigsaw Puzzle.  It can import a puzzle from an image file and
    create the puzzle pieces.
    """

    print_debug_messages = True

    # DEFAULT_PIECE_WIDTH = 28  # Width of a puzzle in pixels
    DEFAULT_PIECE_WIDTH = 25  # Width of a puzzle in pixels

    # Define the number of dimensions in the BGR space (i.e. blue, green, red)
    NUMBER_BGR_DIMENSIONS = 3

    export_with_border = True
    border_width = 3
    border_outer_stripe_width = 1

    def __init__(self, id_number, image_filename=None, piece_width=None):
        """Puzzle Constructor

        Constructor that will optionally load an image into the puzzle as well.

        Args:
            id_number (int): ID number for the image.  It is used for multiple image puzzles.
            image_filename (Optional str): File path of the image to load
            piece_width (Optional int): Width of a puzzle piece in pixels

        Returns:
            Puzzle Object

        """
        # Internal Pillow Image object.
        self._id = id_number
        self._img = None
        self._img_LAB = None

        # Initialize the puzzle information.
        self._grid_size = None
        self._piece_width = piece_width if piece_width is not None else Puzzle.DEFAULT_PIECE_WIDTH
        self._img_width = None
        self._img_height = None

        # No pieces for the puzzle yet.
        self._pieces = []

        if image_filename is None:
            self._filename = ""
            return

        # Stores the image file and then loads it.
        self._filename = image_filename
        self._load_puzzle_image()

        # Make image pieces.
        self.make_pieces()

    def _load_puzzle_image(self):
        """Puzzle Image Loader

        Loads the puzzle image file a specified filename.  Loads the specified puzzle image into memory.
        It also stores information on the puzzle dimensions (e.g. width, height) into the puzzle object.

        """

        # If the filename does not exist, then raise an error.
        if not os.path.exists(self._filename):
            raise ValueError("Invalid \"%s\" value.  File does not exist" % self._filename)

        self._img = cv2.imread(self._filename)  # Note this imports in BGR format not RGB.
        if self._img is None:
            raise IOError("Unable to load the image at the specified location \"%s\"." % self._filename)

        # Get the image dimensions.
        self._img_height, self._img_width = self._img.shape[:2]

        # Make a LAB version of the image.
        self._img_LAB = cv2.cvtColor(self._img, cv2.COLOR_BGR2LAB)

    def make_pieces(self):
        """Puzzle Generator

        Given a puzzle, this function turns the puzzle into a set of pieces.

        **Note:** When creating the pieces, some of the source image may need to be discarded
        if the image size is not evenly divisible by the number of pieces specified
        as parameters to this function.

        """

        # Calculate the piece information.
        numb_cols = int(math.floor(self._img_width / self.piece_width))  # Floor in python returns a float
        numb_rows = int(math.floor(self._img_height / self.piece_width))  # Floor in python returns a float
        if numb_cols == 0 or numb_rows == 0:
            raise ValueError("Image size is too small for the image.  Check your setup")

        # Store the grid size.
        self._grid_size = (numb_rows, numb_cols)

        # Store the original width and height and recalculate the new width and height.
        original_width = self._img_width
        original_height = self._img_height
        self._img_width = numb_cols * self.piece_width
        self._img_height = numb_rows * self.piece_width

        # Shave off the edge of the image LAB and BGR images
        puzzle_upper_left = ((original_height - self._img_height) / 2, (original_width - self._img_width) / 2)
        self._img = Puzzle.extract_subimage(self._img, puzzle_upper_left, (self._img_height, self._img_width))
        self._img_LAB = Puzzle.extract_subimage(self._img_LAB, puzzle_upper_left, (self._img_height, self._img_width))

        # Break the board into pieces.
        piece_size = (self.piece_width, self.piece_width)
        self._pieces = []  # Create an empty array to hold the puzzle pieces.
        for row in range(0, numb_rows):
            for col in range(0, numb_cols):
                piece_upper_left = (puzzle_upper_left[0] + row * piece_size[0],
                                    puzzle_upper_left[1] + col * piece_size[1])
                piece_img = Puzzle.extract_subimage(self._img_LAB, piece_upper_left, piece_size)

                # Create the puzzle piece and assign to the location.
                location = (row, col)
                self._pieces.append(PuzzlePiece(self._id, location, piece_img))

    @property
    def id_number(self):
        """
        Puzzle Identification Number

        Gets the identification number for a puzzle.

        Returns (int): Identification number for the puzzle
        """
        return self._id

    @property
    def pieces(self):
        """
        Gets all of the pieces in this puzzle.

        Returns ([PuzzlePiece]):
        """
        return self._pieces

    @property
    def piece_width(self):
        """
        Gets the size of a puzzle piece.

        Returns (int): Height/width of each piece in pixels.

        """
        return self._piece_width

    @staticmethod
    def reconstruct_from_pieces(pieces, id_numb=-1, display_image=False):
        """
        Constructs a puzzle from a set of pieces.

        Args:
            pieces ([PuzzlePiece]): Set of puzzle pieces that comprise the puzzle.
            id_numb (Optional int): Identification number for the puzzle
            display_image (Optional Boolean): Select whether to display the eimage at the end of reconstruction

        Returns (Puzzle):
        Puzzle constructed from the pieces.
        """

        if len(pieces) == 0:
            raise ValueError("Error: Each puzzle must have at least one piece.")

        # Create the puzzle to return.  Give it an ID number.
        output_puzzle = Puzzle(id_numb)
        output_puzzle._id = id_numb

        # Create a copy of the pieces.
        output_puzzle._pieces = copy.deepcopy(pieces)

        # Get the first piece and use its information
        first_piece = output_puzzle._pieces[0]
        output_puzzle._piece_width = first_piece.width

        # Find the min and max row and column.
        (min_row, max_row, min_col, max_col) = output_puzzle.get_min_and_max_row_and_columns()

        # Normalize their locations based off all the pieces in the board.
        for piece in output_puzzle._pieces:
            loc = piece.location
            piece.location = (loc[0] - min_row, loc[1] - min_col)

        # Store the grid size
        output_puzzle._grid_size = (max_row - min_row + 1, max_col - min_col + 1)
        # Calculate the size of the image
        output_puzzle._img_width = output_puzzle._grid_size[1] * output_puzzle.piece_width
        output_puzzle._img_height = output_puzzle._grid_size[0] * output_puzzle.piece_width

        # Define the numpy array that will hold the reconstructed image.
        puzzle_array_size = (output_puzzle._img_height, output_puzzle._img_width)
        # noinspection PyTypeChecker
        output_puzzle._img = Puzzle.create_solid_bgr_image(puzzle_array_size, ImageColor.black)

        # Insert the pieces into the puzzle
        for piece in output_puzzle._pieces:
            output_puzzle.insert_piece_into_image(piece)

        # Convert the image to LAB format.
        output_puzzle._img_LAB = cv2.cvtColor(output_puzzle._img, cv2.COLOR_BGR2LAB)
        if display_image:
            Puzzle.display_image(output_puzzle._img)

        return output_puzzle

    def randomize_puzzle_piece_locations(self):
        """
        Puzzle Piece Location Randomizer

        Randomly assigns puzzle pieces to different locations.
        """

        # Get all locations in the image.
        all_locations = []
        for piece in self._pieces:
            all_locations.append(piece.location)

        # Shuffle the image locations
        random.shuffle(all_locations)

        # Reassign the pieces to random locations
        for i in range(0, len(self._pieces)):
            self._pieces[i].location = all_locations[i]

    def randomize_puzzle_piece_rotations(self):
        """
        Puzzle Piece Rotation Randomizer

        Assigns a random rotation to each piece in the puzzle.
        """
        for piece in self._pieces:
            piece.rotation = PuzzlePieceRotation.random_rotation()

    def get_min_and_max_row_and_columns(self):
        """
        Min/Max Row and Column Finder

        For a given set of pieces, this function returns the minimum and maximum of the columns and rows
        across all of the pieces.

        Returns ([int]):
        Tuple in the form: (min_row, max_row, min_column, max_column)
        """
        first_piece = self._pieces[0]
        min_row = max_row = first_piece.location[0]
        min_col = max_col = first_piece.location[1]
        for i in range(0, len(self._pieces)):
            # Verify all pieces are the same size
            if Puzzle.print_debug_messages:
                assert(self.piece_width == self._pieces[i].width)
            # Get the location of the piece
            temp_loc = self._pieces[i].location
            # Update the min and max row if needed
            if min_row > temp_loc[0]:
                min_row = temp_loc[0]
            elif max_row < temp_loc[0]:
                max_row = temp_loc[0]
            # Update the min and max column if needed
            if min_col > temp_loc[1]:
                min_col = temp_loc[1]
            elif max_col < temp_loc[1]:
                max_col = temp_loc[1]

        # Return the minimum and maximum row/column information
        return min_row, max_row, min_col, max_col

    def _assign_all_pieces_to_original_location(self):
        """Piece Correct Assignment

        Test Method Only. Assigns each piece to its original location for debug purposes.
        """
        for piece in self._pieces:
            # noinspection PyProtectedMember
            piece._assign_to_original_location()

    # noinspection PyUnusedLocal
    @staticmethod
    def create_solid_bgr_image(size, color):
        """
        Solid BGR Image Creator

        Creates a BGR Image (i.e. NumPy) array of the specified size.

        RIGHT NOW ONLY BLACK is supported.

        Args:
            size ([int]): Size of the image in height by width
            color (ImageColor): Solid color of the image.

        Returns:
        NumPy array representing a BGR image of the specified solid color
        """
        dimensions = (size[0], size[1], Puzzle.NUMBER_BGR_DIMENSIONS)
        return numpy.zeros(dimensions, numpy.uint8)

    @staticmethod
    def extract_subimage(img, upper_left, size):
        """
        Given an image (in the form of a Numpy array) extract a subimage.

        Args:
            img : Image in the form of a numpy array.
            upper_left ([int]): upper left location of the image to extract
            size ([int]): Size of the of the sub

        Returns:
        Sub image as a numpy array
        """

        # Calculate the lower right of the image
        img_end = []
        for i in range(0, 2):
            img_end.append(upper_left[i] + size[i])

        # Return the sub image.
        return img[upper_left[0]:img_end[0], upper_left[1]:img_end[1], :]

    def insert_piece_into_image(self, piece):
        """
        Takes a puzzle piece and converts its image into BGR then adds it to the master image.

        Args:
            piece (PuzzlePiece): Puzzle piece to be inserted into the puzzle's image.
        """
        piece_loc = piece.location

        # Define the upper left corner of the piece to insert
        upper_left = (piece_loc[0] * piece.width, piece_loc[1] * piece.width)

        # Select whether to display the image rotated
        piece_bgr = piece.bgr_image()
        if piece.rotation is None or piece.rotation == PuzzlePieceRotation.degree_0:
            Puzzle.insert_subimage(self._img, upper_left, piece_bgr)
        else:
            rotated_img = numpy.rot90(piece_bgr, piece.rotation.value / 90)
            Puzzle.insert_subimage(self._img, upper_left, rotated_img)

    @staticmethod
    def insert_subimage(master_img, upper_left, subimage):
        """
        Given an image (in the form of a NumPy array), insert another image into it.

        Args:
            master_img : Image in the form of a NumPy array where the sub-image will be inserted
            upper_left ([int]): upper left location of the the master image where the sub image will be inserted
            subimage ([int]): Sub-image to be inserted into the master image.

        Returns:
        Sub image as a numpy array
        """

        # Verify the upper left input value is valid.
        if Puzzle.print_debug_messages and upper_left[0] < 0 or upper_left[1] < 0:
            raise ValueError("Error: upper left is off the image grid. Row and column must be >=0")

        # Calculate the lower right of the image
        subimage_shape = subimage.shape
        bottom_right = [upper_left[0] + subimage_shape[0], upper_left[1] + subimage_shape[1]]

        # Verify that the shape information is valid.
        if Puzzle.print_debug_messages:
            master_shape = master_img.shape
            assert master_shape[0] >= bottom_right[0] and master_shape[1] >= bottom_right[1]

        # Insert the subimage.
        master_img[upper_left[0]:bottom_right[0], upper_left[1]:bottom_right[1], :] = subimage

    @staticmethod
    def display_image(img):
        """
        Displays the image in a window for debug viewing.

        Args:
            img: OpenCV image in the form of a Numpy array

        """
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def _save_to_file(filename, img):
        """
        Save Image to a File

        Saves any numpy array to an image file.

        Args:
            filename (str): Filename and path to save the OpenCV image.
            img: OpenCV image in the form of a Numpy array
        """
        cv2.imwrite(filename, img)

    def save_to_file(self, filename):
        """
        Save Puzzle to a File

        Saves a puzzle to the specified file name.

        Args:
            filename (str): Filename and path to save the OpenCV image.
        """
        Puzzle._save_to_file(filename, self._img)


class PuzzleTester(object):
    """
    Puzzle tester class used for debugging the solver.
    """

    PIECE_WIDTH = 5
    NUMB_PUZZLE_PIECES = 9
    GRID_SIZE = (int(math.sqrt(NUMB_PUZZLE_PIECES)), int(math.sqrt(NUMB_PUZZLE_PIECES)))
    NUMB_PIXEL_DIMENSIONS = 3

    TEST_ARRAY_FIRST_PIXEL_VALUE = 0

    # Get the information on the test image
    TEST_IMAGE_FILENAME = ".\\test\\test.jpg"
    TEST_IMAGE_WIDTH = 300
    TEST_IMAGE_HEIGHT = 200

    @staticmethod
    def build_pixel_list(start_value, is_row, reverse_list=False):
        """
        Pixel List Builder

        Given a starting value for the first pixel in the first dimension, this function gets the pixel values
        in an array similar to a call to "get_row_pixels" or "get_column_pixels" for a puzzle piece.

        Args:
            start_value (int): Value of the first (i.e. lowest valued) pixel's first dimension

            is_row (bool): True if building a pixel list for a row and "False" if it is a column.  This is used to
            determine the stepping factor from one pixel to the next.

            reverse_list (bool): If "True", HIGHEST valued pixel dimension is returned in the first index of the list
            and all subsequent pixel values are monotonically DECREASING.  If "False", the LOWEST valued pixel dimension
            is returned in the first index of the list and all subsequent pixel values are monotonically increasing.

        Returns ([int]): An array of individual values simulating a set of pixels
        """

        # Determine the pixel to pixel step size
        if is_row:
            pixel_offset = PuzzleTester.NUMB_PIXEL_DIMENSIONS
        else:
            pixel_offset = PuzzleTester.row_to_row_step_size()

        # Build the list of pixel values
        pixels = numpy.zeros((PuzzleTester.PIECE_WIDTH, PuzzleTester.NUMB_PIXEL_DIMENSIONS))
        for i in range(0, PuzzleTester.PIECE_WIDTH):
            pixel_start = start_value + i * pixel_offset
            for j in range(0, PuzzleTester.NUMB_PIXEL_DIMENSIONS):
                pixels[i, j] = pixel_start + j

        # Return the result either reversed or not.
        if reverse_list:
            return pixels[::-1]
        else:
            return pixels

    @staticmethod
    def row_to_row_step_size():
        """
        Row to Row Step Size

        For a given pixel's given dimension, this function returns the number of dimensions between this pixel and
        the matching pixel exactly one row below.

        It is essentially the number of dimensions multiplied by the width of the original image (in pixels).

        Returns (int): Offset in dimensions.
        """
        step_size = PuzzleTester.NUMB_PIXEL_DIMENSIONS * PuzzleTester.PIECE_WIDTH * math.sqrt(PuzzleTester.NUMB_PUZZLE_PIECES)
        return int(step_size)

    @staticmethod
    def piece_to_piece_step_size():
        """
        Piece to Piece Step Size

        For a given pixel's given dimension, this function returns the number of dimensions between this pixel and
        the matching pixel exactly one puzzle piece away.

        It is essentially the number of dimensions multiplied by the width of a puzzle piece (in pixels).

        Returns (int): Offset in dimensions.
        """
        return PuzzleTester.NUMB_PIXEL_DIMENSIONS * PuzzleTester.PIECE_WIDTH

    @staticmethod
    def build_dummy_puzzle():
        """
        Dummy Puzzle Builder

        Using an image on the disk, this function builds a dummy puzzle using a Numpy array that is manually
        loaded with sequentially increasing pixel values.

        Returns (Puzzle): A puzzle where each pixel dimension from left to right sequentially increases by
        one.
        """

        # Create a puzzle whose image data will be overridden
        puzzle = Puzzle(0, PuzzleTester.TEST_IMAGE_FILENAME)

        # Define the puzzle side
        piece_width = PuzzleTester.PIECE_WIDTH
        numb_pieces = PuzzleTester.NUMB_PUZZLE_PIECES
        numb_dim = PuzzleTester.NUMB_PIXEL_DIMENSIONS

        # Define the array
        dummy_img = numpy.zeros((piece_width * math.sqrt(numb_pieces), piece_width * math.sqrt(numb_pieces), numb_dim))
        # populate the array
        val = PuzzleTester.TEST_ARRAY_FIRST_PIXEL_VALUE
        img_shape = dummy_img.shape
        for row in range(0, img_shape[0]):
            for col in range(0, img_shape[1]):
                for dim in range(0, img_shape[2]):
                    dummy_img[row, col, dim] = val
                    val += 1

        # Overwrite the image parameters
        puzzle._img = dummy_img
        puzzle._img_LAB = dummy_img
        puzzle._img_width = img_shape[1]
        puzzle._img_height = img_shape[0]
        puzzle._piece_width = PuzzleTester.PIECE_WIDTH
        puzzle._grid_size = (math.sqrt(PuzzleTester.NUMB_PUZZLE_PIECES), math.sqrt(PuzzleTester.NUMB_PUZZLE_PIECES))

        # Remake the puzzle pieces
        puzzle.make_pieces()
        return puzzle


if __name__ == "__main__":
    myPuzzle = Puzzle(0, ".\images\muffins_300x200.jpg")
    x = 1
