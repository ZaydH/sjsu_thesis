"""Jigsaw Puzzle Problem Object

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""
import copy
import os
import math
# noinspection PyUnresolvedReferences
import numpy
import cv2  # OpenCV
from enum import Enum

from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePiece


class PuzzleType(Enum):
    """
    Type of the puzzle to solve.  Type 1 has no piece rotation while type 2 allows piece rotation.
    """

    type1 = 1
    type2 = 2


class Puzzle(object):
    """
    Puzzle Object represents a single Jigsaw Puzzle.  It can import a puzzle from an image file and
    create the puzzle pieces.
    """

    print_debug_messages = True

    # DEFAULT_PIECE_WIDTH = 28  # Width of a puzzle in pixels
    DEFAULT_PIECE_WIDTH = 25  # Width of a puzzle in pixels

    export_with_border = True
    border_width = 3
    border_outer_stripe_width = 1

    def __init__(self, id_number, image_filename=None):
        """Puzzle Constructor

        Constructor that will optionally load an image into the puzzle as well.

        Args:
            id_number (int): ID number for the image.  It is used for multiple image puzzles.
            image_filename (Optional str): File path of the image to load

        Returns:
            Puzzle Object

        """
        # Internal Pillow Image object.
        self._id = id_number
        self._img = None
        self._img_LAB = None

        # Initialize the puzzle information.
        self._grid_size = None
        self._piece_width = Puzzle.DEFAULT_PIECE_WIDTH
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
        puzzle_upper_left = ((original_width - self._img_width) / 2, (original_height - self._img_height) / 2)
        self._img = Puzzle.extract_subimage(self._img, puzzle_upper_left, (self._img_width, self._img_height))
        self._img_LAB = Puzzle.extract_subimage(self._img_LAB, puzzle_upper_left, (self._img_width, self._img_height))
        # Puzzle.display_image(self._img)
        #
        # for i in range(0, 5):
        #     for j in range(0, 5):
        #         pixel = self._img[i,j]
        #         x =1

        # Break the board into pieces.
        piece_size = (self.piece_width, self.piece_width)
        self._pieces = []  # Create an empty array to hold the puzzle pieces.
        for col in range(0, numb_cols):
            for row in range(0, numb_rows):
                piece_upper_left = (puzzle_upper_left[0] + col * piece_size[0],
                                    puzzle_upper_left[1] + row * piece_size[1])
                piece_img = Puzzle.extract_subimage(self._img_LAB, piece_upper_left, piece_size)

                # Create the puzzle piece and assign to the location.
                location = (row, col)
                self._pieces.append(PuzzlePiece(self._id, location, piece_img))

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
    def reconstruct_from_pieces(pieces, id_numb=-1):
        """
        Constructs a puzzle from a set of pieces.

        Args:
            pieces ([PuzzlePiece]): Set of puzzle pieces that comprise the puzzle.
            id_numb (Optional int): Identification number for the puzzle

        Returns (Puzzle):
        Puzzle constructed from the pieces.
        """

        if len(pieces) == 0:
            raise ValueError("Error: Each puzzle must have at least one piece.")

        # Create the puzzle to return.
        output_puzzle = Puzzle(id_numb)

        # Create a copy of the pieces.
        copy_pieces = copy.deepcopy(pieces)

        # Get the first piece and use its information
        first_piece = copy_pieces[0]
        output_puzzle._piece_width = first_piece.width
        first_piece_loc = first_piece.location

        # Find the min and max row and column.
        min_row = max_row = first_piece_loc[0]
        min_col = max_col = first_piece_loc[1]
        for i in range(0, len(copy_pieces)):

            # Verify all pieces are the same size
            if Puzzle.print_debug_messages:
                assert(output_puzzle.piece_width == copy_pieces[i].width)

            # Get the location of the piece
            temp_loc = copy_pieces[1]
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

        # Store the grid size
        output_puzzle._grid_size = (max_row - min_row + 1, max_col - min_col + 1)
        # Calculate the size of the image
        output_puzzle._img_width = output_puzzle._grid_size[1] * output_puzzle.piece_width
        output_puzzle._img_height = output_puzzle._grid_size[0] * output_puzzle.piece_width
        

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
        return img[upper_left[1]:img_end[1], upper_left[0]:img_end[0], :]

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
    def save_to_file(img, filename):
        """

        Args:
            img: OpenCV image in the form of a Numpy array
            filename (str): Filename and path to save the OpenCV image.

        """
        cv2.imwrite(filename, img)

if __name__ == "__main__":
    myPuzzle = Puzzle(0, ".\images\muffins_300x200.jpg")
    x = 1
