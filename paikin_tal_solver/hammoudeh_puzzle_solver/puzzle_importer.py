"""Jigsaw Puzzle Problem Object

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""
import os

# from PIL import Image
import math
import numpy
import cv2  # OpenCV
import pickle


class Puzzle(object):
    """
    """
    print_debug_messages = True

    DEFAULT_PIECE_WIDTH = 28  # Width of a puzzle in pixels

    export_with_border = True
    border_width = 3
    border_outer_stripe_width = 1

    def __init__(self, id_number, image_filename):
        """Puzzle Constructor

        Constructor that will optionally load an image into the puzzle as well.

        Args:
            id_number (int): ID number for the image.  It is used for multiple image puzzles.
            image_filename (str): File path of the image to load

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
        (self._img_width, self._img_height, _) = self._img.shape

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
        grid_x_size = math.floor(self._img_width / self.piece_width)
        grid_y_size = math.floor(self._img_height / self.piece_width)
        if grid_x_size == 0 or grid_y_size == 0:
            raise ValueError("Image size is too small for the image.  Check your setup")

        # Store the grid size.
        self._grid_size = (grid_x_size, grid_y_size)

        # Store the original width and height and recalulate the new width and height.
        original_width = self._img_width
        original_height = self._img_height
        self._img_width = grid_x_size * self.piece_width
        self._img_height = grid_y_size * self.piece_width

        # Shave off the edge of the image.
        upper_left = ((original_width - self._img_width) / 2, (original_height - self._img_height) / 2)
        self._img = Puzzle.extract_subimage(self._img, upper_left, (self._img_width, self._img_height))
        #Puzzle.display_image(img2)

    @property
    def piece_width(self):
        """
        Gets the size of a puzzle piece.

        Returns (int): Height/width of each piece in pixels.

        """
        return self._piece_width

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

        #
        # def pickle_export(self, pickle_filename):
        #     """Puzzle Pickle Exporter
        #
        #     Exports the puzzle object to pickle for serialization.
        #
        #     Args:
        #         pickle_filename (str): Name of the pickle output file.
        #
        #     """
        #     try:
        #         # Configure the puzzle's image for export.
        #         self._pil_img_pickle = {'pixels': self._img.tobytes(),
        #                                 'size': self._img.size,
        #                                 'mode': self._img.mode}
        #
        #         # Configure the puzzle pieces for export.
        #         for x in range(0, self._grid_x_size):
        #             for y in range(0, self._grid_y_size):
        #                 self._pieces[x][y].pickle_export_configure()
        #
        #         f = open(pickle_filename, 'w')
        #         pickle.dump(self, f)
        #         f.close()
        #     except:
        #         raise IOError("Unable to write the pickle file to location \"%s\"." % pickle_filename)
        #
        # @staticmethod
        # def pickle_import(filename):
        #     """Puzzle Pickle Importer
        #
        #     When importing a puzzle from Pickle, this function must be used.  It overcomes some of the limitations in the
        #     Pillow library around pickling.  It will reconstruct the associated image information that is not correctly
        #     configured by Pickle.
        #
        #     This function is essentially a Factory that does the object creation and configuring given a specified
        #     pickle file.
        #
        #     Args:
        #         filename (str): Name of the pickle input file.
        #
        #     Returns (Puzzle): A reconstructed Puzzle object from a pickle file.
        #
        #     """
        #     f = open(filename, 'r')
        #     obj = pickle.load(f)
        #     f.close()
        #
        #     # Reinitialize the Image from the bytes representation.
        #     # noinspection PyProtectedMember
        #     obj._pil_img = Image.frombytes(obj._pil_img_pickle['mode'],
        #                                    obj._pil_img_pickle['size'],
        #                                    obj._pil_img_pickle['pixels'])
        #
        #     # Configure the puzzle pieces for export.
        #     for x in range(0, obj.grid_x_size):
        #         for y in range(0, obj.grid_y_size):
        #             # noinspection PyProtectedMember
        #             obj._pieces[x][y].pickle_import_configure()
        #
        #     # Return the imported Puzzle.
        #     return obj
        #
        # def convert_to_pieces(self, grid_x_size, grid_y_size):
        #     """Puzzle Generator
        #
        #     Given a puzzle, this function turns the puzzle into a set of pieces.
        #     **Note:** When creating the pieces, some of the source image may need to be discarded
        #     if the image size is not evenly divisible by the number of pieces specified
        #     as parameters to this function.
        #
        #     Args:
        #         grid_x_size (int): Number of pieces along the width of the puzzle
        #         grid_y_size (int): Number of pieces along the height of the puzzle
        #
        #     """
        #     # Verify a valid pixel count.
        #     numb_pixels = self._image_width * self._image_height
        #     numb_pieces = grid_x_size * grid_y_size
        #     self._grid_x_size = grid_x_size
        #     self._grid_y_size = grid_y_size
        #     if numb_pixels < numb_pieces:
        #         raise ValueError("The number of pieces is more than the number of pixes. This is not allowed.")
        #
        #     # Calculate the piece width based off the
        #     self._piece_width = min(self._image_width // grid_x_size, self._image_height // grid_y_size)
        #     # noinspection PyUnusedLocal
        #     self._pieces = [[None for y in range(0, grid_y_size)] for x in range(0, grid_x_size)]
        #
        #     # Calculate ignored pixel count for debugging purposes.
        #     ignored_pixels = numb_pixels - (self._piece_width * self._piece_width * numb_pieces)
        #     if Puzzle.DEFAULT_IMAGE_PATH and ignored_pixels > 0:
        #         print "NOTE: %d pixels were not included in the puzzle." % ignored_pixels
        #
        #     # Only take the center of the images and exclude the ignored pixels
        #     x_offset = (self._image_width - self._grid_x_size * self._piece_width) // 2
        #     y_offset = (self._image_height - self._grid_y_size * self._piece_width) // 2
        #
        #     # Build the pixels
        #     for x in range(0, grid_x_size):
        #         x_start = x_offset + x * self._piece_width
        #         for y in range(0, grid_y_size):
        #             y_start = y_offset + y * self._piece_width
        #             actual_location = (x, y)  # Location of the piece in the original board.
        #             self._pieces[x][y] = PuzzlePiece(self._piece_width, actual_location, self._pil_img, x_start, y_start)
        #
        # def export_puzzle(self, filename):
        #     """Puzzle Image Exporter
        #
        #     For a specified puzzle, it writes an image file to the specified filename.
        #
        #     Note:
        #         Image file format is dependent on the file extension in the specified filename.
        #
        #     Args:
        #         filename (str): File path of the image file name
        #
        #     """
        #     puzzle_width = self._piece_width * self._grid_x_size
        #     puzzle_height = self._piece_width * self._grid_y_size
        #     # Widen the picture if it should have a border.
        #     if Puzzle.export_with_border:
        #         puzzle_width += (self._grid_x_size - 1) * Puzzle.border_width
        #         puzzle_height += (self._grid_y_size - 1) * Puzzle.border_width
        #
        #     # Create the array containing the pixels.
        #     pixels = Image.new("RGB", (puzzle_width, puzzle_height), "black")
        #
        #     # Iterate through the pixels
        #     for x_piece in range(0, self._grid_x_size):
        #         start_x = x_piece * self._piece_width
        #         # Add the cell border if applicable
        #         if Puzzle.export_with_border:
        #             start_x += x_piece * Puzzle.border_width
        #
        #         for y_piece in range(0, self._grid_y_size):
        #             start_y = y_piece * self._piece_width
        #             # Add the cell border if applicable
        #             if Puzzle.export_with_border:
        #                 start_y += y_piece * Puzzle.border_width
        #
        #             # Get the image for thie specified piece.
        #             piece_image = self._pieces[x_piece][y_piece].image
        #             assert(piece_image.size == (self._piece_width, self._piece_width))  # Verify the size
        #             # Define the box where the piece will be placed
        #             box = (start_x, start_y, start_x + self._piece_width, start_y + self._piece_width)
        #             # Paste the image from the piece.
        #             pixels.paste(piece_image, box)
        #
        #     # Add a white border
        #     if Puzzle.export_with_border:
        #         # Shorten variable names for readability
        #         border_width = Puzzle.border_width
        #         outer_strip_width = Puzzle.border_outer_stripe_width
        #         # create row borders one at a time
        #         for row in range(1, self._grid_y_size):  # Skip the first and last row
        #             # Define the box for the border.
        #             top_left_x = 0
        #             top_left_y = (row - 1) * border_width + row * self._piece_width + outer_strip_width
        #             bottom_right_x = puzzle_width
        #             bottom_right_y = top_left_y + (border_width - 2 * outer_strip_width)
        #             # Create the row border via a white box.
        #             box = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        #             pixels.paste("white", box)
        #         # Create the column white separators
        #         for col in range(1, self._grid_x_size):  # Skip the first and last row
        #             # Define the box for the border.
        #             top_left_x = (col - 1) * border_width + col * self._piece_width + outer_strip_width
        #             top_left_y = 0
        #             bottom_right_x = top_left_x + (border_width - 2 * outer_strip_width)
        #             bottom_right_y = puzzle_height
        #             # Create the row border via a white box.
        #             box = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        #             pixels.paste("white", box)
        #
        #     # Output the image file.
        #     pixels.save(filename)
        #
        # @property
        # def pieces(self):
        #     return self._pieces
        #
        # @property
        # def grid_x_size(self):
        #     return self._grid_x_size
        #
        # @property
        # def grid_y_size(self):
        #     return self._grid_y_size
        #
        #


if __name__ == "__main__":
    myPuzz = Puzzle(0, ".\images\muffins_300x200.jpg")
    x = 1
