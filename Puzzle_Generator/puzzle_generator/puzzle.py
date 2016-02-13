"""Jigsaw Puzzle Problem Object

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""


# Tkinter use for a file dialog box.
import Tkinter
import tkFileDialog
from PIL import Image
from puzzle_piece import PuzzlePiece
import pickle
import random

class Puzzle:
    """
    """

    DEFAULT_IMAGE_PATH = "./images/"
    print_debug_messages = True

    export_with_border = False
    border_width = 3
    border_outer_stripe_width = 1


    # @staticmethod
    # def pickle_import(obj, filename):
    #     f = open(filename, 'r')
    #     obj = pickle.load(f)
    #     f.close

    @staticmethod
    def make_puzzle_from_pieces(pieces):
        output_puzzle = Puzzle()
        # Build the piece counts
        output_puzzle._x_piece_count = len(pieces)
        output_puzzle._y_piece_count = len(pieces[0])

        # Store the piece width
        (output_puzzle._piece_width, _) = pieces[0][0].size

        # Store the list of pieces.
        output_puzzle._pieces = pieces

        return output_puzzle


    def pickle_export(self, pickle_filename):
        """Puzzle Pickle Exporter

        Exports the puzzle object to pickle for serialization.

        Args:
            pickle_filename (str): String Filename to write the pickle object to.

        """
        try:
            f = open(pickle_filename, 'w')
            pickle.dump(self, f)
            f.close()
        except:
            raise IOError("Unable to write the pickle file to location \"%s\"." % (pickle_filename))

    def __init__(self, image_filename=None):
        """ Puzzle Constructor

        Constructor that will optionally load an image into the puzzle as well.

        Args:
            image_filename (str): Name of the image to load into the puzzle. Defaults to None.

        Returns:
            Puzzle Object

        """
        self._filename = "<Not Specified>"
        # Internal Pillow Image object.
        self._pil_img = None
        # Set the details on the image width
        self._image_width = 0
        self._image_height = 0
        # Initialize the puzzle information.
        self._pieces = []
        self._piece_width = 0
        self._x_piece_count = 0
        self._y_piece_count = 0
        if(image_filename is not None):
            self.load_puzzle_image(image_filename)

    def load_puzzle_image(self, filename=None):
        """Puzzle Image Loader

        Sets the puzzle image file.  If the user specifies a puzzle image filename, then the function will load it.
        If no image file name is speficified, the functiuon will open a file dialog box for the user to explore
        to the desired file.

        Args:
            filename (str): If no filename is specified, then a file dialog is shown.  Defaults to None.

        """

        # Check if a filename was specified.  If it was, store it and return.
        if filename is not None:
            self._filename = filename
            self._open_image()
            return

        # If no filename is specified, then open a file dialog.
        root = Tkinter.Tk()
        root.wm_title("Image Browser")
        w = Tkinter.Label(root, text="Please choose a .pages file to convert.")
        # Only display bitmap images in the browser.
        file_options = {}
        file_options['defaultextension'] = '.bmp'
        file_options['filetypes'] = [('Bitmap Files', '.bmp')]
        file_options['title'] = 'Image File Browser'
        # Store the selected file name
        self._filename = tkFileDialog.askopenfilename(**file_options) # **file_option mean keyword based arguments
        self._open_image()


    def _open_image(self):
        """
        Loads the specified puzzle image into memory.  It also stores information on the puzzle
        dimensions (e.g. width, height) into the puzzle object.
        """
        try:
            self._pil_img = Image.open(self._filename)
            self._image_width, self._image_height = self._pil_img.size
        except:
            raise IOError("Unable to load the image at the specified location \"%s\"." % (self._filename))

    def convert_to_pieces(self, x_piece_count, y_piece_count):
        """Puzzle Generator

        Given a puzzle, this function turns the puzzle into a set of pieces.
        **Note:** When creating the pieces, some of the source image may need to be discarded
        if the image size is not evenly divisible by the number of pieces specified
        as parameters to this function.

        Args:
            x_piece_count (int): Number of pieces along the width of the puzzle
            y_piece_count (int): Number of pieces along the height of the puzzle

        """
        # Verify a valid pixel count.
        numb_pixels = self._image_width * self._image_height
        numb_pieces = x_piece_count * y_piece_count
        self._x_piece_count = x_piece_count
        self._y_piece_count = y_piece_count
        if numb_pixels < numb_pieces:
            raise ValueError("The number of pieces is more than the number of pixes. This is not allowed.")

        # Calculate the piece width based off the
        self._piece_width = min(self._image_width // x_piece_count, self._image_height // y_piece_count)
        # noinspection PyUnusedLocal
        self._pieces = [[None for y in range(0, y_piece_count)] for x in range(0, x_piece_count)]

        # Calculate ignored pixel count for debugging purposes.
        ignored_pixels = numb_pixels - (self._piece_width * self._piece_width * numb_pieces)
        if Puzzle.DEFAULT_IMAGE_PATH and ignored_pixels > 0:
            print "NOTE: %d pixels were not included in the puzzle." % (ignored_pixels)

        # Only take the center of the images and exclude the ignored pixels
        x_offset = (self._image_width - self._x_piece_count * self._piece_width) // 2
        y_offset = (self._image_height - self._y_piece_count * self._piece_width) // 2

        # Build the pixels
        for x in range(0, x_piece_count):
            x_start = x_offset + x * self._piece_width
            for y in range(0, y_piece_count):
                y_start = y_offset + y * self._piece_width
                actual_location = (x, y)  # Location of the piece in the original board.
                self._pieces[x][y] = PuzzlePiece(self._piece_width, actual_location, self._pil_img, x_start, y_start)

    def shuffle_pieces(self):
        """Puzzle Piece Shuffler

        Perform a Fisher-Yates shuffle of the puzzle's pieces.  If rotation of Puzzle Pieces
        is enabled, then this function will also perform piece rotation.

        """
        numb_pieces = self._x_piece_count * self._y_piece_count
        for i in range(numb_pieces - 1, 0, -1):
            # Select a random piece.
            random_index = random.randint(0, i)
            # If the indexes are different, swap the pieces.
            if i != random_index:
                last_piece = self._pieces[i % self._x_piece_count][i // self._x_piece_count]
                random_piece = self._pieces[random_index % self._x_piece_count][random_index // self._x_piece_count]
                # Rotate the selected piece
                if PuzzlePiece.rotation_enabled:
                    random_piece.randomize_rotation()

                self._pieces[i % self._x_piece_count][i // self._x_piece_count] = random_piece
                self._pieces[random_index % self._x_piece_count][random_index // self._x_piece_count] = last_piece

        # Since the [0, 0] piece is not shuffled, then randomize its rotation
        if PuzzlePiece.rotation_enabled:
            self._pieces[0][0].randomize_rotation()

    def export_puzzle(self, filename):
        """Puzzle Image Exporter

        For a specified puzzle, it writes an image file to the specified filename.

        Note:
            Image file format is dependent on the file extension in the specified filename.

        Args:
            filename (str): File path of the image file name

        """
        puzzle_width = self._piece_width * self._x_piece_count
        puzzle_height = self._piece_width * self._y_piece_count
        # Widen the picture if it should have a border.
        if Puzzle.export_with_border:
            puzzle_width += (self._x_piece_count - 1) * Puzzle.border_width
            puzzle_height += (self._y_piece_count - 1) * Puzzle.border_width

        # Create the array containing the pixels.
        pixels = Image.new("RGB", (puzzle_width, puzzle_height), "black")

        # Iterate through the pixels
        for x_piece in range(0, self._x_piece_count):
            start_x = x_piece * self._piece_width
            # Add the cell border if applicable
            if Puzzle.export_with_border:
                start_x += x_piece * Puzzle.border_width

            for y_piece in range(0, self._y_piece_count):
                start_y = y_piece * self._piece_width
                # Add the cell border if applicable
                if Puzzle.export_with_border:
                    start_y += y_piece * Puzzle.border_width

                # Paste the image from the piece.
                box = (start_x, start_y, start_x + self._piece_width, start_y + self._piece_width)
                piece_image = self._pieces[x_piece][y_piece].get_image()
                assert(piece_image.size == (self._piece_width, self._piece_width))  # Verify the size
                pixels.paste(piece_image, box)

        # Add a white border
        if Puzzle.export_with_border:
            # Shorten variable names for readability
            border_width = Puzzle.border_width
            outer_strip_width = Puzzle.border_outer_stripe_width
            # create row borders one at a time
            for row in range(1, self._x_piece_count):  # Skip the first and last row
                # Define the box for the border.
                top_left_x = 0
                top_left_y = (row - 1) * border_width + row * self._piece_width + outer_strip_width
                bottom_right_x = puzzle_width
                bottom_right_y = top_left_y + (border_width - 2 * outer_strip_width)
                # Create the row border via a white box.
                box = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
                pixels.paste("white", box)
            for col in range(1, self._y_piece_count):  # Skip the first and last row
                # Define the box for the border.
                top_left_x = (col - 1) * border_width + col * self._piece_width + outer_strip_width
                top_left_y = 0
                bottom_right_x = top_left_x + (border_width - 2 * outer_strip_width)
                bottom_right_y = puzzle_height
                # Create the row border via a white box.
                box = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
                pixels.paste("white", box)

        # Output the image file.
        pixels.save(filename)

    def get_pieces(self):
        return self._pieces

    def _transpose_image(self):
        """ Image Transposer

        Transposes an image by doing a pixel by pixel copy of the puzzle's image.  It then outputs the image.

        Note:
            This function was originally written to get experience with Pillow and as a basic debug framework.
            Its proper functionality is not guaranteed for all revisions.

        """
        # This has image height and width transposed.
        transpose_img = Image.new("RGB", (self._image_height, self._image_width), "white")

        # Convert both images to pixels.

        # Iterate through all pixels in the original image
        for w in range(0, self._image_width):
            for h in range(0, self._image_height):
                # Get the pixel's rgb signature from the source image
                pixel = self._pil_img.getpixel((w, h))
                transpose_img.putpixel((h, w), pixel)

        # Save the image to a file.
        transpose_img.save(Puzzle.DEFAULT_IMAGE_PATH + "transpose.bmp")

if __name__ == '__main__':
    # Take some images and shuffle then export them.
    puzzles = [("duck.bmp", (10, 10)), ("two_faced_cat.jpg", (20, 10))]
    for puzzle_info in puzzles:
        # Extract the information on the images
        file = puzzle_info[0]
        (x_count, y_count) = puzzle_info[1]
        # Build a test puzzle
        test_puzzle = Puzzle(Puzzle.DEFAULT_IMAGE_PATH + file)
        # test_puzzle.set_puzzle_image(Puzzle.DEFAULT_IMAGE_PATH + file )
        # test_puzzle.open_image()
        test_puzzle.convert_to_pieces(x_count, y_count)
        test_puzzle.shuffle_pieces()
        test_puzzle.export_puzzle(Puzzle.DEFAULT_IMAGE_PATH + "puzzle_" + file)

    # filename = 'test_puzzle.pk'
    # # Puzzle.pickle_export(test_puzzle, filename)
    # f = open(filename, 'r')
    # test_puzzle = pickle.load(f)
    # f.close()
    # test_puzzle.export_puzzle(Puzzle.DEFAULT_IMAGE_PATH + "_Export.bmp")

    # Debug message to let me know the run was completed.
    if Puzzle.print_debug_messages:
        print "Run complete."
