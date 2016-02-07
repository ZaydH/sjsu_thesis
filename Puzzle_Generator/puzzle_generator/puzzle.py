# Tkinter use for a file dialog box.
import Tkinter
import tkFileDialog
from PIL import Image
import math
from puzzle_piece import PuzzlePiece
import pickle
import random

class Puzzle:

    DEFAULT_PATH = "./images/"
    PRINT_DEBUG_MESSAGES = True

    @staticmethod
    def pickle_export(obj, filename):
        """
        Export a specified object to a pickle file.
        :param obj:
        :param filename:
        """
        f = open(filename, 'w')
        pickle.dump(obj, f)
        f.close()

    @staticmethod
    def pickle_import(obj, filename):
        f = open(filename, 'r')
        obj = pickle.load(f)
        f.close()

    def __init__(self):
        """
            Initalizes the base memory of an image including set a default file
            name and the like.
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

    def select_puzzle_image(self):
        """
        Provides a file dialog for selecting the image to turn
        into a puzzle.
        """
        root = Tkinter.Tk()
        root.wm_title("Image Browser")
        w = Tkinter.Label(root, text="Please choose a .pages file to convert.")
        # Only display bitmap images in the browser.
        file_options = {}
        file_options['defaultextension'] = '.bmp'
        file_options['filetypes'] = [ ('Bitmap Files', '.bmp') ]
        file_options['title'] = 'Image File Browser'
        if __name__== '__main__':
            self._filename = Puzzle.DEFAULT_PATH + "duck.bmp"
        else:
            # Store the selected file name
            self._filename = tkFileDialog.askopenfilename(**file_options) # **file_option mean keyword based arguments

    def open_image(self):
        """
        Opens the specified image filename and stores it with the puzzle.
        """
        self._pil_img = Image.open(self._filename)
        self._image_width, self._image_height = self._pil_img.size

    def convert_to_pieces(self, x_piece_count, y_piece_count):
        """
        Given a puzzle, this function turns the puzzle into a set of pieces.
        **Note:** When creating the pieces, some of the source image may need to be discarded
        if the image size is not evenly divisible by the number of pieces specified
        as parameters to this function.
        :param x_piece_count: int   Number of pieces along the width of the puzzle
        :param y_piece_count: int   Number of pieces along the height of the puzzle
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
        if Puzzle.DEFAULT_PATH and ignored_pixels > 0:
            print "NOTE: %d pixels were not included in the puzzle." % {ignored_pixels}

        # Only take the center of the images and exclude the ignored pixels
        x_offset = (self._image_width - self._x_piece_count * self._piece_width) // 2
        y_offset = (self._image_height - self._y_piece_count * self._piece_width) // 2

        # Build the pixels
        for x in range(0, x_piece_count):
            x_start = x_offset + x * self._piece_width
            for y in range(0, y_piece_count):
                y_start = y_offset + y * self._piece_width
                self._pieces[x][y] = PuzzlePiece(self._piece_width, self._pil_img, x_start, y_start)

    def shuffle_pieces(self):
        """
        Perform a Fisher-Yates shuffle of the puzzle's pieces and then
        rotate the pieces.
        """
        numb_pieces = self._x_piece_count * self._y_piece_count
        for i in range(numb_pieces - 1, 0, -1):
            # Select a random piece.
            random_index = random.randint(0, i)
            # If the indexes are different, swap the pieces.
            if i != random_index:
                last_piece = self._pieces[i % self._x_piece_count][i // self._y_piece_count]
                random_piece = self._pieces[random_index % self._x_piece_count][random_index // self._y_piece_count]
                # Rotate the selected piece
                if PuzzlePiece.rotation_enabled:
                    random_piece.randomize_rotation()

                self._pieces[i % self._x_piece_count][i // self._y_piece_count] = random_piece
                self._pieces[random_index % self._x_piece_count][random_index // self._y_piece_count] = last_piece

        # Since the [0, 0] piece is not shuffled, then randomize its rotation
        if PuzzlePiece.rotation_enabled:
            self._pieces[0][0].randomize_rotation()

    def output_puzzle(self, filename):
        """
        For a specified puzzle, it writes a bitmap image of the puzzle
        to the specified filename.
        :param filename: String  Name of output file
        """
        puzzle_width = self._piece_width * self._x_piece_count
        puzzle_height = self._piece_width * self._y_piece_count

        # Create the array containing the pixels.
        pixels = Image.new("RGB", (puzzle_width, puzzle_height), "black")

        # Iterate through the pixels
        for x_piece in range(0, self._x_piece_count):
            x_offset = x_piece * self._piece_width
            for y_piece in range(0, self._y_piece_count):
                y_offset = y_piece * self._piece_width
                # Do pixel by pixel copy
                for x in range(0, self._piece_width):
                    for y in range(0, self._piece_width):
                        # Get the pixels from the selected piece
                        pixels.putpixel((x + x_offset, y + y_offset), self._pieces[x_piece][y_piece].getpixel(x, y))

        # Output the image file.
        pixels.save(filename)


    def transpose_image(self):
        """
        Creates a new transposed image Image object.
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
        transpose_img.save(Puzzle.DEFAULT_PATH + "transpose.bmp")

if __name__== '__main__':
    test_image = Puzzle()
    test_image.select_puzzle_image()
    test_image.open_image()
    #test_image.transpose_image()
    test_image.convert_to_pieces(10, 10)
    test_image.shuffle_pieces()
    #Puzzle.pickle_export(test_image, 'test_image.p')
    test_image.output_puzzle(Puzzle.DEFAULT_PATH + "Puzzle_Export.bmp")


    print "Run complete."