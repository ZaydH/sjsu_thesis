# Tkinter use for a file dialog box.
import Tkinter
import tkFileDialog
from PIL import Image
import math
from puzzle_piece import PuzzlePiece

class Puzzle:

    DEFAULT_PATH = "C:/Users/Zayd/Desktop/"
    PRINT_DEBUG_MESSAGES = True

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

        # Verify a valid pixel count.
        numb_pixels = self._image_width * self._image_height
        numb_pieces = x_piece_count * y_piece_count
        if numb_pixels < numb_pieces:
            raise ValueError("The number of pieces is more than the number of pixes. This is not allowed.")

        # Calculate the piece width based off the
        self._piece_width = math.min(self._image_width / x_piece_count, self._image_height / y_piece_count)
        # noinspection PyUnusedLocal
        self._pieces = [[None for y in range(0, y_piece_count)] for x in range(0, xpiece_count)]

        # Calculate ignored pixel count for debugging purposes.
        ignored_pixels = numb_pixels - (self._piece_width * x_piece_count * y_piece_count)
        if Puzzle.DEFAULT_PATH and ignored_pixels > 0:
            print "NOTE: %d pixels were not included in the puzzle." % {ignored_pixels}

        # Only take the center of the images and exclude the ignored pixels
        x_offset = (self._image_width - self._x_piece_count * self._piece_width) // 2
        y_offset = (self._image_height - self._y_piece_count * self._piece_height) // 2

        # Build the pixels
        loaded_image = self._pil_img.load()
        for x in range(0, x_piece_count):
            x_start = x_offset + x * self._piece_width
            for y in range(0, y_piece_count):
                y_start = y_offset + y * self._piece_width
                self._pieces[x][y] = PuzzlePiece(self._piece_width, loaded_image, x_start, y_start)


    def shuffle_pieces(self):
        pass


    def output_puzzle(self, filename):
        puzzle_width = self._piece_width * self._x_piece_count
        puzzle_height = self._piece_height * self._y_piece_count

        # Create the array containing the pixels.
        # noinspection PyUnusedLocal
        pixels = [[None for y in range(0, puzzle_height)] for x in range(0, puzzle_width)]

        # Iterate through the pixels
        for x_piece in range(0, self._x_piece_count):
            x_offset = x_piece * self._piece_width
            for y_piece in range(0, self._y_piece_count):
                y_offset = y_piece * self._piece_width
                # Do pixel by pixel copy
                for x in range(0, self._piece_width):
                    for y in range(0, self._piece_width):
                        # Get the pixels from the selected piece
                        pixels[x + x_offset][y + y_offset] = self._pieces[x_piece][y_piece].get_pixel(x,y)

        # Output the image file.
        self._create_image_file(pixels, filename)

    def _create_image_file(self, pixels, filename):
        """
        Writes a built image to a file.
        :param pixels:
        :param filename:
        :return:
        """
        # Extract
        width = len(pixels)
        try:
            height = len(pixels[0])
        except:
            raise ValueError("Pixel array must be two dimensional")
        # Starting building the output image
        output_img = Image.new("RGB", (width, height), "white")
        for x in range(0, self._image_width):
            for y in range(0, self._image_height):
                output_img.putpixel(pixels[x][y])
        # Save the output image.
        output_img.save(filename)

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
                pixel = self._pil_img.getpixel((w,h))
                transpose_img.putpixel((h,w), pixel)

        # Save the image to a file.
        transpose_img.save(Puzzle.DEFAULT_PATH + "transpose.bmp")

if __name__== '__main__':
    test_image = Puzzle()
    test_image.select_puzzle_image()
    test_image.open_image()
    #test_image.transpose_image()
    test_image.output_puzzle(Puzzle.DEFAULT_PATH + "Puzzle_Export.bmp")
    test_image.convert_to_pieces(10, 10)

    print "Run complete."