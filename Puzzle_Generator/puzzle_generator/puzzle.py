# Tkinter use for a file dialog box.
import Tkinter
import tkFileDialog
from PIL import Image
import math

class Puzzle:

    DEFAULT_PATH = "C:/Users/Zayd/Desktop/"
    MINIMUM_PIECE_WIDTH = 10

    def __init__(self):
        """
            Initalizes the base memory of an image including set a default file
            name and the like.
        """
        self._filename = "<Not Specified>"
        # Internal Pillow Image object.
        self._pil_img = None
        # Set the details on the image width
        self._width = 0
        self._height = 0
        # Initialize the puzzle information.
        self._pieces = []
        self._x_piece_count = 0
        self._y_piece_count = 0

    def set_filename(self):
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
        self._width, self._height = self._pil_img.size

    def convert_to_pieces(self, x_piece_count, y_piece_count):

        # Verify a valid pixel count.
        numb_pixels = self._width * self._height
        numb_pieces = x_piece_count * y_piece_count
        if numb_pixels < numb_pieces:
            raise ValueError("The number of pieces is more than the number of pixes. This is not allowed.")

        # Calculate the piece width based off the
        piece_width = math.min(self._width / x_piece_count, self._height / y_piece_count)
        if Puzzle.MINIMUM_PIECE_WIDTH < piece_width:
            raise ValueError("For the specified piece counts, the piece size is less than the minimum. of %." % {Puzzle.MINIMUM_PIECE_WIDTH})


    def shuffle_pieces(self):
        pass


    def transpose_image(self):
        """
        Creates a new transposed image Image object.
        """

        # This has image height and width transposed.
        transpose_img = Image.new("RGB", (self._height, self._width), "white")

        # Convert both images to pixels.

        # Iterate through all pixels in the original image
        for w in range(0, self._width):
            for h in range(0, self._height):
                # Get the pixel's rgb signature from the source image
                pixel = self._pil_img.getpixel((w,h))
                transpose_img.putpixel((h,w), pixel)

        # Save the image to a file.
        transpose_img.save(zsh_image.DEFAULT_PATH + "transpose.bmp")

if __name__== '__main__':
    test_image = Puzzle()
    test_image.set_filename()
    test_image.open_image()
    test_image.transpose_image()
    print "Run complete."