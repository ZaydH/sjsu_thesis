# Tkinter use for a file dialog box.
import Tkinter, tkFileDialog
from PIL import Image

class Puzzle:

    DEFAULT_PATH = "C:/Users/Zayd/Desktop/"

    def __init__(self):
        """
            Initalizes the base memory of an image including set a default file
            name and the like.
        """
        self._filename = "<Not Specified>"
        # Internal Pillow Image object.
        self._pil_img = None
        self._width = -1
        self._height = -1

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
            self._filename = zsh_image.DEFAULT_PATH + "duck.bmp"
        else:
            # Store the selected file name
            self._filename = tkFileDialog.askopenfilename(**file_options) # **file_option mean keyword based arguments

    def open_image(self):
        self._pil_img = Image.open(self._filename)
        (self._width, self._height) = self._pil_img.size

    def transpose_image(self):
        # Create the new transposed image Image object.
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