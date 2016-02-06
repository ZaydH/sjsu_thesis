import Tkinter
import tkFileDialog


def get_image_filename():
    """
    Provides a file exporer for selecting the image to turn
    into a puzzle.

    :return: Path to the image file to turn into a puzzle.
    """

    root = Tkinter.Tk()
    root.wm_title("Image Browser")
    w = Tkinter.Label(root, text="Please choose a .pages file to convert.")
    file_options = {}
    file_options['defaultextension'] = '.bmp'
    file_options['filetypes'] = [ ('Bitmap Files', '.bmp') ]
    file_options['title'] = 'Image File Browser'
    # Open the filename
    filename = tkFileDialog.askopenfilename(**file_options) ## **file_options means keyword based arguments
    # Return the image name.
    return filename



if __name__== '__main__':
    filename = get_image_filename()
    print "1"
    print filename
