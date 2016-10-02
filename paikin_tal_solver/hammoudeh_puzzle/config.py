import logging
import random

DEFAULT_PIECE_WIDTH = 28

PERFORM_ASSERT_CHECKS = True

IMAGE_DIRECTORY = '.\\images\\'

RESULTS_FILE = '.\\results.csv'

NUMBER_805_PIECE_PUZZLES = 20
MINIMUM_805_PIECE_IMAGE_NUMBER = 1
MAXIMUM_805_PIECE_IMAGE_NUMBER = MINIMUM_805_PIECE_IMAGE_NUMBER + NUMBER_805_PIECE_PUZZLES - 1
_DIRECTORY_805_PIECE_IMAGES = "805_pieces\\"
_IMAGE_FILE_EXTENSION_805_PIECE_IMAGES = ".jpg"


def setup_logging(filename="solver_driver.log", log_level=logging.DEBUG):
    """
    Configures the logger for process tasks

    Args:
        filename (str): Name of the log file to be generated
        log_level (int): Logger level (e.g. DEBUG, INFO, WARNING)

    """
    data_format = '%m/%d/%Y %I:%M:%S %p'  # Example Time Format - 12/12/2010 11:46:36 AM
    # noinspection SpellCheckingInspection
    logging.basicConfig(filename=filename, level=log_level, format='%(asctime)s -- %(message)s', datefmt=data_format)

    # Also print to stdout
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    logging.info("*********************************** New Run Beginning ***********************************")


def add_image_folder_path(image_filenames):
    """
    Prepends the image file path to the input image file names.

    Args:
        image_filenames (list[str]): File names for the input images.

    Returns (list[str]):
        File names with the image directory path prepended.
    """
    return [IMAGE_DIRECTORY + img_file for img_file in image_filenames]


def get_random_805_piece_image():
    """
    Gets a random image file name from the 805 image dataset from Ben Gurion University.

    Returns (str): File path to a random 805 piece image.
    """
    rand_img_numb = random.randint(MINIMUM_805_PIECE_IMAGE_NUMBER, MAXIMUM_805_PIECE_IMAGE_NUMBER)
    return build_805_piece_filename(rand_img_numb)


def build_805_piece_filename(image_number):
    """
    Creates the image name for the 805 piece image

    Args:
        image_number (int): 805 piece image number.

    Returns (str): Name of the images for the 805 piece image.
    """
    if not isinstance(image_number, int):
        raise ValueError("Image number must be an integer.")

    if image_number < MINIMUM_805_PIECE_IMAGE_NUMBER or image_number > MAXIMUM_805_PIECE_IMAGE_NUMBER:
        raise ValueError("Invalid 805 piece image number")

    return _DIRECTORY_805_PIECE_IMAGES + str(image_number) + _IMAGE_FILE_EXTENSION_805_PIECE_IMAGES
