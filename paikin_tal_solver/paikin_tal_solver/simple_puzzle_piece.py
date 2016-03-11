import copy

from enum import Enum



class SimplePuzzlePiece(object):

    # When running debug tests, extra more time consuming error checking is done.
    RUN_DEBUG_TESTS = True

    # Number of dimensions in the LAB colorspace.
    NUMB_LAB_DIMENSIONS = 3

    # Define the minimum and maximum values for a LAB pixel.  Helps double check
    # and catch errors.
    PIXEL_LAB_MINIMUM_VALUE = 0
    PIXEL_LAB_MAXIMUM_VALUE = 255

    def __init__(self, id_numb, lab_pixel_data):
        """
        Constructor for the SimplePuzzlePiece.

        Args:
            id_numb (int):          Puzzle Piece ID number
            lab_pixel_data ([int]:  3D Matrix containing the pixel data.  Matrix size is (width x width x 3) where 3 is
                                    the number of dimensions in the LAB color space.

        """

        # Store the information on the piece.
        self._id = id_numb
        self._width = len(lab_pixel_data)
        self._pixel_data = copy.deepcopy(lab_pixel_data)

        # In debug mode, check
        if SimplePuzzlePiece.RUN_DEBUG_TESTS:
            self._check_piece_dimensions()

