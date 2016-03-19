import random
import unittest
import math
import numpy

from hammoudeh_puzzle_solver.puzzle_importer import Puzzle
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePiece, PuzzlePieceSide


class PuzzleTester(unittest.TestCase):
    PIECE_WIDTH = 5
    NUMB_PUZZLE_PIECES = 9
    GRID_SIZE = (int(math.sqrt(NUMB_PUZZLE_PIECES)), int(math.sqrt(NUMB_PUZZLE_PIECES)))
    NUMB_PIXEL_DIMENSIONS = 3

    TEST_ARRAY_FIRST_PIXEL_VALUE = 0

    # Get the information on the test image
    TEST_IMAGE_FILENAME = ".\\test\\test.jpg"
    TEST_IMAGE_WIDTH = 300
    TEST_IMAGE_HEIGHT = 200

    def test_puzzle_creation(self):
        """
        Puzzle Import Parameter Checker

        Checks that for an image, the parameters of the image are successfully parsed.
        """
        test_img_id = 999

        # Create a dummy image for testing purposes
        Puzzle.DEFAULT_PIECE_WIDTH = PuzzleTester.PIECE_WIDTH
        puzzle = Puzzle(test_img_id, PuzzleTester.TEST_IMAGE_FILENAME)

        # Verify the test image id number
        assert(puzzle._id == test_img_id)

        # Verify the piece width information
        assert(puzzle._piece_width == PuzzleTester.PIECE_WIDTH)

        # Verify the image filename information
        assert(puzzle._filename == PuzzleTester.TEST_IMAGE_FILENAME)

        # Verify the image size info
        assert(puzzle._img_height == PuzzleTester.TEST_IMAGE_HEIGHT)
        assert(puzzle._img_width == PuzzleTester.TEST_IMAGE_WIDTH)

        # Verify the grid side is correct
        assert(puzzle._grid_size == (PuzzleTester.TEST_IMAGE_HEIGHT / PuzzleTester.PIECE_WIDTH,
                                     PuzzleTester.TEST_IMAGE_WIDTH / PuzzleTester.PIECE_WIDTH))

        # Verify the number of pieces are correct.
        assert(len(puzzle._pieces) == (PuzzleTester.TEST_IMAGE_HEIGHT / PuzzleTester.PIECE_WIDTH) *
               (PuzzleTester.TEST_IMAGE_WIDTH / PuzzleTester.PIECE_WIDTH))

        # Check information about the piece
        all_pieces = puzzle.pieces  # type: [PuzzlePiece]
        for piece in all_pieces:
            assert(piece.width == PuzzleTester.PIECE_WIDTH)

            assert(piece._orig_puzzle_id == test_img_id)
            assert(piece._assigned_puzzle_id is None)

            assert(piece.rotation is None)  # No rotation by default

            rand_loc = (random.randint(0, 9999), random.randint(0, 9999))
            piece.location = rand_loc
            assert(piece.location == rand_loc)
            piece._assign_to_original_location()
            assert(piece._orig_loc == piece.location)

    def test_puzzle_piece_maker(self):
        """
        Puzzle Piece Maker Checker

        Checks that puzzle pieces are made as expected.  It also checks the get puzzle piece row/column values.
        """

        # Build a known test puzzle.
        puzzle = PuzzleTester.build_dummy_puzzle()

        # Get the puzzle pieces
        pieces = puzzle.pieces
        for piece in pieces:
            orig_loc = piece._orig_loc
            upper_left_dim = orig_loc[0] * PuzzleTester.PIECE_WIDTH * PuzzleTester.row_to_row_step_size()
            upper_left_dim += orig_loc[1] * PuzzleTester.piece_to_piece_step_size()

            # Test the Extraction of row pixel values
            for row in range(0, PuzzleTester.PIECE_WIDTH):
                first_dim_val = upper_left_dim + row * PuzzleTester.row_to_row_step_size()

                # Test the extraction of pixel values.
                test_arr = PuzzleTester.build_pixel_list(first_dim_val, True)
                row_val = piece.get_row_pixels(row)
                assert(numpy.array_equal(row_val, test_arr))  # Verify the two arrays are equal.

                # Test the reversing
                reverse_list = True
                test_arr = PuzzleTester.build_pixel_list(first_dim_val, True, reverse_list)
                row_val = piece.get_row_pixels(row, reverse_list)
                assert(numpy.array_equal(row_val, test_arr))

            for col in range(0, PuzzleTester.PIECE_WIDTH):
                first_dim_val = upper_left_dim + col * PuzzleTester.NUMB_PIXEL_DIMENSIONS

                # Test the extraction of pixel values.
                is_col = False
                test_arr = PuzzleTester.build_pixel_list(first_dim_val, is_col)
                col_val = piece.get_column_pixels(col)
                assert(numpy.array_equal(col_val, test_arr))  # Verify the two arrays are equal.

                # Test the reversing
                reverse_list = True
                test_arr = PuzzleTester.build_pixel_list(first_dim_val, is_col, reverse_list)
                col_val = piece.get_column_pixels(col, reverse_list)
                assert(numpy.array_equal(col_val, test_arr))

        # Calculate the asymmetric distance for two neighboring pieces on ADJACENT SIDES
        asym_dist = PuzzlePiece.calculate_asymmetric_distance(pieces[0], PuzzlePieceSide.right,
                                                              pieces[1], PuzzlePieceSide.left)
        assert(asym_dist == 0)
        asym_dist = PuzzlePiece.calculate_asymmetric_distance(pieces[1], PuzzlePieceSide.left,
                                                              pieces[0], PuzzlePieceSide.right)
        assert(asym_dist == 0)
        # Calculate the asymmetric distance for two neighboring pieces on ADJACENT SIDES
        pieces_per_row = int(math.sqrt(PuzzleTester.NUMB_PUZZLE_PIECES))
        asym_dist = PuzzlePiece.calculate_asymmetric_distance(pieces[0], PuzzlePieceSide.bottom,
                                                              pieces[pieces_per_row], PuzzlePieceSide.top)
        assert(asym_dist == 0)
        asym_dist = PuzzlePiece.calculate_asymmetric_distance(pieces[pieces_per_row], PuzzlePieceSide.top,
                                                              pieces[0], PuzzlePieceSide.bottom)
        assert(asym_dist == 0)

        # Calculate the asymmetric distance for pieces two spaces away
        asym_dist = PuzzlePiece.calculate_asymmetric_distance(pieces[0], PuzzlePieceSide.right,
                                                              pieces[2], PuzzlePieceSide.left)
        expected_dist = PuzzleTester.PIECE_WIDTH * PuzzleTester.NUMB_PIXEL_DIMENSIONS * PuzzleTester.piece_to_piece_step_size()
        assert(asym_dist == expected_dist)
        # Calculate the asymmetric distance for pieces two spaces away
        asym_dist = PuzzlePiece.calculate_asymmetric_distance(pieces[2], PuzzlePieceSide.left,
                                                              pieces[0], PuzzlePieceSide.right)
        expected_dist = PuzzleTester.PIECE_WIDTH * PuzzleTester.NUMB_PIXEL_DIMENSIONS * PuzzleTester.piece_to_piece_step_size()
        assert(asym_dist == expected_dist)

        # Calculate the asymmetric distance for two neighboring pieces on non-adjacent
        asym_dist = PuzzlePiece.calculate_asymmetric_distance(pieces[0], PuzzlePieceSide.top,
                                                              pieces[1], PuzzlePieceSide.top)
        # Distance between first pixel in top row of piece to last pixel of piece j almost like to puzzle pieces
        pixel_to_pixel_dist = -1 * ((2 * PuzzleTester.PIECE_WIDTH - 1) * PuzzleTester.NUMB_PIXEL_DIMENSIONS)
        pixel_to_pixel_dist -= PuzzleTester.row_to_row_step_size()
        # Calculate the expected distance
        expected_dist = 0
        for i in range(0, PuzzleTester.PIECE_WIDTH * PuzzleTester.NUMB_PIXEL_DIMENSIONS):
            expected_dist += abs(pixel_to_pixel_dist)
            if i % PuzzleTester.NUMB_PIXEL_DIMENSIONS == PuzzleTester.NUMB_PIXEL_DIMENSIONS - 1:
                pixel_to_pixel_dist += 2 * PuzzleTester.NUMB_PIXEL_DIMENSIONS
        assert(asym_dist == expected_dist)

    @staticmethod
    def build_pixel_list(start_value, is_row, reverse_list=False):
        """
        Pixel List Builder

        Given a starting value for the first pixel in the first dimension, this function gets the pixel values
        in an array similar to a call to "get_row_pixels" or "get_column_pixels" for a puzzle piece.

        Args:
            start_value (int): Value of the first (i.e. lowest valued) pixel's first dimension

            is_row (bool): True if building a pixel list for a row and "False" if it is a column.  This is used to
            determine the stepping factor from one pixel to the next.

            reverse_list (bool): If "True", HIGHEST valued pixel dimension is returned in the first index of the list
            and all subsequent pixel values are monotonically DECREASING.  If "False", the LOWEST valued pixel dimension
            is returned in the first index of the list and all subsequent pixel values are monotonically increasing.

        Returns ([int]): An array of individual values simulating a set of pixels
        """

        # Determine the pixel to pixel step size
        if is_row:
            pixel_offset = PuzzleTester.NUMB_PIXEL_DIMENSIONS
        else:
            pixel_offset = PuzzleTester.row_to_row_step_size()

        # Build the list of pixel values
        pixels = numpy.zeros((PuzzleTester.PIECE_WIDTH, PuzzleTester.NUMB_PIXEL_DIMENSIONS))
        for i in range(0, PuzzleTester.PIECE_WIDTH):
            pixel_start = start_value + i * pixel_offset
            for j in range(0, PuzzleTester.NUMB_PIXEL_DIMENSIONS):
                pixels[i, j] = pixel_start + j

        # Return the result either reversed or not.
        if reverse_list:
            return pixels[::-1]
        else:
            return pixels

    @staticmethod
    def row_to_row_step_size():
        """
        Row to Row Step Size

        For a given pixel's given dimension, this function returns the number of dimensions between this pixel and
        the matching pixel exactly one row below.

        It is essentially the number of dimensions multiplied by the width of the original image (in pixels).

        Returns (int): Offset in dimensions.
        """
        step_size = PuzzleTester.NUMB_PIXEL_DIMENSIONS * PuzzleTester.PIECE_WIDTH * math.sqrt(PuzzleTester.NUMB_PUZZLE_PIECES)
        return int(step_size)

    @staticmethod
    def piece_to_piece_step_size():
        """
        Piece to Piece Step Size

        For a given pixel's given dimension, this function returns the number of dimensions between this pixel and
        the matching pixel exactly one puzzle piece away.

        It is essentially the number of dimensions multiplied by the width of a puzzle piece (in pixels).

        Returns (int): Offset in dimensions.
        """
        return PuzzleTester.NUMB_PIXEL_DIMENSIONS * PuzzleTester.PIECE_WIDTH


    @staticmethod
    def build_dummy_puzzle():
        """
        Dummy Puzzle Builder

        Using an image on the disk, this function builds a dummy puzzle using a Numpy array that is manually
        loaded with sequentially increasing pixel values.

        Returns (Puzzle): A puzzle where each pixel dimension from left to right sequentially increases by
        one.
        """

        # Create a puzzle whose image data will be overridden
        puzzle = Puzzle(0, PuzzleTester.TEST_IMAGE_FILENAME)

        # Define the puzzle side
        piece_width = PuzzleTester.PIECE_WIDTH
        numb_pieces = PuzzleTester.NUMB_PUZZLE_PIECES
        numb_dim = PuzzleTester.NUMB_PIXEL_DIMENSIONS

        # Define the array
        dummy_img = numpy.zeros((piece_width * math.sqrt(numb_pieces), piece_width * math.sqrt(numb_pieces), numb_dim))
        # populate the array
        val = PuzzleTester.TEST_ARRAY_FIRST_PIXEL_VALUE
        img_shape = dummy_img.shape
        for row in range(0, img_shape[0]):
            for col in range(0, img_shape[1]):
                for dim in range(0, img_shape[2]):
                    dummy_img[row, col, dim] = val
                    val += 1

        # Overwrite the image parameters
        puzzle._img = dummy_img
        puzzle._img_LAB = dummy_img
        puzzle._img_width = img_shape[1]
        puzzle._img_height = img_shape[0]
        puzzle._piece_width = PuzzleTester.PIECE_WIDTH
        puzzle._grid_size = (math.sqrt(PuzzleTester.NUMB_PUZZLE_PIECES), math.sqrt(PuzzleTester.NUMB_PUZZLE_PIECES))

        # Remake the puzzle pieces
        puzzle.make_pieces()

        return puzzle

if __name__ == '__main__':
    unittest.main()
