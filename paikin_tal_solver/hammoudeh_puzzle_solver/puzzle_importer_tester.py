import random
import unittest
import math
import numpy

from hammoudeh_puzzle_solver.puzzle_importer import Puzzle
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePiece, PuzzlePieceRotation


class PuzzleTester(unittest.TestCase):
    PIECE_WIDTH = 5
    NUMB_PUZZLE_PIECES = 9

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

        # Create a puzzle whose image data will be overridden
        puzzle = Puzzle(0, PuzzleTester.TEST_IMAGE_FILENAME)

        # Build a dummy image for testing.
        img = PuzzleTester.build_dummy_array()
        img_shape = img.shape

        # Overwrite the image parameters
        puzzle._img = img
        puzzle._img_LAB = img
        puzzle._img_width = img_shape[1]
        puzzle._img_height = img_shape[0]
        puzzle._piece_width = PuzzleTester.PIECE_WIDTH
        puzzle._grid_size = (math.sqrt(PuzzleTester.NUMB_PUZZLE_PIECES), math.sqrt(PuzzleTester.NUMB_PUZZLE_PIECES))

        # Remake the puzzle pieces
        puzzle.make_pieces()

        # Get the puzzle pieces
        pieces = puzzle.pieces
        # Get the first piece
        first_piece = pieces[0]

    @staticmethod
    def build_dummy_array():
        """

        Returns:

        """
        # Define the puzzle side
        piece_width = PuzzleTester.PIECE_WIDTH
        numb_pieces = PuzzleTester.NUMB_PUZZLE_PIECES

        # Define the array
        arr = numpy.zeros((piece_width * math.sqrt(numb_pieces), piece_width * math.sqrt(numb_pieces), 3))
        # populate the array
        val = 0
        shape = arr.shape
        for row in range(0, shape[0]):
            for col in range(0, shape[1]):
                for dim in range(0, shape[2]):
                    arr[row, col, dim] = val
                    val += 1
        return arr

if __name__ == '__main__':
    unittest.main()
