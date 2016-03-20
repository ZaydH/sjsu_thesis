import random
import unittest
import math
import numpy

from hammoudeh_puzzle_solver.puzzle_importer import Puzzle, PuzzleTester
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePiece, PuzzlePieceSide


class PuzzleImporterTester(unittest.TestCase):

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

    def test_piece_rotation_helper_functions(self):

        # Test calculation of the top side.
        assert PuzzlePiece.get_neighbor_piece_rotated_side((-1, 0), (0, 0)) == PuzzlePieceSide.top
        assert PuzzlePiece.get_neighbor_piece_rotated_side((10, 10), (11, 10)) == PuzzlePieceSide.top

        # Test calculation of the bottom side.
        assert PuzzlePiece.get_neighbor_piece_rotated_side((0, 0), (-1, 0)) == PuzzlePieceSide.bottom
        assert PuzzlePiece.get_neighbor_piece_rotated_side((6, 6), (5, 6)) == PuzzlePieceSide.bottom

        # Test calculation of the left side.
        assert PuzzlePiece.get_neighbor_piece_rotated_side((0, -1), (0, 0)) == PuzzlePieceSide.left
        assert PuzzlePiece.get_neighbor_piece_rotated_side((10, 10), (10, 11)) == PuzzlePieceSide.left

        # Test calculation of the right side.
        assert PuzzlePiece.get_neighbor_piece_rotated_side((0, 0), (0, -1)) == PuzzlePieceSide.right
        assert PuzzlePiece.get_neighbor_piece_rotated_side((6, 6), (6, 5)) == PuzzlePieceSide.right






if __name__ == '__main__':
    unittest.main()
