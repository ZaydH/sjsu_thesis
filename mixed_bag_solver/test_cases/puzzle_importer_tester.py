"""Jigsaw Puzzle and Puzzle Piece Unittest Module

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""
import math
import random
import unittest
import os

import numpy as np

from hammoudeh_puzzle.puzzle_importer import Puzzle, PuzzleTester, PuzzleResultsCollection, \
    PieceDirectAccuracyResult
from hammoudeh_puzzle.puzzle_piece import PuzzlePiece, PuzzlePieceSide, PuzzlePieceRotation


# noinspection PyMethodMayBeStatic
from hammoudeh_puzzle.solver_helper import PuzzleLocation


# noinspection PyMethodMayBeStatic
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
                assert(np.array_equal(row_val, test_arr))  # Verify the two arrays are equal.

                # Test the reversing
                reverse_list = True
                test_arr = PuzzleTester.build_pixel_list(first_dim_val, True, reverse_list)
                row_val = piece.get_row_pixels(row, reverse_list)
                assert(np.array_equal(row_val, test_arr))

            for col in range(0, PuzzleTester.PIECE_WIDTH):
                first_dim_val = upper_left_dim + col * PuzzleTester.NUMB_PIXEL_DIMENSIONS

                # Test the extraction of pixel values.
                is_col = False
                test_arr = PuzzleTester.build_pixel_list(first_dim_val, is_col)
                col_val = piece.get_column_pixels(col)
                assert(np.array_equal(col_val, test_arr))  # Verify the two arrays are equal.

                # Test the reversing
                reverse_list = True
                test_arr = PuzzleTester.build_pixel_list(first_dim_val, is_col, reverse_list)
                col_val = piece.get_column_pixels(col, reverse_list)
                assert(np.array_equal(col_val, test_arr))

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

    def test__get_neighbor_piece_rotated_side(self):

        # Test calculation of the top side.
        assert PuzzlePiece._get_neighbor_piece_rotated_side((-1, 0), (0, 0)) == PuzzlePieceSide.top
        assert PuzzlePiece._get_neighbor_piece_rotated_side((10, 10), (11, 10)) == PuzzlePieceSide.top

        # Test calculation of the bottom side.
        assert PuzzlePiece._get_neighbor_piece_rotated_side((0, 4), (-1, 4)) == PuzzlePieceSide.bottom
        assert PuzzlePiece._get_neighbor_piece_rotated_side((6, 8), (5, 8)) == PuzzlePieceSide.bottom

        # Test calculation of the left side.
        assert PuzzlePiece._get_neighbor_piece_rotated_side((0, -1), (0, 0)) == PuzzlePieceSide.left
        assert PuzzlePiece._get_neighbor_piece_rotated_side((20, 10), (20, 11)) == PuzzlePieceSide.left

        # Test calculation of the right side.
        assert PuzzlePiece._get_neighbor_piece_rotated_side((0, 0), (0, -1)) == PuzzlePieceSide.right
        assert PuzzlePiece._get_neighbor_piece_rotated_side((13, 6), (13, 5)) == PuzzlePieceSide.right

    def test_determine_unrotated_side(self):

        # Find the unrotated side of a puzzle piece
        assert PuzzlePiece._determine_unrotated_side(PuzzlePieceRotation.degree_0, PuzzlePieceSide.top) == PuzzlePieceSide.top
        assert PuzzlePiece._determine_unrotated_side(PuzzlePieceRotation.degree_0, PuzzlePieceSide.right) == PuzzlePieceSide.right
        assert PuzzlePiece._determine_unrotated_side(PuzzlePieceRotation.degree_0, PuzzlePieceSide.bottom) == PuzzlePieceSide.bottom
        assert PuzzlePiece._determine_unrotated_side(PuzzlePieceRotation.degree_0, PuzzlePieceSide.left) == PuzzlePieceSide.left

        # Find the puzzle piece side when the piece is rotated 90 degrees
        assert PuzzlePiece._determine_unrotated_side(PuzzlePieceRotation.degree_90, PuzzlePieceSide.top) == PuzzlePieceSide.left
        assert PuzzlePiece._determine_unrotated_side(PuzzlePieceRotation.degree_90, PuzzlePieceSide.right) == PuzzlePieceSide.top
        assert PuzzlePiece._determine_unrotated_side(PuzzlePieceRotation.degree_90, PuzzlePieceSide.bottom) == PuzzlePieceSide.right
        assert PuzzlePiece._determine_unrotated_side(PuzzlePieceRotation.degree_90, PuzzlePieceSide.left) == PuzzlePieceSide.bottom

        # Find the puzzle piece side when the piece is rotated 180 degrees
        assert PuzzlePiece._determine_unrotated_side(PuzzlePieceRotation.degree_180, PuzzlePieceSide.top) == PuzzlePieceSide.bottom
        assert PuzzlePiece._determine_unrotated_side(PuzzlePieceRotation.degree_180, PuzzlePieceSide.right) == PuzzlePieceSide.left
        assert PuzzlePiece._determine_unrotated_side(PuzzlePieceRotation.degree_180, PuzzlePieceSide.bottom) == PuzzlePieceSide.top
        assert PuzzlePiece._determine_unrotated_side(PuzzlePieceRotation.degree_180, PuzzlePieceSide.left) == PuzzlePieceSide.right

        # Find the puzzle piece side when the piece is rotated 270 degrees
        assert PuzzlePiece._determine_unrotated_side(PuzzlePieceRotation.degree_270, PuzzlePieceSide.top) == PuzzlePieceSide.right
        assert PuzzlePiece._determine_unrotated_side(PuzzlePieceRotation.degree_270, PuzzlePieceSide.right) == PuzzlePieceSide.bottom
        assert PuzzlePiece._determine_unrotated_side(PuzzlePieceRotation.degree_270, PuzzlePieceSide.bottom) == PuzzlePieceSide.left
        assert PuzzlePiece._determine_unrotated_side(PuzzlePieceRotation.degree_270, PuzzlePieceSide.left) == PuzzlePieceSide.top

    def test_calculate_placed_piece_rotation_degree_0(self):

        # Calculate for unrotated pieces placed in complementary locations (TOP)
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.top, PuzzlePieceSide.bottom,
                                                                       PuzzlePieceRotation.degree_0)
        assert placed_rotation == PuzzlePieceRotation.degree_0

        # Calculate for unrotated pieces placed in complementary locations (RIGHT)
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.right, PuzzlePieceSide.left,
                                                                       PuzzlePieceRotation.degree_0)
        assert placed_rotation == PuzzlePieceRotation.degree_0

        # Calculate for unrotated pieces placed in complementary locations (BOTTOM)
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.bottom, PuzzlePieceSide.top,
                                                                       PuzzlePieceRotation.degree_0)
        assert placed_rotation == PuzzlePieceRotation.degree_0
        # Calculate for unrotated pieces placed in complementary locations (BOTTOM)
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.bottom, PuzzlePieceSide.left,
                                                                       PuzzlePieceRotation.degree_90)
        assert placed_rotation == PuzzlePieceRotation.degree_0
        # Calculate for unrotated pieces placed in complementary locations (BOTTOM)
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.bottom, PuzzlePieceSide.bottom,
                                                                       PuzzlePieceRotation.degree_180)
        assert placed_rotation == PuzzlePieceRotation.degree_0
        # Calculate for unrotated pieces placed in complementary locations (BOTTOM)
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.bottom, PuzzlePieceSide.right,
                                                                       PuzzlePieceRotation.degree_270)
        assert placed_rotation == PuzzlePieceRotation.degree_0

        # Calculate for unrotated pieces placed in complementary locations (LEFT)
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.left, PuzzlePieceSide.right,
                                                                       PuzzlePieceRotation.degree_0)
        assert placed_rotation == PuzzlePieceRotation.degree_0

    def test_calculate_placed_piece_rotation_degree_90(self):

        # Calculate for unrotated pieces placed in complementary locations (TOP)
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.top, PuzzlePieceSide.left,
                                                                       PuzzlePieceRotation.degree_0)
        assert placed_rotation == PuzzlePieceRotation.degree_90
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.top, PuzzlePieceSide.bottom,
                                                                       PuzzlePieceRotation.degree_90)
        assert placed_rotation == PuzzlePieceRotation.degree_90

        # Calculate for unrotated pieces placed in complementary locations (RIGHT)
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.right, PuzzlePieceSide.top,
                                                                       PuzzlePieceRotation.degree_0)
        assert placed_rotation == PuzzlePieceRotation.degree_90
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.right, PuzzlePieceSide.left,
                                                                       PuzzlePieceRotation.degree_90)
        assert placed_rotation == PuzzlePieceRotation.degree_90

        # Calculate for unrotated pieces placed in complementary locations (BOTTOM)
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.bottom, PuzzlePieceSide.right,
                                                                       PuzzlePieceRotation.degree_0)
        assert placed_rotation == PuzzlePieceRotation.degree_90

        # Calculate for unrotated pieces placed in complementary locations (LEFT)
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.left, PuzzlePieceSide.bottom,
                                                                       PuzzlePieceRotation.degree_0)
        assert placed_rotation == PuzzlePieceRotation.degree_90

    def test_calculate_placed_piece_rotation_degree_180(self):

        # Calculate for unrotated pieces placed in complementary locations (TOP)
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.top, PuzzlePieceSide.top,
                                                                       PuzzlePieceRotation.degree_0)
        assert placed_rotation == PuzzlePieceRotation.degree_180
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.top, PuzzlePieceSide.left,
                                                                       PuzzlePieceRotation.degree_90)
        assert placed_rotation == PuzzlePieceRotation.degree_180

        # Calculate for unrotated pieces placed in complementary locations (RIGHT)
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.right, PuzzlePieceSide.right,
                                                                       PuzzlePieceRotation.degree_0)
        assert placed_rotation == PuzzlePieceRotation.degree_180
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.right, PuzzlePieceSide.top,
                                                                       PuzzlePieceRotation.degree_90)
        assert placed_rotation == PuzzlePieceRotation.degree_180

        # Calculate for unrotated pieces placed in complementary locations (BOTTOM)
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.bottom, PuzzlePieceSide.bottom,
                                                                       PuzzlePieceRotation.degree_0)
        assert placed_rotation == PuzzlePieceRotation.degree_180

        # Calculate for unrotated pieces placed in complementary locations (LEFT)
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.left, PuzzlePieceSide.left,
                                                                       PuzzlePieceRotation.degree_0)
        assert placed_rotation == PuzzlePieceRotation.degree_180
        # Calculate for unrotated pieces placed in complementary locations (LEFT)
        placed_rotation = PuzzlePiece._calculate_placed_piece_rotation(PuzzlePieceSide.left, PuzzlePieceSide.top,
                                                                       PuzzlePieceRotation.degree_270)
        assert placed_rotation == PuzzlePieceRotation.degree_180

    def test_get_neighbor_locations_and_sides(self):

        # Test with no rotation
        test_loc = (10, 20)
        test_rotation = PuzzlePieceRotation.degree_0
        location_and_sides = PuzzlePiece._get_neighbor_locations_and_sides(test_loc, test_rotation)
        assert location_and_sides[0] == ((test_loc[0] - 1, test_loc[1]), PuzzlePieceSide.top)
        assert location_and_sides[1] == ((test_loc[0], test_loc[1] + 1), PuzzlePieceSide.right)
        assert location_and_sides[2] == ((test_loc[0] + 1, test_loc[1]), PuzzlePieceSide.bottom)
        assert location_and_sides[3] == ((test_loc[0], test_loc[1] - 1), PuzzlePieceSide.left)

        # Test with 90 degrees of rotation
        test_loc = (35, 44)
        test_rotation = PuzzlePieceRotation.degree_90
        location_and_sides = PuzzlePiece._get_neighbor_locations_and_sides(test_loc, test_rotation)
        assert location_and_sides[0] == ((test_loc[0] - 1, test_loc[1]), PuzzlePieceSide.left)
        assert location_and_sides[1] == ((test_loc[0], test_loc[1] + 1), PuzzlePieceSide.top)
        assert location_and_sides[2] == ((test_loc[0] + 1, test_loc[1]), PuzzlePieceSide.right)
        assert location_and_sides[3] == ((test_loc[0], test_loc[1] - 1), PuzzlePieceSide.bottom)

        # Test with 180 degrees of rotation
        test_loc = (66, -15)
        test_rotation = PuzzlePieceRotation.degree_180
        location_and_sides = PuzzlePiece._get_neighbor_locations_and_sides(test_loc, test_rotation)
        assert location_and_sides[0] == ((test_loc[0] - 1, test_loc[1]), PuzzlePieceSide.bottom)
        assert location_and_sides[1] == ((test_loc[0], test_loc[1] + 1), PuzzlePieceSide.left)
        assert location_and_sides[2] == ((test_loc[0] + 1, test_loc[1]), PuzzlePieceSide.top)
        assert location_and_sides[3] == ((test_loc[0], test_loc[1] - 1), PuzzlePieceSide.right)

        # Test with 270 degrees of rotation
        test_loc = (-56, 23)
        test_rotation = PuzzlePieceRotation.degree_270
        location_and_sides = PuzzlePiece._get_neighbor_locations_and_sides(test_loc, test_rotation)
        assert location_and_sides[0] == ((test_loc[0] - 1, test_loc[1]), PuzzlePieceSide.right)
        assert location_and_sides[1] == ((test_loc[0], test_loc[1] + 1), PuzzlePieceSide.bottom)
        assert location_and_sides[2] == ((test_loc[0] + 1, test_loc[1]), PuzzlePieceSide.left)
        assert location_and_sides[3] == ((test_loc[0], test_loc[1] - 1), PuzzlePieceSide.top)

    def test_neighbor_id_numbers(self):
        """
        Test the functionality where the module gets the neighbor identification numbers for different puzzle pieces
        """

        # Build a known test puzzle.
        puzzle = PuzzleTester.build_dummy_puzzle()

        # Get the pieces from the dummy puzzle.
        pieces = puzzle.pieces

        # Check the neighbor coordinates of the first piece
        assert pieces[0].original_neighbor_id_numbers_and_sides == ((None, PuzzlePieceSide.top), (1, PuzzlePieceSide.right),
                                                                    (3, PuzzlePieceSide.bottom), (None, PuzzlePieceSide.left))
        assert pieces[1].original_neighbor_id_numbers_and_sides == ((None, PuzzlePieceSide.top), (2, PuzzlePieceSide.right),
                                                                    (4, PuzzlePieceSide.bottom), (0, PuzzlePieceSide.left))
        assert pieces[2].original_neighbor_id_numbers_and_sides == ((None, PuzzlePieceSide.top), (None, PuzzlePieceSide.right),
                                                                    (5, PuzzlePieceSide.bottom), (1, PuzzlePieceSide.left))
        assert pieces[3].original_neighbor_id_numbers_and_sides == ((0, PuzzlePieceSide.top), (4, PuzzlePieceSide.right),
                                                                    (6, PuzzlePieceSide.bottom), (None, PuzzlePieceSide.left))
        assert pieces[8].original_neighbor_id_numbers_and_sides == ((5, PuzzlePieceSide.top), (None, PuzzlePieceSide.right),
                                                                    (None, PuzzlePieceSide.bottom), (7, PuzzlePieceSide.left))

    def test_puzzle_polygon(self):
        # Define the color side pairing for the test
        color_side = [(PuzzlePieceSide.top, PieceDirectAccuracyResult.different_puzzle.value),
                      (PuzzlePieceSide.right, PieceDirectAccuracyResult.correct_placement.value),
                      (PuzzlePieceSide.bottom, PieceDirectAccuracyResult.wrong_location.value),
                      (PuzzlePieceSide.left, PieceDirectAccuracyResult.wrong_rotation.value)]
        # noinspection PyUnusedLocal
        image = PuzzlePiece.create_side_polygon_image(color_side, 25, 25)
        # Puzzle.display_image(image)

    def test_accuracy_calculator(self):

        # Build a known test puzzle.
        dummy_puzzle = PuzzleTester.build_dummy_puzzle()
        for piece in dummy_puzzle.pieces:
            piece._assign_to_original_location()
            piece._set_id_number_to_original_id()
            piece.rotation = PuzzlePieceRotation.degree_0

        # Test different conditions
        for i in xrange(0, 2):

            if i == 0:
                # Set all piece rotation to the default
                for piece in dummy_puzzle.pieces:
                    piece.rotation = PuzzlePieceRotation.degree_0
            elif i == 1:
                dummy_puzzle.pieces[0].rotation = PuzzlePieceRotation.degree_90

            # Build the results information collection
            results_information = PuzzleResultsCollection([dummy_puzzle.pieces], "." + os.sep + "dummy.jpg")
            results_information.calculate_accuracies([dummy_puzzle])

            # Verify the neighbor results
            numb_pieces = len(dummy_puzzle.pieces)
            results = results_information.results[0]

            if i == 0:
                assert results.standard_direct_accuracy.numb_correct_placements == numb_pieces
                assert results.standard_direct_accuracy.numb_wrong_rotation == 0

                assert results.modified_direct_accuracy.numb_correct_placements == numb_pieces
                assert results.modified_direct_accuracy.numb_wrong_rotation == 0
            elif i == 1:
                assert results.standard_direct_accuracy.numb_correct_placements == numb_pieces - 1
                assert results.standard_direct_accuracy.numb_wrong_rotation == 1

                assert results.modified_direct_accuracy.numb_correct_placements == numb_pieces - 1
                assert results.modified_direct_accuracy.numb_wrong_rotation == 1

            assert results.standard_direct_accuracy.numb_different_puzzle == 0
            assert results.standard_direct_accuracy.numb_wrong_location == 0
            assert results.standard_direct_accuracy.total_numb_pieces_in_solved_puzzle == numb_pieces

            assert results.modified_direct_accuracy.numb_wrong_location == 0
            assert results.modified_direct_accuracy.total_numb_pieces_in_solved_puzzle == numb_pieces

            # Check the neighbor accuracy
            numb_sides = PuzzlePieceSide.get_numb_sides()
            if i == 0:
                assert results.modified_neighbor_accuracy.correct_neighbor_count == numb_sides * numb_pieces
                assert results.modified_neighbor_accuracy.wrong_neighbor_count == 0
            elif i == 1:
                assert results.modified_neighbor_accuracy.correct_neighbor_count == numb_sides * numb_pieces - 5
                assert results.modified_neighbor_accuracy.wrong_neighbor_count == 5
            assert results.modified_neighbor_accuracy.total_numb_pieces_in_solved_puzzle == numb_pieces

    def test_side_of_primary_adjacent_to_other_piece(self):
        # Check bottom
        primary = PuzzleLocation(1, 47, 48)
        other = PuzzleLocation(1, 48, 48)
        assert PuzzlePieceSide.bottom == Puzzle.get_side_of_primary_adjacent_to_other_piece(primary, other)

        # Check bottom
        primary = PuzzleLocation(1, 48, 48)
        other = PuzzleLocation(1, 47, 48)
        assert PuzzlePieceSide.top == Puzzle.get_side_of_primary_adjacent_to_other_piece(primary, other)

        # Check right
        primary = PuzzleLocation(1, 48, 47)
        other = PuzzleLocation(1, 48, 48)
        assert PuzzlePieceSide.right == Puzzle.get_side_of_primary_adjacent_to_other_piece(primary, other)

        # Check left
        primary = PuzzleLocation(1, 48, 49)
        other = PuzzleLocation(1, 48, 48)
        assert PuzzlePieceSide.left == Puzzle.get_side_of_primary_adjacent_to_other_piece(primary, other)

if __name__ == '__main__':
    unittest.main()
