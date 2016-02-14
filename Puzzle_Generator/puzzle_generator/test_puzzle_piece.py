import unittest
from puzzle_piece import PuzzlePiece, PieceRotation, PieceSide
from PIL import Image


class PuzzlePieceTestCase(unittest.TestCase):

    PRINT_DEBUG_MESSAGES = False

    def test_width(self):
        """
        Tests the "width" property for the class "PuzzlePiece."
        """
        for width in range(PuzzlePiece.MINIMUM_WIDTH, 2*PuzzlePiece.MINIMUM_WIDTH + 1):
            piece = PuzzlePiece(width)
            self.assertTrue(width == piece.width)

        # Test with a passed in image
        width = 10 * PuzzlePiece.MINIMUM_WIDTH
        white_piece = PuzzlePiece(width, (0, 0), Image.new("RGB", (width, width), "white"), 0, 0)
        self.assertTrue(white_piece.width == width)

        # Create a black piece and test its width since it is essentially free
        black_piece = PuzzlePiece(width, (0, 0), Image.new("RGB", (width, width), "black"), 0, 0)
        self.assertTrue(black_piece.width == width)

        # Calculate distance between a set of white and black pixels
        predicted_dist = 3 * (255 - 0) ** 2
        predicted_dist *= width

        # Test with different rotations and sides and ensure it does not affect the calculations.
        for white_rotation in PieceRotation.get_all_rotations():
            white_piece.rotation = white_rotation
            for black_rotation in PieceRotation.get_all_rotations():
                black_piece.rotation = black_rotation
                for side in PieceSide.get_all_sides():
                    actual_dist = PuzzlePiece.calculate_pieces_edge_distance(white_piece, side, black_piece)
                    self.assertTrue(actual_dist == predicted_dist)

                    # Ensure the calculation is the same bidirectionally
                    other_side = side.paired_edge
                    actual_dist = PuzzlePiece.calculate_pieces_edge_distance(black_piece, other_side, white_piece)
                    self.assertTrue(actual_dist == predicted_dist)

                    # Ensure the distance between a piece and itself is 0
                    self.assertTrue(0 == PuzzlePiece.calculate_pieces_edge_distance(white_piece, side, white_piece))
                    self.assertTrue(0 == PuzzlePiece.calculate_pieces_edge_distance(black_piece, side, black_piece))

    def test_get_rotated_puzzle_piece(self):
        """
        Checks whether rotation of a puzzle piece works correctly
        """
        width = PuzzlePiece.MINIMUM_WIDTH + 5
        piece = PuzzlePiece(width)
        piece.rotation = PieceRotation.degree_90
        # Rotate upper left piece
        self.assertTrue(piece._get_rotated_coordinates(0, 0) == (width - 1, 0))
        # Rotate lower left piece
        self.assertTrue(piece._get_rotated_coordinates(0, width - 1) == (0, 0))
        # Rotate piece to the right of the top left
        self.assertTrue(piece._get_rotated_coordinates(1, 0) == (width - 1, 1))

        piece.rotation = PieceRotation.degree_180
        # Rotate upper left piece
        self.assertTrue(piece._get_rotated_coordinates(0, 0) == (width - 1, width - 1))
        # Rotate lower left piece
        self.assertTrue(piece._get_rotated_coordinates(0, width - 1) == (width - 1, 0))
        # Rotate piece to the right of the top left
        self.assertTrue(piece._get_rotated_coordinates(1, 0) == (width - 1 - 1, width - 1))

        # Test a second width
        width = 15
        self.assertTrue(width % 2 == 1)
        piece = PuzzlePiece(width)
        for rotation in PieceRotation.get_all_rotations():
            piece.rotation = rotation
            # Since center of the puzzle, rotation should have no effect
            self.assertTrue(piece._get_rotated_coordinates(width // 2, width // 2) == (width // 2, width // 2))

    def test_get_numb_90_degree_rotations(self):
        """
        Tests the method "get_numb_90_rotations_to_other" in the class "PieceRotation"
        """
        # Get all of the possible rotations
        all_rotations = PieceRotation.get_all_rotations()
        tot_numb_rotation = len(all_rotations)

        # Iterate through all rotations and get the difference
        for i in range(0, len(all_rotations)):
            rot1 = all_rotations[i]
            for j in range(0, len(all_rotations)):
                rot2 = all_rotations[j]
                # Calculate the predicted number of rotations.
                predicted_numb_rotations = (tot_numb_rotation + (j - i)) % tot_numb_rotation
                # Verify actual matches predicted
                self.assertTrue(rot1.get_numb_90_rotations_to_other(rot2) == predicted_numb_rotations)

    def test_get_unrotated_puzzle_piece(self):
        """
        Tests getting the unrotated x/y coordinate behavior of the puzzle piece
        module.  It does this test by first performing a rotation, then performing
        an unrotation and seeing if the unrotated values match the original
        x/y coordinates.
        """

        width = 20
        piece = PuzzlePiece(width)
        piece.rotation = PieceRotation.degree_180
        self.assertTrue(piece._get_unrotated_coordinates(width - 1, width - 1) == (0, 0))

        # Test  a set of widths
        test_widths = [11, 43, 55, 76, 105, 155]
        for width in test_widths:
            # Create an example piece
            piece = PuzzlePiece(width)
            # Go through all the rotations
            for rotation in PieceRotation.get_all_rotations():
                piece.rotation = rotation
                # Test all possible x coordinates
                for x in range(0, width):
                    # Test all possible y coordinates
                    for y in range(0, width):
                        rotated_x, rotated_y = piece._get_rotated_coordinates(x, y)
                        unrotated_coordinate = piece._get_unrotated_coordinates(rotated_x, rotated_y)
                        # Print test conditions if enabled.
                        if PuzzlePieceTestCase.PRINT_DEBUG_MESSAGES:
                            print "Setting: width = %d, x = %d, y = %d, rotation = %d" % (width, x, y, rotation.value)
                            print "rotated_x = %d, rotated_y = %d" % (rotated_x, rotated_y)
                            print "unrotated_x = %d, unrotated_y = %d" % unrotated_coordinate
                        # Check the calculated value.
                        self.assertTrue((x, y) == unrotated_coordinate)

    def test_piece_neighbor(self):
        """
        Tests the method "get_neighbor_coordinate" for the class "PuzzlePiece."
        """
        # Define the puzzle piece
        piece = PuzzlePiece(PuzzlePiece.MINIMUM_WIDTH)
        coordinate = (PuzzlePiece.MINIMUM_WIDTH // 2, PuzzlePiece.MINIMUM_WIDTH // 2)
        piece.assigned_location = coordinate

        # Define the test conditions
        sides = PieceSide.get_all_sides()
        offset = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        # Iterate through the sides and offsets and verify the neighbor is correct.
        for i in range(0, len(sides)):
            neighbor_coord = piece.get_neighbor_coordinate(sides[i])
            self.assertTrue(neighbor_coord == (coordinate[0] + offset[i][0], coordinate[1] + offset[i][1]))

    def test_edge_start_corner_coordinate_and_pixel_step(self):
        """
        Tests the method ::"get_edge_start_corner_coordinate_and_pixel_step":: in the class "PuzzlePiece."
        """
        # Check top edge x and y
        width = 10
        piece = PuzzlePiece(width)
        start_coord, offset = piece.get_edge_start_corner_coordinate_and_pixel_step(PieceSide.top_side)
        self.assertTrue(start_coord == (0, 0) and offset == (1, 0))

        # Check bottom edge x and y
        width = 500
        piece = PuzzlePiece(width)
        start_coord, offset = piece.get_edge_start_corner_coordinate_and_pixel_step(PieceSide.bottom_side)
        self.assertTrue(start_coord == (0, width - 1) and offset == (1, 0))

        # Check top edge x and y
        width = 20
        piece = PuzzlePiece(width)
        start_coord, offset = piece.get_edge_start_corner_coordinate_and_pixel_step(PieceSide.left_side)
        self.assertTrue(start_coord == (0, 0) and offset == (0, 1))

        # Check bottom edge x and y
        width = 500
        piece = PuzzlePiece(width)
        start_coord, offset = piece.get_edge_start_corner_coordinate_and_pixel_step(PieceSide.right_side)
        self.assertTrue(start_coord == (width - 1, 0) and offset == (0, 1))


# If this function is main, then run the unit tests.
if __name__ == "__main__":
    unittest.main()
