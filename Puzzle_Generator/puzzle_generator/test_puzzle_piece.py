import unittest
from puzzle_piece import PuzzlePiece, Rotation, PieceSide, get_edge_starting_and_ending_x_and_y


class PuzzlePieceTestCase(unittest.TestCase):

    PRINT_DEBUG_MESSAGES = False

    def test_get_rotated_puzzle_piece(self):
        """
        Checks whether rotation of a puzzle piece works correctly
        """
        width = 13
        piece = PuzzlePiece(width)
        piece.set_rotation(Rotation.degree_90)
        # Rotate upper left piece
        self.assertTrue(piece._get_rotated_coordinates(0, 0) == (width - 1, 0))
        # Rotate lower left piece
        self.assertTrue(piece._get_rotated_coordinates(0, width - 1) == (0, 0))
        # Rotate piece to the right of the top left
        self.assertTrue(piece._get_rotated_coordinates(1, 0) == (width - 1, 1))

        piece.set_rotation(Rotation.degree_180)
        # Rotate upper left piece
        self.assertTrue(piece._get_rotated_coordinates(0, 0) == (width - 1, width - 1))
        # Rotate lower left piece
        self.assertTrue(piece._get_rotated_coordinates(0, width - 1) == (width - 1, 0))
        # Rotate piece to the right of the top left
        self.assertTrue(piece._get_rotated_coordinates(1, 0) == (width - 1 -1, width - 1))

        # Test a second width
        width = 15
        self.assertTrue(width % 2 == 1)
        piece = PuzzlePiece(width)
        for rotation in Rotation.get_all_rotations():
            piece.set_rotation(rotation)
            # Since center of the puzzle, rotation should have no effect
            self.assertTrue(piece._get_rotated_coordinates(width // 2, width // 2) == (width // 2, width // 2))

    def test_get_unrotated_puzzle_piece(self):
        """
        Tests getting the unrotated x/y coordinate behavior of the puzzle piece
        module.  It does this test by first performing a rotation, then performing
        an unrotation and seeing if the unrotated values match the original
        x/y coordinates.
        """

        width = 20
        piece = PuzzlePiece(width)
        piece.set_rotation(Rotation.degree_180)
        self.assertTrue(piece._get_unrotated_coordinates(width - 1, width - 1) == (0, 0))

        # Test  a set of widths
        test_widths = [11, 43, 55, 76, 105, 155]
        for width in test_widths:
            # Create an example piece
            piece = PuzzlePiece(width)
            # Go through all the rotations
            for rotation in Rotation.get_all_rotations():
                piece.set_rotation(rotation)
                # Test all possible x coordinates
                for x in range(0, width):
                    # Test all possible y coordinates
                    for y in range(0, width):
                        rotated_x, rotated_y = piece._get_rotated_coordinates(x, y)
                        unrotated_x, unrotated_y = piece._get_unrotated_coordinates(rotated_x,
                                                                                    rotated_y)
                        # Print test conditions if enabled.
                        if PuzzlePieceTestCase.PRINT_DEBUG_MESSAGES:
                            print "Setting: width = %d, x = %d, y = %d, rotation = %d" % (width, x, y, rotation.value)
                            print "rotated_x = %d, rotated_y = %d" % (rotated_x, rotated_y)
                            print "unrotated_x = %d, unrotated_y = %d" % (unrotated_x, unrotated_y)
                        # Check the calculated value.
                        self.assertTrue((x, y) == (unrotated_x, unrotated_y))


    def test_piece_edge_starting_and_ending_coordinates(self):

        # Check top edge x and y
        width = 10
        (x1, y1), (x2, y2) = get_edge_starting_and_ending_x_and_y(width, PieceSide.top_side)
        self.assertTrue((x1 == 0) and (y1 == 0) and (x2 == width - 1) and (y2 == 0))

        # Check bottom edge x and y
        width = 500
        (x1, y1), (x2, y2) = get_edge_starting_and_ending_x_and_y(width, PieceSide.bottom_side)
        self.assertTrue((x1 == 0) and (y1 == width - 1) and (x2 == width - 1) and (y2 == width - 1))

        # Check top edge x and y
        width = 20
        (x1, y1), (x2, y2) = get_edge_starting_and_ending_x_and_y(width, PieceSide.left_side)
        self.assertTrue((x1 == 0) and (y1 == 0) and (x2 == 0) and (y2 == width - 1))

        # Check bottom edge x and y
        width = 500
        (x1, y1), (x2, y2) = get_edge_starting_and_ending_x_and_y(width, PieceSide.right_side)
        self.assertTrue((x1 == width - 1) and (y1 == 0) and (x2 == width - 1) and (y2 == width - 1))

# If this function is main, then run the unit tests.
if __name__ == "__main__":
    unittest.main()