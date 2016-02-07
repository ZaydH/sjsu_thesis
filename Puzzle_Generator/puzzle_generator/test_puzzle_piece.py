import unittest
from puzzle_piece import PuzzlePiece

class PuzzlePieceTestCase(unittest.TestCase):

    PRINT_DEBUG_MESSAGES = False

    def test_get_rotated_puzzle_piece(self):
        """
        Checks whether rotation of a puzzle piece works correctly
        """
        width = 3
        piece = PuzzlePiece(width)
        piece.set_rotation(PuzzlePiece.Rotation.degree_90)
        # Rotate upper left piece
        self.assertTrue(piece._get_rotated_coordinates(0, 0) == (2, 0))
        # Rotate lower left piece
        self.assertTrue(piece._get_rotated_coordinates(0, 2) == (0, 0))
        # Rotate piece to the right of the top left
        self.assertTrue(piece._get_rotated_coordinates(1, 0) == (2, 1))

        piece.set_rotation(PuzzlePiece.Rotation.degree_180)
        # Rotate upper left piece
        self.assertTrue(piece._get_rotated_coordinates(0, 0) == (2, 2))
        # Rotate lower left piece
        self.assertTrue(piece._get_rotated_coordinates(0, 2) == (2, 0))
        # Rotate piece to the right of the top left
        self.assertTrue(piece._get_rotated_coordinates(1, 0) == (1, 2))

        # Test a second width
        width = 9
        piece = PuzzlePiece(width)
        for rotation in PuzzlePiece.Rotation.get_all_rotations():
            piece.set_rotation(rotation)
            # Since center of the puzzle, rotation should have no effect
            self.assertTrue(piece._get_rotated_coordinates(4, 4) == (4, 4))

    def test_get_unrotated_puzzle_piece(self):
        """
        Tests getting the unrotated x/y coordinate behavior of the puzzle piece
        module.  It does this test by first performing a rotation, then performing
        an unrotation and seeing if the unrotated values match the original
        x/y coordinates.
        """

        piece = PuzzlePiece(3)
        piece.set_rotation(PuzzlePiece.Rotation.degree_180)
        self.assertTrue(piece._get_unrotated_coordinates(2, 2) == (0, 0))

        # Test  a set of widths
        test_widths = [1, 3, 5, 6, 10, 15]
        for width in test_widths:
            # Create an example piece
            piece = PuzzlePiece(width)
            # Go through all the rotations
            for rotation in PuzzlePiece.Rotation.get_all_rotations():
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


# If this function is main, then run the unit tests.
if __name__ == "__main__":
    unittest.main()