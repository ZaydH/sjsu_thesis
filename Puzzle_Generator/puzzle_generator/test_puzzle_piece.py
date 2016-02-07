import unittest
from puzzle_piece import PuzzlePiece

class PuzzlePieceTestCase(unittest.TestCase):

    def test_get_rotated_puzzle_piece(self):
        """
        Checks whether rotation of a puzzle piece works correctly
        """
        width = 3
        piece = PuzzlePiece(width)
        piece.set_rotation(PuzzlePiece.Rotation.degree_90)
        # Rotate upper left piece
        self.assertTrue( piece._get_rotated_coordinates(0,0) == (2,0))
        # Rotate lower left piece
        self.assertTrue( piece._get_rotated_coordinates(0,2) == (0,0))
        # Rotate piece to the right of the top left
        self.assertTrue( piece._get_rotated_coordinates(1, 0) == (2,1))

# If this function is main, then run the unit tests.
if __name__ == "__main__":
    unittest.main()