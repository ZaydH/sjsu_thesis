import unittest

from hammoudeh_puzzle_solver.puzzle_importer import PuzzleType
from hammoudeh_puzzle_solver.puzzle_importer_tester import PuzzleTester
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePieceSide, PuzzlePiece
from paikin_tal_solver.inter_piece_distance import InterPieceDistance


class PaikinTalTester(unittest.TestCase):

    def distance_calculator(self):
        # Make a dummy puzzle
        puzzle = PuzzleTester.build_dummy_puzzle()

        # Get the distance info
        dist_info = InterPieceDistance(puzzle.pieces, PuzzlePiece.calculate_asymmetric_distance, PuzzleType.type2)
        # Verify the best buddy info for neighboring pieces
        for i in range(0, puzzle.pieces):
            # Check not an end piece on the right side of the image
            if i % PuzzleTester.GRID_SIZE[1] != PuzzleTester.GRID_SIZE[1] -1:
                assert(dist_info.best_buddies(i, PuzzlePieceSide.right) == [(i + 1, PuzzlePieceSide.left)])

            # Check not an end piece on the top of the image
            if i >= PuzzleTester.GRID_SIZE[1]:
                assert(dist_info.best_buddies(i, PuzzlePieceSide.top) == [(i - PuzzleTester.GRID_SIZE[1], PuzzlePieceSide.bottom)])

        # Verify the middle piece is selected as the starting piece
        middle_piece = 4
        assert(dist_info.next_starting_piece() == middle_piece)

        # Verify you get the same result even if passed an array of pieces.
        all_pieces_false = [False] * len(puzzle.pieces)
        assert(dist_info.next_starting_piece(all_pieces_false) == middle_piece)

        # Verify that if all other pieces with more than one neighbor are excluded, then the only piece
        # with exactly one neighbor is selected.
        seed_piece_mask = [False] * len(puzzle.pieces)
        seed_piece_mask[1] = seed_piece_mask[3] = seed_piece_mask[4] = seed_piece_mask[5] = False
        assert(dist_info.next_starting_piece(seed_piece_mask) == 7)

if __name__ == '__main__':
    unittest.main()
