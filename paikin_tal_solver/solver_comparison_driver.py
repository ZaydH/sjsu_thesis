import os

from hammoudeh_puzzle import config
from hammoudeh_puzzle.puzzle_importer import PuzzleType
from paikan_tal_solver_driver import paikin_tal_driver

SINGLE_PUZZLE_PROGRESS_FILENAME = ".\\single_puzzle_progress.txt"

PUZZLE_TYPE = PuzzleType.type2
PIECE_WIDTH = 28  # pixels


def run_comparison_driver():
    _perform_single_puzzle_solving()


def _perform_single_puzzle_solving():
    """
    Runs all of the 20 805 piece images as individual puzzles.  These serve as the baseline for optimal performance.
    """
    # Run single puzzle solver
    while True:

        # If file does not exist, then
        if not os.path.exists(SINGLE_PUZZLE_PROGRESS_FILENAME):
            next_puzzle_id = 1
        else:
            # Tet the
            with open(SINGLE_PUZZLE_PROGRESS_FILENAME, 'r') as progress_file:
                next_puzzle_id = int(progress_file.readline())

        # If all images already completed, then exit.
        if next_puzzle_id > config.NUMBER_805_PIECE_PUZZLES:
            break

        # Build the image file
        image_filename = [config.build_805_piece_filename(next_puzzle_id)]
        paikin_tal_driver(image_filename, PUZZLE_TYPE, PIECE_WIDTH)

        # Go to the next puzzle and update the progress tracker
        next_puzzle_id = 1
        with open(SINGLE_PUZZLE_PROGRESS_FILENAME, 'w') as progress_file:
            progress_file.write(str(next_puzzle_id))

if '__name__' == '__main__':
    run_comparison_driver()
