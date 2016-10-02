import os

from hammoudeh_puzzle import config
from hammoudeh_puzzle.puzzle_importer import PuzzleType
from paikan_tal_solver_driver import paikin_tal_driver

SINGLE_PUZZLE_PROGRESS_FILENAME = ".\\single_puzzle_progress.txt"
TWO_PUZZLE_PROGRESS_FILENAME = ".\\two_puzzle_progress.txt"

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
            next_puzzle_id = config.MINIMUM_805_PIECE_IMAGE_NUMBER
        else:
            # Get the next puzzle from the file
            with open(SINGLE_PUZZLE_PROGRESS_FILENAME, 'r') as progress_file:
                next_puzzle_id = int(progress_file.readline())

        # If all images already completed, then exit.
        if next_puzzle_id > config.MAXIMUM_805_PIECE_IMAGE_NUMBER:
            break

        # Build the image file
        image_filename = [config.build_805_piece_filename(next_puzzle_id)]
        paikin_tal_driver(image_filename, PUZZLE_TYPE, PIECE_WIDTH)

        # Go to the next puzzle and update the progress tracker
        next_puzzle_id += 1
        with open(SINGLE_PUZZLE_PROGRESS_FILENAME, 'w') as progress_file:
            progress_file.write(str(next_puzzle_id))

def _perform_two_puzzle_solving():

    # Run single puzzle solver
    while True:

        # If file does not exist, then
        if not os.path.exists(TWO_PUZZLE_PROGRESS_FILENAME):
            first_puzzle_id = config.MINIMUM_805_PIECE_IMAGE_NUMBER
            second_puzzle_id = first_puzzle_id + 1
        else:
            # Get the image numbers for the first and second puzzle
            with open(SINGLE_PUZZLE_PROGRESS_FILENAME, 'r') as progress_file:
                split_line = progress_file.readline().split(",")
            first_puzzle_id = int(split_line[0])
            second_puzzle_id = int(split_line[1])

        # If all images already completed, then exit.
        if first_puzzle_id > config.MAXIMUM_805_PIECE_IMAGE_NUMBER:
            break


        # Go to the next puzzle and update the progress tracker
        second_puzzle_id += 1
        # Handle overflow of second puzzle IDD
        if second_puzzle_id > config.MAXIMUM_805_PIECE_IMAGE_NUMBER:
            first_puzzle_id += 1
            second_puzzle_id += 1
        with open(SINGLE_PUZZLE_PROGRESS_FILENAME, 'w') as progress_file:
            progress_file.write(str(first_puzzle_id) + "," str(second_puzzle_id))


if __name__ == '__main__':
    config.setup_logging()

    run_comparison_driver()
