import logging
import os

from hammoudeh_puzzle import config
from hammoudeh_puzzle.puzzle_importer import PuzzleType
from multipuzzle_solver_driver import run_multipuzzle_solver_driver
from paikan_tal_solver_driver import run_paikin_tal_driver

_PROGRESS_TRACKING_FOLDER = ".\\progress_tracker\\"
_IMAGE_ID_FILE_SEPARATOR = ","

PUZZLE_TYPE = PuzzleType.type2
PIECE_WIDTH = 28  # pixels


def run_comparison_driver():
    _perform_single_puzzle_solving()


def _perform_805_piece_comparison_puzzle_solving(dataset_name, numb_simultaneous_puzzles, minimum_image_number,
                                                 maximum_image_number, build_image_filename_function):
    """
    Runs all possible combinations of the Paikin and Tal solver of the specified size.  This uses the 805 piece data
    set from Ben Gurion University.

    Args:
        numb_simultaneous_puzzles (int): Number of images to be supplied to the solver.s
    """
    if numb_simultaneous_puzzles < 1:
        raise ValueError("The number of simultaneous puzzle to solve must be greater than or equal to 1."
                         )
    if maximum_image_number < minimum_image_number:
        raise ValueError("The maximum image number is less than the minimum image number.")

    if numb_simultaneous_puzzles < maximum_image_number - minimum_image_number + 1:
        raise ValueError("The number of simultaneous images exceeds the specified size of the dataset.")

    # Run single puzzle solver
    while True:

        progress_filename = _build_progress_filename(dataset_name, numb_simultaneous_puzzles)

        # If file does not exist, then create the list of images
        if not os.path.exists(progress_filename):
            puzzle_id_list = [i + minimum_image_number for i in xrange(0, numb_simultaneous_puzzles)]
        else:
            # Get the next puzzle from the file
            with open(progress_filename, 'r') as progress_file:
                puzzle_id_str_list = progress_file.readline().split(_IMAGE_ID_FILE_SEPARATOR)
            puzzle_id_list = [int(puzzle_id_str) for puzzle_id_str in puzzle_id_str_list]

        # If all images already completed, then exit.
        if puzzle_id_list[0] > maximum_image_number - numb_simultaneous_puzzles:
            return

        # Build the image file
        image_filenames = [build_image_filename_function(puzzle_id) for puzzle_id in puzzle_id_list]

        # Run the Multipuzzle Solver.  Paikin Tal will then use MultiPuzzle's pickle export.
        run_multipuzzle_solver_driver(image_filenames, PUZZLE_TYPE, PIECE_WIDTH)
        run_paikin_tal_driver(image_filenames, PUZZLE_TYPE, PIECE_WIDTH)

        puzzle_id_list = increment_puzzle_id_list(puzzle_id_list, maximum_image_number)
        # Make the tracking file.
        _write_progress_file(progress_filename, puzzle_id_list)


def _build_progress_filename(dataset_name, numb_simultaneous_puzzles):
    """
    Helper method to standardize creation of progress file names f

    Args:
        dataset_name (str): Unique descriptor for solver progress solving.
        numb_simultaneous_puzzles (int): Number of puzzles to solve simultaneously.

    Returns (str): Filename and path for a standardized naming of this dataset progress tracker.
    """
    return _PROGRESS_TRACKING_FOLDER + dataset_name + "_numb_puzzles_" + str(numb_simultaneous_puzzles) + ".csv"


def increment_puzzle_id_list(puzzle_id_list, maximum_puzzle_id_number):
    """
    Manages incrementing the list of puzzle identification number to cover all possible puzzle combinations.

    Args:
        puzzle_id_list (List[int]): Identification numbers of the puzzles being solved.
        maximum_puzzle_id_number (int): Maximum puzzle ID number for this dataset

    Returns (List[int]): Incremented puzzle IDs.
    """

    index_to_increment = len(puzzle_id_list) - 1
    _increment_previous_id(puzzle_id_list, index_to_increment, maximum_puzzle_id_number)
    return puzzle_id_list

def _increment_previous_id(puzzle_id_list, index_to_increment, maximum_puzzle_id_number):
    """
    Recursive approach to increment puzzle identification numbers.  Handles the actual recursion.

    All modifications are done in place.

    Args:
        puzzle_id_list (List[int)): List of identification numbers passed to the puzzle for solving.
        index_to_increment (int): Index of the puzzle identification number to be incremented in this function call.
        maximum_puzzle_id_number (int): Maximum puzzle identification number
    """
    puzzle_id_list[index_to_increment] += 1

    if index_to_increment == 0:
        return

    # This is needed because only last index actually reaches max index.  Previous ones are offset from that
    # since we are doing combinations.
    max_id_offset = len(puzzle_id_list) - index_to_increment - 1
    if puzzle_id_list[index_to_increment] > maximum_puzzle_id_number - max_id_offset:
        # Recurse and increment previous then come back up the stack
        _increment_previous_id(puzzle_id_list, index_to_increment - 1, maximum_puzzle_id_number)
        puzzle_id_list[index_to_increment] = puzzle_id_list[index_to_increment - 1] + 1
    return


def _write_progress_file(progress_filename, puzzle_id_list):
    """
    Writes the puzzle ID list to the specified output file.

    Note: This function takes care of creating the output directory as well.

    Args:
        progress_filename (str): Name and path of the progress file to be created.
        puzzle_id_list (List[int]): Identification of the next puzzle IDs to be run through the solver
    """
    # Create the
    progress_file_dir = os.path.dirname(progress_filename)
    if not os.path.exists(progress_file_dir):
        logging.info("Creating progress file directory: \"progress_file_dir\"")
        os.makedirs(progress_filename)
    with open(progress_filename, 'w') as progress_file:
        for idx, puzzle_id in enumerate(puzzle_id_list):
            if idx > 0:
                progress_file.write(',')
            progress_file.write(str(puzzle_id))


if __name__ == '__main__':
    config.setup_logging()

    run_comparison_driver()
