import logging
import os
import random

from hammoudeh_puzzle import config
from hammoudeh_puzzle.puzzle_importer import PuzzleType
from multipuzzle_solver_driver import run_multipuzzle_solver_driver
from paikin_tal_solver_driver import run_paikin_tal_driver

_PROGRESS_TRACKING_FOLDER = "." + os.sep + "progress_tracker" + os.sep
_IMAGE_ID_FILE_SEPARATOR = ","

PUZZLE_TYPE = PuzzleType.type2
PIECE_WIDTH = 28  # pixels


def run_comparison_driver():
    """
    Primary function used by the comparison driver. It runs all of the planned comparison tests.
    """
    # for numb_simultaneous_puzzles in [1, 2]:
    #     _perform_all_combinations_comparison_puzzle_solving("805_piece_poermanz", numb_simultaneous_puzzles,
    #                                                         config.MINIMUM_POMERANZ_805_PIECE_IMAGE_NUMBER,
    #                                                         config.MAXIMUM_POMERANZ_805_PIECE_IMAGE_NUMBER,
    #                                                         config.build_pomeranz_805_piece_filename)

    numb_simultaneous_puzzles = [2, 3, 4, 5]
    numb_iterations = [50, 20, 10, 5]
    for i in xrange(0, len(numb_simultaneous_puzzles)):
        _perform_random_comparison_puzzle_solving("805_piece_pomeranz", numb_simultaneous_puzzles[i],
                                                  numb_iterations[i], config.MINIMUM_POMERANZ_805_PIECE_IMAGE_NUMBER,
                                                  config.MAXIMUM_POMERANZ_805_PIECE_IMAGE_NUMBER,
                                                  config.build_pomeranz_805_piece_filename)


def _perform_all_combinations_comparison_puzzle_solving(dataset_name, numb_simultaneous_puzzles, minimum_image_number,
                                                        maximum_image_number, build_image_filename_function):
    """
    Runs all possible combinations of the Paikin and Tal solver of the specified size.  The dataset selected is fully
    configurable and can support arbitrary numbers of puzzles simultaneously.

    Args:
        dataset_name (str): Unique name for the dataset to be used to create the progress tracking file.
        numb_simultaneous_puzzles (int): Number of puzzles that will be analyzed simultaneously by the solver
        minimum_image_number (int): Minimum number (inclusive) for images in the dataset
        maximum_image_number (int): Maximum number (inclusive) for images in the dataset
        build_image_filename_function: Function used to get the name of the image file from the image number
    """
    if numb_simultaneous_puzzles < 1:
        raise ValueError("The number of simultaneous puzzle to solve must be greater than or equal to 1.")

    if maximum_image_number < minimum_image_number:
        raise ValueError("The maximum image number is less than the minimum image number.")

    if numb_simultaneous_puzzles > maximum_image_number - minimum_image_number + 1:
        raise ValueError("The number of simultaneous images exceeds the specified size of the dataset.")

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
        if puzzle_id_list[0] > maximum_image_number - numb_simultaneous_puzzles + 1:
            return

        # Build the image file
        image_filenames = [build_image_filename_function(puzzle_id) for puzzle_id in puzzle_id_list]

        # Run the Multipuzzle Solver.  Paikin Tal will then use MultiPuzzle's pickle export.
        _perform_multipuzzle_solver_and_paikin_solvers(image_filenames, PUZZLE_TYPE, PIECE_WIDTH)

        puzzle_id_list = increment_puzzle_id_list(puzzle_id_list, maximum_image_number)
        # Make the tracking file.
        _write_progress_file(progress_filename, puzzle_id_list)


def _perform_random_comparison_puzzle_solving(dataset_name, numb_simultaneous_puzzles, max_numb_iterations,
                                              minimum_image_number, maximum_image_number, build_image_filename_function):
    """
    Runs all possible combinations of the Paikin and Tal solver of the specified size.  The dataset selected is fully
    configurable and can support arbitrary numbers of puzzles simultaneously.

    Args:
        dataset_name (str): Unique name for the dataset to be used to create the progress tracking file.
        numb_simultaneous_puzzles (int): Number of puzzles that will be analyzed simultaneously by the solver
        max_numb_iterations (int): Number of random iterations to run.
        minimum_image_number (int): Minimum number (inclusive) for images in the dataset
        maximum_image_number (int): Maximum number (inclusive) for images in the dataset
        build_image_filename_function: Function used to get the name of the image file from the image number
    """
    if numb_simultaneous_puzzles < 1:
        raise ValueError("The number of simultaneous puzzle to solve must be greater than or equal to 1.")

    if maximum_image_number < minimum_image_number:
        raise ValueError("The maximum image number is less than the minimum image number.")

    if numb_simultaneous_puzzles > maximum_image_number - minimum_image_number + 1:
        raise ValueError("The number of simultaneous images exceeds the specified size of the dataset.")

    if max_numb_iterations < 1:
        raise ValueError("At least a single iteration must be run")

    while True:
        progress_filename = _build_progress_filename(dataset_name, numb_simultaneous_puzzles)

        # If file does not exist, then create the list of images
        if not os.path.exists(progress_filename):
            iteration_count = 0
        else:
            # Get the next puzzle from the file
            with open(progress_filename, 'r') as progress_file:
                iteration_count = int(progress_file.readline())

        # If all images already completed, then exit.
        if iteration_count >= max_numb_iterations:
            return

        # Build the image file
        img_ids = []
        while len(img_ids) < numb_simultaneous_puzzles:
            temp_id = random.randint(minimum_image_number, maximum_image_number)
            if temp_id not in img_ids:
                img_ids.append(temp_id)
        image_filenames = [build_image_filename_function(temp_id) for temp_id in img_ids]

        # Run the Multipuzzle Solver.  Paikin Tal will then use MultiPuzzle's pickle export.
        _perform_multipuzzle_solver_and_paikin_solvers(image_filenames, PUZZLE_TYPE, PIECE_WIDTH)

        # Make the tracking file.
        _write_progress_file(progress_filename, [iteration_count])
        iteration_count += 1


def _perform_multipuzzle_solver_and_paikin_solvers(image_filenames, puzzle_type, piece_width):
    """
    Runs the Paikin and Tal and Multipuzzle Solver with the specified conditions

    Args:
        image_filenames (List[str]): Name and path of the images to run in the solver
        puzzle_type (PuzzleType): Puzzle type to be run
        piece_width (int): Width of the puzzle in pieces.
    """
    # Run the Multipuzzle Solver.  Paikin Tal will then use MultiPuzzle's pickle export.
    run_multipuzzle_solver_driver(image_filenames, puzzle_type, piece_width)
    run_paikin_tal_driver(image_filenames, puzzle_type, piece_width)


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
    # Create the progress file directory if needed
    progress_file_dir = os.path.dirname(os.path.abspath(progress_filename))
    if not os.path.exists(progress_file_dir):
        logging.info("Creating progress file directory: \"" + progress_file_dir + "\"")
        os.makedirs(progress_filename)

    # Write the progress file itself.
    with open(progress_filename, 'w') as progress_file:
        for idx, puzzle_id in enumerate(puzzle_id_list):
            if idx > 0:
                progress_file.write(',')
            progress_file.write(str(puzzle_id))


if __name__ == '__main__':
    config.setup_logging()

    config.IS_SOLVER_COMPARISON_RUNNING = True

    run_comparison_driver()
