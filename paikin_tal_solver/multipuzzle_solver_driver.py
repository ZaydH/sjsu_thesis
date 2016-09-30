import logging
import os
import random

import time

from hammoudeh_puzzle import config
from hammoudeh_puzzle import puzzle_importer
from hammoudeh_puzzle.pickle_helper import PickleHelper
from hammoudeh_puzzle.puzzle_importer import Puzzle, PuzzleType
from hammoudeh_puzzle.puzzle_piece import top_level_calculate_asymmetric_distance
from multipuzzle_solver.multipuzzle_solver import MultiPuzzleSolver

_FORCE_RECALCULATE_DISTANCES = False


def run_multipuzzle_solver_driver(image_files, puzzle_type, piece_width):
    """
    Runs the multipuzzle solver on a set of images.

    Args:
        image_files (List[str]): List of puzzle file paths.
        puzzle_type (PuzzleType): Type of the puzzle to solve.
        piece_width (int): Puzzle piece width in number of pixels
    """

    image_filenames = config.add_image_folder_path(image_files)

    logging.info("Starting Multipuzzle Solver Driver.")
    puzzle_importer.log_puzzle_filenames(image_filenames)

    multipuzzle_solver = build_multipuzzle_solver(image_filenames, puzzle_type, piece_width)

    # Run the solver
    multipuzzle_solver.run()


def build_multipuzzle_solver(image_filenames, puzzle_type, piece_width):
    """
    Build the multipuzzle solver object.

    Args:
        image_filenames (List[str]): List of puzzle file paths.
        puzzle_type (PuzzleType): Type of the puzzle to solve.
        piece_width (int): Puzzle piece width in number of pixels

    Returns (MultiPuzzleSolver): The multipuzzle solver object built from the input image files.
    """

    pieces, puzzles = Puzzle.get_combined_pieces_multiple_images(image_filenames, piece_width)

    pickle_filename = PickleHelper.build_filename("multipuzzle_distances", image_filenames, puzzle_type)

    # Initialize the distance information
    if _FORCE_RECALCULATE_DISTANCES or not os.path.exists(pickle_filename):
        multipuzzle_solver = MultiPuzzleSolver(image_filenames, pieces, top_level_calculate_asymmetric_distance,
                                               puzzle_type)
        if PickleHelper.PICKLE_ENABLED:
            PickleHelper.exporter(multipuzzle_solver, pickle_filename)
        return multipuzzle_solver

    # Read the pickle information from the
    else:
        multipuzzle_solver = PickleHelper.importer(pickle_filename)
        multipuzzle_solver.reset_timestamp()
        return multipuzzle_solver


def test_random_mcgill():
    """
    Randomly selects a set of 805 piece puzzles and runs the multipuzzle solver on them.
    """

    # Improve the seed quality
    random.seed(time.time())

    while True:
        # Get a number of puzzles
        minimum_numb_puzzles = 2
        maximum_numb_puzzles = 3
        numb_puzzles = random.randint(minimum_numb_puzzles, maximum_numb_puzzles)

        logging.info("Number of 805 Piece Input Puzzles: %d" % numb_puzzles)

        # Build the puzzle list
        DIRECTORY_805_PIECE_IMAGES = "805_pieces//"
        IMAGE_FILE_EXTENSION = ".jpg"
        images_file_list = []
        while len(images_file_list) < numb_puzzles:
            filename = DIRECTORY_805_PIECE_IMAGES + str(random.randint(1, 20)) + IMAGE_FILE_EXTENSION
            # Ensure no duplicate puzzles
            if filename not in images_file_list:
                images_file_list.append(filename)

        # Run the solver
        run_multipuzzle_solver_driver(images_file_list, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)


if __name__ == "__main__":

    # Setup the logger
    config.setup_logging()
    #
    # images = ["7.jpg", "dandelion_pixabay.jpg", "beautiful-1168104_640.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    # images = ["book_tunnel_pixabay.jpg", "duck.bmp", "7.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # MultiPuzzleSolver.run_imported_segmentation_experiment(images, PuzzleType.type2, segmentation_round_numb=1)
    # MultiPuzzleSolver.run_imported_segmentation_round(images, PuzzleType.type2, segmentation_round_numb=1)
    #
    # test_random_mcgill()

    # images = ["book_tunnel_pixabay.jpg", "duck.bmp", "7.jpg", "mcgill_03.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    # images = ["bgu_805_08.jpg", "mcgill_20.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # MultiPuzzleSolver.run_imported_hierarchical_clustering(images, PuzzleType.type2)
    # MultiPuzzleSolver.run_imported_select_starting_pieces(images, PuzzleType.type2)
    # MultiPuzzleSolver.run_imported_final_puzzle_solving(images, PuzzleType.type2)

    # MultiPuzzleSolver.run_imported_segmentation_round(images, PuzzleType.type2, 1)
    # MultiPuzzleSolver.run_imported_stitching_piece_solving(images, PuzzleType.type2)
    # MultiPuzzleSolver.run_imported_similarity_matrix_calculation(images, PuzzleType.type2)

    # images = ["7.jpg", "dandelion_pixabay.jpg", "beautiful-1168104_640.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    # images = ["book_tunnel_pixabay.jpg", "duck.bmp", "7.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    # images = ["book_tunnel_pixabay.jpg", "duck.bmp", "7.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    images = ["bgu_805_08.jpg", "mcgill_20.jpg"]
    run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    images = ["bgu_805_08.jpg", "mcgill_20.jpg", "3300_1.jpg"]
    run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    images = ["805_pieces//2.jpg", "805_pieces//1.jpg"]
    run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    images = ["805_pieces//9.jpg", "805_pieces//10.jpg"]
    run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    images = ["805_pieces//5.jpg", "805_pieces//20.jpg", "805_pieces//1.jpg"]
    run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    images = ["805_pieces//8.jpg", "805_pieces//18.jpg", "805_pieces//15.jpg"]
    run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    images = ["805_pieces//2.jpg", "805_pieces//13.jpg", "805_pieces//14.jpg", "805_pieces//19.jpg"]
    run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    test_random_mcgill()
