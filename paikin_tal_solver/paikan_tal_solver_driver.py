"""Main Puzzle Solver Driver

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""
# noinspection PyUnresolvedReferences
import logging
import os
import time

from hammoudeh_puzzle import config
from hammoudeh_puzzle import puzzle_importer
from hammoudeh_puzzle.pickle_helper import PickleHelper
from hammoudeh_puzzle.puzzle_importer import Puzzle, PuzzleType, PuzzleSolver
from hammoudeh_puzzle.puzzle_piece import top_level_calculate_asymmetric_distance
from hammoudeh_puzzle.solver_helper import print_elapsed_time
from paikin_tal_solver.solver import PaikinTalSolver

# Select whether to display the images after reconstruction
DISPLAY_IMAGES = False
DEFAULT_PUZZLE_TYPE = PuzzleType.type2

# Criteria for modifying the low
FORCE_RECALCULATE_DISTANCES = False
SKIP_PLACEMENT = False

USE_KNOWN_PUZZLE_DIMENSIONS = False

_PERFORM_ASSERT_CHECKS = config.PERFORM_ASSERT_CHECKS


def paikin_tal_driver(img_files, puzzle_type, piece_width):
    """
    Runs the Paikin and Tal image solver.

    Args:
        img_files ([str]): An array of one or more image file path(s).
        puzzle_type (Optional PuzzleType): Type of the puzzle to solve
        piece_width (Optional int): Width of a puzzle piece in pixels.
    """

    image_filenames = config.add_image_folder_path(img_files)

    # Print the names of the images being solved:
    logging.info("Standard Paikin & Tal Driver")
    puzzle_importer.log_puzzle_filenames(image_filenames)

    # When skipping placement, simply import the solved results.
    if SKIP_PLACEMENT:
        paikin_tal_solver = PaikinTalSolver.pickle_import_after_standard_run_placement(image_filenames,
                                                                                       puzzle_type)
    else:
        paikin_tal_solver = run_paikin_tal_solver(image_filenames, puzzle_type, piece_width)

    # Get the results
    paikin_tal_solver.segment(color_segments=True)
    (pieces_partitioned_by_puzzle_id, _) = paikin_tal_solver.get_solved_puzzles()

    timestamp = time.time()
    Puzzle.output_results_information_and_puzzles(PuzzleSolver.PaikinTal, image_filenames, paikin_tal_solver,
                                                  pieces_partitioned_by_puzzle_id, timestamp)


def run_paikin_tal_solver(image_filenames, puzzle_type, piece_width):
    """
    Paikin & Tal Solver

    This function takes a set of inputs and runs the Paikin and Tal solver.  It can be sped-up by importing
    the calculations of distances from existing Pickle files.

    Args:
        image_filenames (List[String]): Path to the image files used to create the puzzles
        puzzle_type (PuzzleType): Type of the puzzle to be solved
        piece_width (int): Width/length of all puzzle pieces

    Returns (PaikinTalSolver):
        Solved Paikin & Tal result.

    """
    pickle_filename = PickleHelper.build_filename(PaikinTalSolver.POST_INITIAL_CONSTRUCTOR_PICKLE_FILE_DESCRIPTOR,
                                                  image_filenames, puzzle_type)

    # Optionally import the images from disk
    if FORCE_RECALCULATE_DISTANCES or not os.path.exists(pickle_filename):
        combined_pieces, puzzles = Puzzle.get_combined_pieces_multiple_images(image_filenames, piece_width)

        # Select whether or not to use fixed puzzle dimensions
        puzzle_dimensions = puzzles[0].grid_size if USE_KNOWN_PUZZLE_DIMENSIONS and len(images) == 1 else None

        # Create the Paikin Tal Solver
        logging.info("Beginning calculating of Paikin & Tal inter-piece distance information")
        start_time = time.time()
        paikin_tal_solver = PaikinTalSolver(combined_pieces, top_level_calculate_asymmetric_distance,
                                            len(image_filenames), puzzle_type,
                                            fixed_puzzle_dimensions=puzzle_dimensions, image_filenames=image_filenames)
        print_elapsed_time(start_time, "Paikin & Tal inter-piece distance calculations")

    else:
        paikin_tal_solver = PaikinTalSolver.pickle_import_after_initial_construction(image_filenames, puzzle_type)

    # Run the Solver
    paikin_tal_solver.run_standard()

    # Export the solved results
    return paikin_tal_solver


if __name__ == "__main__":

    # Setup the logger
    config.setup_logging()

    # Select the files to parse

    # # PaikinTalSolver.use_best_buddy_placer = False
    images = ["muffins_300x200.jpg"]
    paikin_tal_driver(images, PuzzleType.type1, 25)
    paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    # PaikinTalSolver.use_best_buddy_placer = True
    # paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # images = ["duck.bmp"]
    # paikin_tal_driver(images, PuzzleType.type1, 25)
    # images = ["cat_sleeping_boy.jpg"]
    # paikin_tal_driver(images, PuzzleType.type1, config.DEFAULT_PIECE_WIDTH)
    # paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    # images = ["kitten_white_background.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    # images = [["book_tunnel_pixabay.jpg"],
    #           ["dessert_pixabay.jpg"],
    #           ["dandelion_pixabay.jpg"],
    #           ["primula_pixabay.jpg"],
    #           ["small_pink_flowers_pixabay.jpg"]]
    # for image_arr in images:
    #     paikin_tal_driver(image_arr, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    # images = ["two_faced_cat.jpg"]
    # paikin_tal_driver(images, PuzzleType.type1, 25)
    # paikin_tal_driver(images, PuzzleType.type2, 25)
    # images = ["7.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # images = ["mcgill_20.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # images = ["7.jpg", "mcgill_20.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # images = ["mcgill_20.jpg", "two_faced_cat.jpg", "muffins_300x200.jpg"]
    # paikin_tal_driver(images, PuzzleType.type1, config.DEFAULT_PIECE_WIDTH)
    # paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # images = ["mcgill_03.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # images = ["bgu_805_08.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # images = ["bgu_805_10.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # images = ["3300_1.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # images = ["boat_100x100.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 25)
    # images = ["che_100x100.jpg"]
    # paikin_tal_driver(images, PuzzleType.type1, 25)
    # paikin_tal_driver(images, PuzzleType.type2, 25)
    #
    # images = ["bgu_805_08.jpg", "mcgill_20.jpg", "3300_1.jpg"]
    # PaikinTalSolver._CLEAR_BEST_BUDDY_HEAP_ON_SPAWN = True
    # InterPieceDistance._USE_ONLY_NEIGHBORS_FOR_STARTING_PIECE_TOTAL_COMPATIBILITY = True
    # InterPieceDistance._NEIGHBOR_COMPATIBILITY_SCALAR = 1
    # paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    # # # images = ["bgu_805_08.jpg", "mcgill_20.jpg", "3300_1.jpg"]
    images = ["bgu_805_08.jpg", "mcgill_20.jpg"]
    paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    images = ["bgu_805_08.jpg", "mcgill_20.jpg", "3300_1.jpg"]
    paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    # images = ["bgu_805_08.jpg", "mcgill_20.jpg", "3300_1.jpg"]
    # PaikinTalSolver._CLEAR_BEST_BUDDY_HEAP_ON_SPAWN = True
    # InterPieceDistance._USE_ONLY_NEIGHBORS_FOR_STARTING_PIECE_TOTAL_COMPATIBILITY = False
    # InterPieceDistance._NEIGHBOR_COMPATIBILITY_SCALAR = 4
    # paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    #
    # images = ["bgu_805_08.jpg", "mcgill_20.jpg", "3300_1.jpg"]
    # PaikinTalSolver._CLEAR_BEST_BUDDY_HEAP_ON_SPAWN = False
    # InterPieceDistance._USE_ONLY_NEIGHBORS_FOR_STARTING_PIECE_TOTAL_COMPATIBILITY = False
    # InterPieceDistance._NEIGHBOR_COMPATIBILITY_SCALAR = 1
    # paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    #
    # images = ["bgu_805_08.jpg", "mcgill_20.jpg", "3300_1.jpg"]
    # PaikinTalSolver._CLEAR_BEST_BUDDY_HEAP_ON_SPAWN = False
    # InterPieceDistance._USE_ONLY_NEIGHBORS_FOR_STARTING_PIECE_TOTAL_COMPATIBILITY = False
    # InterPieceDistance._NEIGHBOR_COMPATIBILITY_SCALAR = 4
    # paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    # images = ["bgu_805_08.jpg", "mcgill_20.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
