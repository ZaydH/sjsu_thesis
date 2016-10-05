import logging
import os
import random

from hammoudeh_puzzle import config
from hammoudeh_puzzle import puzzle_importer
from hammoudeh_puzzle.pickle_helper import PickleHelper
from hammoudeh_puzzle.puzzle_importer import Puzzle, PuzzleType
from hammoudeh_puzzle.puzzle_piece import top_level_calculate_asymmetric_distance
from multipuzzle_solver.multipuzzle_solver import MultiPuzzleSolver

_FORCE_RECALCULATE_DISTANCES = False
_POST_INITIAL_CONSTRUCTION_PICKLE_EXPORT = True


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

        if _POST_INITIAL_CONSTRUCTION_PICKLE_EXPORT:
            PickleHelper.exporter(multipuzzle_solver, pickle_filename)
        return multipuzzle_solver

    # Read the pickle information from the
    else:
        multipuzzle_solver = PickleHelper.importer(pickle_filename)
        multipuzzle_solver.reset_timestamp()
        return multipuzzle_solver


def test_random_pomeranz_805_pieces_images():
    """
    Randomly selects a set of 805 piece puzzles and runs the multipuzzle solver on them.
    """

    # Improve the seed quality
    while True:
        # Get a number of puzzles
        minimum_numb_puzzles = 2
        maximum_numb_puzzles = 3
        numb_puzzles = random.randint(minimum_numb_puzzles, maximum_numb_puzzles)

        logging.info("Number of 805 Piece Input Puzzles: %d" % numb_puzzles)

        # Build the puzzle list
        images_file_list = []
        while len(images_file_list) < numb_puzzles:
            filename = config.get_random_pomeranz_805_piece_image()
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
    # MultiPuzzleSolver.run_imported_final_puzzle_solving(images, PuzzleType.type2)

    # images = ["book_tunnel_pixabay.jpg", "duck.bmp", "7.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # MultiPuzzleSolver.run_imported_segmentation_experiment(images, PuzzleType.type2, segmentation_round_numb=1)
    # MultiPuzzleSolver.run_imported_segmentation_round(images, PuzzleType.type2, segmentation_round_numb=1)
    #
    # test_random_pomeranz_805_pieces_images()

    # images = ["book_tunnel_pixabay.jpg", "duck.bmp", "7.jpg", "mcgill_03.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    # images = ["pomeranz_805_08.jpg", "mcgill_20.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # MultiPuzzleSolver.run_imported_hierarchical_clustering(images, PuzzleType.type2)
    # MultiPuzzleSolver.run_imported_select_starting_pieces(images, PuzzleType.type2)
    # MultiPuzzleSolver.run_imported_final_puzzle_solving(images, PuzzleType.type2)

    # MultiPuzzleSolver.run_imported_segmentation_round(images, PuzzleType.type2, 1)
    # MultiPuzzleSolver.run_imported_stitching_piece_solving(images, PuzzleType.type2)
    # MultiPuzzleSolver.run_imported_similarity_matrix_calculation(images, PuzzleType.type2)

    # images = ["7.jpg", "dandelion_pixabay.jpg", "beautiful-1168104_640.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # 
    # images = ["book_tunnel_pixabay.jpg", "duck.bmp", "7.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # 
    # images = ["pomeranz_805_08.jpg", "mcgill_20.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # 
    # images = ["pomeranz_805_08.jpg", "mcgill_20.jpg", "3300_1.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # 
    # images = ["pomeranz_805//2.jpg", "pomeranz_805//1.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # 
    # images = ["pomeranz_805//9.jpg", "pomeranz_805//10.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # 
    # images = ["pomeranz_805//5.jpg", "pomeranz_805//20.jpg", "pomeranz_805//1.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    # 
    # images = ["pomeranz_805//8.jpg", "pomeranz_805//18.jpg", "pomeranz_805//15.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    # images = [config.build_cho_432_piece_filename(18)]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    images = ["primula_pixabay.jpg",
              "dandelion_pixabay.jpg",
              config.build_mcgill_540_piece_filename(15),
              config.build_mcgill_540_piece_filename(11),
              config.build_cho_432_piece_filename(18),
              config.build_pomeranz_805_piece_filename(8),
              config.build_pomeranz_805_piece_filename(10),
              config.build_pomeranz_805_piece_filename(13),
              config.build_pomeranz_805_piece_filename(14),
              config.build_pomeranz_805_piece_filename(19)]
    run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    images = ["primula_pixabay.jpg",
              "dandelion_pixabay.jpg",
              config.build_mcgill_540_piece_filename(15),
              config.build_mcgill_540_piece_filename(11),
              config.build_cho_432_piece_filename(18),
              config.build_pomeranz_805_piece_filename(8),
              config.build_pomeranz_805_piece_filename(10),
              config.build_pomeranz_805_piece_filename(13),
              config.build_pomeranz_805_piece_filename(14),
              config.build_pomeranz_805_piece_filename(19),
              "3300_1.jpg"]
    run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    # # This has issues in reconstruction.
    # images = ["pomeranz_805//2.jpg", "pomeranz_805//13.jpg", "pomeranz_805//14.jpg", "pomeranz_805//19.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    # test_random_pomeranz_805_pieces_images()
