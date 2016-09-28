import logging

from hammoudeh_puzzle import config
from hammoudeh_puzzle import puzzle_importer
from hammoudeh_puzzle.pickle_helper import PickleHelper
from hammoudeh_puzzle.puzzle_importer import Puzzle, PuzzleType
from hammoudeh_puzzle.puzzle_piece import top_level_calculate_asymmetric_distance
from multipuzzle_solver.multipuzzle_solver import MultiPuzzleSolver

_RECALCULATE_DISTANCES = True


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
    if _RECALCULATE_DISTANCES:
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


if __name__ == "__main__":

    # Setup the logger
    config.setup_logging()

    images = ["7.jpg", "dandelion_pixabay.jpg", "beautiful-1168104_640.jpg"]
    run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    images = ["book_tunnel_pixabay.jpg", "duck.bmp", "7.jpg"]
    run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    images = ["book_tunnel_pixabay.jpg", "duck.bmp", "7.jpg", "mcgill_03.jpg"]
    run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    images = ["bgu_805_08.jpg", "mcgill_20.jpg"]
    run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    # MultiPuzzleSolver.run_imported_segmentation_round(images, PuzzleType.type2, 1)
    # MultiPuzzleSolver.run_imported_stitching_piece_solving(images, PuzzleType.type2)
    MultiPuzzleSolver.run_imported_similarity_matrix_calculation(images, PuzzleType.type2)

    images = ["bgu_805_08.jpg", "mcgill_20.jpg", "3300_1.jpg"]
    # run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
    MultiPuzzleSolver.run_imported_similarity_matrix_calculation(images, PuzzleType.type2)
