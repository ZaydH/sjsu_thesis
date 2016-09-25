import logging

from hammoudeh_puzzle import config
from hammoudeh_puzzle import puzzle_importer
from hammoudeh_puzzle.pickle_helper import PickleHelper
from hammoudeh_puzzle.puzzle_importer import Puzzle, PuzzleType
from hammoudeh_puzzle.puzzle_piece import top_level_calculate_asymmetric_distance
from multipuzzle_solver.multipuzzle_solver import MultiPuzzleSolver

_RECALCULATE_DISTANCES = True


def run_multipuzzle_solver_driver(image_filenames, puzzle_type, piece_width):
    """
    Runs the multipuzzle solver on a set of images.

    Args:
        image_filenames (List[str]): List of puzzle file paths.
        puzzle_type (PuzzleType): Type of the puzzle to solve.
        piece_width (int): Puzzle piece width in number of pixels
    """

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
                                               piece_width)
        if PickleHelper.PICKLE_ENABLED:
            PickleHelper.exporter(multipuzzle_solver, pickle_filename)
        return multipuzzle_solver

    # Read the pickle information from the
    else:
        return PickleHelper.importer(pickle_filename)


if __name__ == "__main__":

    # Setup the logger
    config.setup_logging()

    images = [".\\images\\bgu_805_08.jpg", ".\\images\\mcgill_20.jpg"]
    run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)

    images = [".\\images\\bgu_805_08.jpg", ".\\images\\mcgill_20.jpg", ".\\images\\3300_1.jpg"]
    run_multipuzzle_solver_driver(images, PuzzleType.type2, config.DEFAULT_PIECE_WIDTH)
