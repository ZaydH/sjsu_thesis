import numpy

from hammoudeh_puzzle_solver.puzzle_importer import Puzzle, PuzzleTester
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePiece
from paikin_tal_solver.solver import PaikinTalSolver


def paikin_tal_driver(image_files):
    """
    Runs the Paikin and Tal image solver.

    Args:
        image_files ([str]): An array of one or more image file path(s).
    """

    # numb_puzzles = len(image_files)  # Extract the number of puzzles
    # puzzles = []  # Stores all of the puzzles.
    # combined_pieces = []  # Merge all the pieces together
    # for i in range(0, numb_puzzles):
    #     puzzles.append(Puzzle(i, image_files[i]))
    #     # Concatenate to the list of all pieces.
    #     combined_pieces += puzzles[i].pieces

    # Puzzle.display_image(numpy.rot90(puzzles[0]._img, 2))
    # puzzles[0]._assign_all_pieces_to_original_location()
    # puzzles[0].randomize_puzzle_piece_locations()
    # Puzzle.reconstruct_from_pieces(puzzles[0]._pieces)
    # puzzles[0].randomize_puzzle_piece_rotations()
    # Puzzle.reconstruct_from_pieces(puzzles[0]._pieces)

    # Use the tester puzzle initially.
    numb_puzzles = 1
    puzzles = [PuzzleTester.build_dummy_puzzle()]
    combined_pieces = puzzles[0].pieces

    # Create the Paikin Tal Solver
    paikin_tal_solver = PaikinTalSolver(numb_puzzles, combined_pieces, PuzzlePiece.calculate_asymmetric_distance)

    # Run the solver
    paikin_tal_solver.run()

    # Get the results
    (paikin_tal_results, _) = paikin_tal_solver.get_solved_puzzles()

    # Print the Paikin Tal Solver Results
    output_puzzles = []
    for puzzle_pieces in paikin_tal_results:
        output_puzzles.append(Puzzle.reconstruct_from_pieces(puzzle_pieces))
        last_puzzle = len(output_puzzles) - 1
        output_puzzles[last_puzzle].display_image()

if __name__ == "__main__":
    images = [".\images\muffins_300x200.jpg"]
    paikin_tal_driver(images)
