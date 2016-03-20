"""Main Puzzle Solver Driver

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""
import random

# noinspection PyUnresolvedReferences
from hammoudeh_puzzle_solver.puzzle_importer import Puzzle, PuzzleTester, PuzzleType
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePiece
from paikin_tal_solver.solver import PaikinTalSolver

# Select whether to display the images after reconstruction
DISPLAY_IMAGES = True
DEFAULT_PUZZLE_TYPE = PuzzleType.type1
DEFAULT_PUZZLE_PIECE_WIDTH = 25


def paikin_tal_driver(image_files, puzzle_type=None, piece_width=None):
    """
    Runs the Paikin and Tal image solver.

    Args:
        image_files ([str]): An array of one or more image file path(s).
        puzzle_type (Optional PuzzleType): Type of the puzzle to solve
        piece_width (Optional int): Width of a puzzle piece in pixels.
    """

    numb_puzzles = len(image_files)  # Extract the number of puzzles
    puzzles = []  # Stores all of the puzzles.
    combined_pieces = []  # Merge all the pieces together
    local_piece_width = piece_width if piece_width is not None else DEFAULT_PUZZLE_PIECE_WIDTH
    for i in range(0, numb_puzzles):
        new_puzzle = Puzzle(i, image_files[i], local_piece_width)
        puzzles.append(new_puzzle)
        # Concatenate to the list of all pieces.
        combined_pieces += puzzles[i].pieces
    # For good measure, shuffle the pieces
    random.shuffle(combined_pieces)

    # Puzzle.display_image(numpy.rot90(puzzles[0]._img, 2))
    # puzzles[0]._assign_all_pieces_to_original_location()
    # puzzles[0].randomize_puzzle_piece_locations()
    # Puzzle.reconstruct_from_pieces(puzzles[0]._pieces)
    # puzzles[0].randomize_puzzle_piece_rotations()
    # Puzzle.reconstruct_from_pieces(puzzles[0]._pieces)

    # # Use the tester puzzle initially.
    # numb_puzzles = 1
    # puzzles = [PuzzleTester.build_dummy_puzzle()]
    # combined_pieces = puzzles[0].pieces

    # Create the Paikin Tal Solver
    local_puzzle_type = puzzle_type if puzzle_type is not None else DEFAULT_PUZZLE_TYPE
    paikin_tal_solver = PaikinTalSolver(numb_puzzles, combined_pieces,
                                        PuzzlePiece.calculate_asymmetric_distance, local_puzzle_type)

    # Run the solver
    paikin_tal_solver.run()

    # Get the results
    (paikin_tal_results, _) = paikin_tal_solver.get_solved_puzzles()

    # Print the Paikin Tal Solver Results
    output_puzzles = []
    for puzzle_pieces in paikin_tal_results:
        # Get the first piece of the puzzle and extract information on it.
        first_piece = puzzle_pieces[0]
        puzzle_id = first_piece.puzzle_id

        # Reconstuct the puzzle
        new_puzzle = Puzzle.reconstruct_from_pieces(puzzle_pieces, puzzle_id)

        # Optionally display the images
        if DISPLAY_IMAGES:
            Puzzle.display_image(new_puzzle._img)
        # Store the reconstructed image
        filename = ".\\solved\\reconstructed_type_" + str(local_puzzle_type.value) + "_puzzle_" + \
                   format(puzzle_id, '04') + ".jpg"
        new_puzzle.save_to_file(filename)

        # Append the puzzle to the list
        output_puzzles.append(new_puzzle)

if __name__ == "__main__":
    # images = [".\\images\\muffins_300x200.jpg"]
    #paikin_tal_driver(images)
    images = [".\\images\\duck.bmp"]
    paikin_tal_driver(images)
    # images = [".\\images\\boat_100x100.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 25)
    # images = [".\\images\\che_100x100.gif"]
    # paikin_tal_driver(images, PuzzleType.type1, 25)

