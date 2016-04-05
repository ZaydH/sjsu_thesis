"""Main Puzzle Solver Driver

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""
import random

# noinspection PyUnresolvedReferences
from hammoudeh_puzzle_solver.puzzle_importer import Puzzle, PuzzleTester, PuzzleType
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePiece
from paikin_tal_solver.solver import PaikinTalSolver, PickleHelper

# Select whether to display the images after reconstruction
DISPLAY_IMAGES = True
DEFAULT_PUZZLE_TYPE = PuzzleType.type2
DEFAULT_PUZZLE_PIECE_WIDTH = 28

# When true, all asymmetric distances are recalculated.
RECALCULATE_DISTANCES = True
USE_KNOWN_PUZZLE_DIMENSIONS = False


def paikin_tal_driver(image_files, puzzle_type=None, piece_width=None):
    """
    Runs the Paikin and Tal image solver.

    Args:
        image_files ([str]): An array of one or more image file path(s).
        puzzle_type (Optional PuzzleType): Type of the puzzle to solve
        piece_width (Optional int): Width of a puzzle piece in pixels.
    """

    puzzles = []  # Stores all of the puzzles.
    combined_pieces = []  # Merge all the pieces together
    local_puzzle_type = puzzle_type if puzzle_type is not None else DEFAULT_PUZZLE_TYPE
    local_piece_width = piece_width if piece_width is not None else DEFAULT_PUZZLE_PIECE_WIDTH
    if RECALCULATE_DISTANCES:
        for i in range(0, len(image_files)):
            # Define the identification number of the first piece
            starting_piece_id = len(combined_pieces)

            # Build the puzzle and add it to the list of puzzles
            new_puzzle = Puzzle(i, image_files[i], local_piece_width, starting_piece_id)
            puzzles.append(new_puzzle)

            # Concatenate to the list of all pieces.
            combined_pieces += puzzles[i].pieces
    # For good measure, shuffle the pieces
    random.shuffle(combined_pieces)

    # Extract the filename of the image(s)
    pickle_file_name = "start"
    for i in range(0, len(image_files)):
        # Get the root of the filename (i.e. without path and file extension
        _, img_root_filename = extract_image_filename_and_file_extension(image_files[i])

        # Append the file name to the information
        pickle_file_name += "_" + img_root_filename
    pickle_file_name += "_type_" + str(local_puzzle_type.value) + ".pk"

    # Determine whether to calculate the interpiece distances or import via pickle
    if RECALCULATE_DISTANCES:
        # Select whether or not to use fixed puzzle dimensions
        puzzle_dimensions = None
        if USE_KNOWN_PUZZLE_DIMENSIONS and len(images) == 1:
            puzzle_dimensions = puzzles[0].grid_size

        # Create the Paikin Tal Solver
        paikin_tal_solver = PaikinTalSolver(len(image_files), combined_pieces,
                                            PuzzlePiece.calculate_asymmetric_distance, local_puzzle_type,
                                            fixed_puzzle_dimensions=puzzle_dimensions)
        # Export the Paikin Tal Object.
        PickleHelper.exporter(paikin_tal_solver, pickle_file_name)
    else:
        print "Beginning import of pickle file: \"" + pickle_file_name + "\"\n\n"
        paikin_tal_solver = PickleHelper.importer(pickle_file_name)
        print "Pickle Import completed.\""

    # Run the Solver
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
            # noinspection PyProtectedMember
            Puzzle.display_image(new_puzzle._img)

        # Store the reconstructed image
        filename = ".\\solved\\reconstructed_type_" + str(paikin_tal_solver.puzzle_type.value) + "_"
        if len(image_files) == 1:
            img_extension, img_root_filename = extract_image_filename_and_file_extension(image_files[0])
            filename += img_root_filename + "." + img_extension
        # Give a generic name if more than one puzzle being solved
        else:
            filename += "puzzle_" + ("%04d" % puzzle_id) + ".jpg"
        new_puzzle.save_to_file(filename)

        # Append the puzzle to the list
        output_puzzles.append(new_puzzle)


def extract_image_filename_and_file_extension(image_filename_and_path):
    """
    Filename Root Extractor

    Given a filename (with a file extension) and a directory, extract just the root of the filename.

    Args:
        image_filename_and_path (String): The path to an image as well as the filename.

    Returns (String): The name of the file without any path information and with the file extension.
    """
    filename_stub = image_filename_and_path[::-1]  # Reverse the string

    # Get the file extension
    file_extension = filename_stub[:filename_stub.index(".")][::-1]
    # Get everything after the before the file extension in the original (unreversed) string
    filename_stub = filename_stub[len(file_extension) + 1:]

    # Get everything after the last slash in the original string
    filename_stub = filename_stub[:filename_stub.index("\\")]

    # Reverse the string again to get the right ordering
    filename_stub = filename_stub[::-1]

    # Return the file extension and the root filename
    return file_extension, filename_stub


if __name__ == "__main__":
    # images = [".\\images\\muffins_300x200.jpg"]
    # paikin_tal_driver(images, PuzzleType.type1, 25)
    # paikin_tal_driver(images, PuzzleType.type2, 25)
    # images = [".\\images\\duck.bmp"]
    # paikin_tal_driver(images, PuzzleType.type1, 25)
    # images = [".\\images\\cat_sleeping_boy.jpg"]
    # paikin_tal_driver(images, PuzzleType.type1, 28)
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\two_faced_cat.jpg"]
    # paikin_tal_driver(images, PuzzleType.type1, 25)
    # paikin_tal_driver(images, PuzzleType.type2, 25)
    # images = [".\\images\\mcgill_20.jpg", ".\\images\\two_faced_cat.jpg", ".\\images\\muffins_300x200.jpg"]
    # paikin_tal_driver(images, PuzzleType.type1, 28)
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\7.jpg", ".\\images\\mcgill_20.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\mcgill_20.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\mcgill_03.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\bgu_805_08.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\bgu_805_10.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    images = [".\\images\\3300_1.jpg"]
    paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\boat_100x100.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 25)
    # images = [".\\images\\che_100x100.gif"]
    # paikin_tal_driver(images, PuzzleType.type1, 25)
    # paikin_tal_driver(images, PuzzleType.type2, 25)
    # images = [".\\images\\bgu_805_08.jpg", ".\\images\\mcgill_20.jpg", ".\\images\\3300_1.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\bgu_805_08.jpg", ".\\images\\mcgill_20.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 28)
