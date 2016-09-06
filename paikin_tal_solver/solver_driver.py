"""Main Puzzle Solver Driver

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""
# noinspection PyUnresolvedReferences
import random
import time
import datetime

import os
# noinspection PyUnresolvedReferences
from hammoudeh_puzzle.puzzle_importer import Puzzle, PuzzleTester, PuzzleType, PuzzleResultsCollection, \
    PickleHelper
from hammoudeh_puzzle.puzzle_piece import PuzzlePiece
# noinspection PyUnresolvedReferences
from paikin_tal_solver.inter_piece_distance import InterPieceDistance
from paikin_tal_solver.solver import PaikinTalSolver

os.getcwd()

# Select whether to display the images after reconstruction
DISPLAY_IMAGES = False
DEFAULT_PUZZLE_TYPE = PuzzleType.type2
DEFAULT_PUZZLE_PIECE_WIDTH = 28

# When true, all asymmetric distances are recalculated.
PERFORM_PLACEMENT = True
RECALCULATE_DISTANCES = True
USE_KNOWN_PUZZLE_DIMENSIONS = False

# Defining a directory where pickle files are stored.
PICKLE_DIRECTORY = ".\\pickle_files\\"

def paikin_tal_driver(image_files, puzzle_type=None, piece_width=None):
    """
    Runs the Paikin and Tal image solver.

    Args:
        image_files ([str]): An array of one or more image file path(s).
        puzzle_type (Optional PuzzleType): Type of the puzzle to solve
        piece_width (Optional int): Width of a puzzle piece in pixels.
    """

    # Define the variables needed through the driver
    local_puzzle_type = puzzle_type if puzzle_type is not None else DEFAULT_PUZZLE_TYPE
    local_piece_width = piece_width if piece_width is not None else DEFAULT_PUZZLE_PIECE_WIDTH

    # Print the names of the images being solved:
    print "\n\nNames of the Image Files:"
    for img_file in image_files:
        print "\t%s" % Puzzle.get_filename_without_extension(img_file)
    print "\n"

    # Extract the filename of the image(s)
    pickle_root_filename = ""
    for i in range(0, len(image_files)):
        # Get the root of the filename (i.e. without path and file extension
        img_root_filename = Puzzle.get_filename_without_extension(image_files[i])
        # Append the file name to the information
        pickle_root_filename += "_" + img_root_filename
    pickle_root_filename += "_type_" + str(local_puzzle_type.value) + ".pk"

    # Build the filenames
    pickle_placement_start_filename = PICKLE_DIRECTORY + "start" + pickle_root_filename
    pickle_placement_complete_filename = PICKLE_DIRECTORY + "placed" + pickle_root_filename

    # When skipping placement, simply import the solved results.
    if PERFORM_PLACEMENT:
        paikin_tal_solver = run_paikin_tal_solver(image_files, local_puzzle_type, local_piece_width,
                                                  pickle_placement_start_filename, pickle_placement_complete_filename)
    else:
        print "Importing solved puzzle from pickle file: \"%s\"" % pickle_placement_complete_filename
        paikin_tal_solver = PickleHelper.importer(pickle_placement_complete_filename)
        print "Pickle import of solved puzzle complete."

    # Get the results
    (pieces_partitioned_by_puzzle_id, _) = paikin_tal_solver.get_solved_puzzles()

    # Create a time stamp for the results
    timestamp = time.time()

    # Iterate through all the puzzles.  Reconstruct them and get their accuracies.
    output_puzzles = []
    for puzzle_pieces in pieces_partitioned_by_puzzle_id:
        # Get the first piece of the puzzle and extract information on it.
        first_piece = puzzle_pieces[0]
        puzzle_id = first_piece.puzzle_id

        # Reconstruct the puzzle
        new_puzzle = Puzzle.reconstruct_from_pieces(puzzle_pieces, puzzle_id)

        # Optionally display the images
        if DISPLAY_IMAGES:
            # noinspection PyProtectedMember
            Puzzle.display_image(new_puzzle._img)

        # Store the reconstructed image
        filename_descriptor = "reconstructed"
        if len(image_files) == 1:
            orig_img_filename = image_files[0]
            puzzle_id_filename = None
        else:
            orig_img_filename = None
            puzzle_id_filename = puzzle_id
        filename = Puzzle.make_image_filename(filename_descriptor, Puzzle.OUTPUT_IMAGE_DIRECTORY,
                                              paikin_tal_solver.puzzle_type, timestamp,
                                              orig_img_filename=orig_img_filename, puzzle_id=puzzle_id_filename)
        new_puzzle.save_to_file(filename)

        # Append the puzzle to the list
        output_puzzles.append(new_puzzle)

    # Determine the image filename
    orig_img_filename = image_files[0] if len(image_files) == 1 else None
    # Print the best buddy accuracy information
    paikin_tal_solver.best_buddy_accuracy.print_results()
    paikin_tal_solver.best_buddy_accuracy.output_results_images(output_puzzles, paikin_tal_solver.puzzle_type,
                                                                timestamp, orig_img_filename=orig_img_filename)

    # Build the results information collection
    results_information = PuzzleResultsCollection(pieces_partitioned_by_puzzle_id, image_files)
    # Calculate and print the accuracy results
    results_information.calculate_accuracies(output_puzzles)
    # Print the results to the console
    results_information.print_results()
    # Print the results as image files
    results_information.output_results_images(output_puzzles, paikin_tal_solver.puzzle_type, timestamp)


def run_paikin_tal_solver(image_files, puzzle_type, piece_width, pickle_placement_start_filename,
                          pickle_placement_complete_filename):
    """
    Paikin & Tal Solver

    This function takes a set of inputs and runs the Paikin and Tal solver.  It can be sped-up by importing
    the calculations of distances from existing Pickle files.

    Args:
        image_files (List[String]): Path to the image files used to create the puzzles
        puzzle_type (PuzzleType): Type of the puzzle to be solved
        piece_width (int): Width/length of all puzzle pieces
        pickle_placement_start_filename (String): Filename to export a pickle file after calculating all
         inter-piece distances and before starting placement.  If recalculating distances is disabled, then
         this serves as the filename to import calculated distances from.
        pickle_placement_complete_filename (String): Filename to e

    Returns (PaikinTalSolver):
        Solved Paikin & Tal result.

    """

    puzzles = []  # Stores all of the puzzles.
    combined_pieces = []  # Merge all the pieces together

    # Optionally import the images from disk
    if RECALCULATE_DISTANCES:
        for i in range(0, len(image_files)):
            # Define the identification number of the first piece
            starting_piece_id = len(combined_pieces)

            # Build the puzzle and add it to the list of puzzles
            new_puzzle = Puzzle(i, image_files[i], piece_width, starting_piece_id)
            puzzles.append(new_puzzle)

            # Concatenate to the list of all pieces.
            combined_pieces += puzzles[i].pieces

        # Select whether or not to use fixed puzzle dimensions
        puzzle_dimensions = None
        if USE_KNOWN_PUZZLE_DIMENSIONS and len(images) == 1:
            puzzle_dimensions = puzzles[0].grid_size

        # Create the Paikin Tal Solver
        print "Inter-piece distance calculation started at: " + time.ctime()
        start_time = time.time()
        paikin_tal_solver = PaikinTalSolver(len(image_files), combined_pieces,
                                            PuzzlePiece.calculate_asymmetric_distance, puzzle_type,
                                            fixed_puzzle_dimensions=puzzle_dimensions)
        elapsed_time = time.time() - start_time
        print_elapsed_time(elapsed_time, "inter-piece distance calculation")
        print "\n"  # Add a blank line.
        # Export the Paikin Tal Object.
        PickleHelper.exporter(paikin_tal_solver, pickle_placement_start_filename)
    else:
        print "Beginning import of pickle file: \"" + pickle_placement_start_filename + "\"\n\n"
        paikin_tal_solver = PickleHelper.importer(pickle_placement_start_filename)
        print "Pickle Import completed.\""
        # noinspection PyProtectedMember
        # This recalculate of start piece is included since how start piece is selected is configurable.
        paikin_tal_solver._inter_piece_distance.find_start_piece_candidates()

    # Run the Solver
    print "Placer started at: " + time.ctime()
    start_time = time.time()
    paikin_tal_solver.run()
    elapsed_time = time.time() - start_time
    print_elapsed_time(elapsed_time, "placement")

    # Export the completed solver results
    start_time = time.time()
    formatted_time = datetime.datetime.fromtimestamp(start_time).strftime('%Y.%m.%d_%H.%M.%S')
    print "Beginning pickle export of solved results at time: \"%s\"" % formatted_time
    PickleHelper.exporter(paikin_tal_solver, pickle_placement_complete_filename)
    print_elapsed_time(time.time() - start_time, "pickle completed solver export")
    print "Completed pickle export of the solved puzzle."
    return paikin_tal_solver


def print_elapsed_time(elapsed_time, task_name):
    """
    Elapsed Time Printer

    Prints the elapsed time for a task in nice formatting.

    Args:
        elapsed_time (int): Elapsed time in seconds
        task_name (string): Name of the task that was performed

    """
    # Get the current time
    ts = time.time()
    current_time = datetime.datetime.fromtimestamp(ts).strftime('%Y.%m.%d_%H.%M.%S')
    # Print elapsed time and the current time.
    print "The task \"%s\" took %d min %d sec and completed at %s." % (task_name, elapsed_time // 60, elapsed_time % 60,
                                                                       current_time)


if __name__ == "__main__":
    # PaikinTalSolver.use_best_buddy_placer = False
    # images = [".\\images\\muffins_300x200.jpg"]
    # paikin_tal_driver(images, PuzzleType.type1, 25)
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    #
    # PaikinTalSolver.use_best_buddy_placer = True
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\duck.bmp"]
    # paikin_tal_driver(images, PuzzleType.type1, 25)
    # images = [".\\images\\cat_sleeping_boy.jpg"]
    # paikin_tal_driver(images, PuzzleType.type1, 28)
    # paikin_tal_driver(images, PuzzleType.type2, 28)

    # images = [".\\images\\kitten_white_background.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 28)

    # images = [[".\\images\\book_tunnel_pixabay.jpg"],
    #           [".\\images\\dessert_pixabay.jpg"],
    #           [".\\images\\dandelion_pixabay.jpg"],
    #           [".\\images\\primula_pixabay.jpg"],
    #           [".\\images\\small_pink_flowers_pixabay.jpg"]]
    # for image_arr in images:
    #     paikin_tal_driver(image_arr, PuzzleType.type2, 28)

    # images = [".\\images\\two_faced_cat.jpg"]
    # paikin_tal_driver(images, PuzzleType.type1, 25)
    # paikin_tal_driver(images, PuzzleType.type2, 25)
    # images = [".\\images\\7.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\mcgill_20.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\7.jpg", ".\\images\\mcgill_20.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\mcgill_20.jpg", ".\\images\\two_faced_cat.jpg", ".\\images\\muffins_300x200.jpg"]
    # paikin_tal_driver(images, PuzzleType.type1, 28)
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\mcgill_03.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\bgu_805_08.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\bgu_805_10.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\3300_1.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    # images = [".\\images\\boat_100x100.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 25)
    # images = [".\\images\\che_100x100.jpg"]
    # paikin_tal_driver(images, PuzzleType.type1, 25)
    # paikin_tal_driver(images, PuzzleType.type2, 25)
    #
    # images = [".\\images\\bgu_805_08.jpg", ".\\images\\mcgill_20.jpg", ".\\images\\3300_1.jpg"]
    # PaikinTalSolver._CLEAR_BEST_BUDDY_HEAP_ON_SPAWN = True
    # InterPieceDistance._USE_ONLY_NEIGHBORS_FOR_STARTING_PIECE_TOTAL_COMPATIBILITY = True
    # InterPieceDistance._NEIGHBOR_COMPATIBILITY_SCALAR = 1
    # paikin_tal_driver(images, PuzzleType.type2, 28)

    #images = [".\\images\\bgu_805_08.jpg", ".\\images\\mcgill_20.jpg", ".\\images\\3300_1.jpg"]
    images = [".\\images\\bgu_805_08.jpg", ".\\images\\mcgill_20.jpg"]
    PaikinTalSolver._CLEAR_BEST_BUDDY_HEAP_ON_SPAWN = True
    InterPieceDistance._USE_ONLY_NEIGHBORS_FOR_STARTING_PIECE_TOTAL_COMPATIBILITY = True
    InterPieceDistance._NEIGHBOR_COMPATIBILITY_SCALAR = 1
    paikin_tal_driver(images, PuzzleType.type2, 28)

    # images = [".\\images\\bgu_805_08.jpg", ".\\images\\mcgill_20.jpg", ".\\images\\3300_1.jpg"]
    # PaikinTalSolver._CLEAR_BEST_BUDDY_HEAP_ON_SPAWN = True
    # InterPieceDistance._USE_ONLY_NEIGHBORS_FOR_STARTING_PIECE_TOTAL_COMPATIBILITY = False
    # InterPieceDistance._NEIGHBOR_COMPATIBILITY_SCALAR = 4
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    #
    # images = [".\\images\\bgu_805_08.jpg", ".\\images\\mcgill_20.jpg", ".\\images\\3300_1.jpg"]
    # PaikinTalSolver._CLEAR_BEST_BUDDY_HEAP_ON_SPAWN = False
    # InterPieceDistance._USE_ONLY_NEIGHBORS_FOR_STARTING_PIECE_TOTAL_COMPATIBILITY = False
    # InterPieceDistance._NEIGHBOR_COMPATIBILITY_SCALAR = 1
    # paikin_tal_driver(images, PuzzleType.type2, 28)
    #
    # images = [".\\images\\bgu_805_08.jpg", ".\\images\\mcgill_20.jpg", ".\\images\\3300_1.jpg"]
    # PaikinTalSolver._CLEAR_BEST_BUDDY_HEAP_ON_SPAWN = False
    # InterPieceDistance._USE_ONLY_NEIGHBORS_FOR_STARTING_PIECE_TOTAL_COMPATIBILITY = False
    # InterPieceDistance._NEIGHBOR_COMPATIBILITY_SCALAR = 4
    # paikin_tal_driver(images, PuzzleType.type2, 28)

    # images = [".\\images\\bgu_805_08.jpg", ".\\images\\mcgill_20.jpg"]
    # paikin_tal_driver(images, PuzzleType.type2, 28)
