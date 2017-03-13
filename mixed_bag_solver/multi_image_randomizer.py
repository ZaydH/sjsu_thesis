"""
Helper file that will generate a randomized, combined image from one or more input images.

This is used to generate images for reports and papers.
"""

import itertools
import logging
import random
# noinspection PyUnresolvedReferences
from types import ListType

from hammoudeh_puzzle import config
from hammoudeh_puzzle import puzzle_importer
from hammoudeh_puzzle.puzzle_importer import Puzzle
from hammoudeh_puzzle.puzzle_piece import PuzzlePiece, PuzzlePieceRotation
from hammoudeh_puzzle.solver_helper import PuzzleLocation


def multi_image_randomizer(image_filenames, piece_width=config.DEFAULT_PIECE_WIDTH):
    """
    Takes a list of one or more image files and randomizes them into a single output image.

    Args:
        image_filenames (List[str]): List of one or more image file names to be randomized
        piece_width (int): Dimensions of the individual puzzle piece

    Returns (Puzzle): A puzzle containing a randomized version of the image(s) specified.
    """

    logging.info("Starting Multi-Image Randomizer")
    puzzle_importer.log_puzzle_filenames(image_filenames)

    # Get a list of puzzle pieces from the image and determine the dimensions of the output image
    image_files = config.add_image_folder_path(image_filenames)
    pieces, puzzles = Puzzle.get_combined_pieces_multiple_images(image_files, piece_width)  # type: list[PuzzlePiece], list[Puzzle]
    all_divisors = divisors(len(pieces))
    # Select the width so it is longer than the length, but ensure the index does not exceed the end of the array
    width_index = min(len(all_divisors), int(len(all_divisors)/2))
    output_width = all_divisors[width_index]  # Use the middle of the array so output is square-ish
    output_length = len(pieces) // output_width
    if config.PERFORM_ASSERT_CHECKS:
        assert(output_length * output_width == len(pieces))

    # Set the rotation and location of the pieces
    DEFAULT_PUZZLE_ID = 0
    random.seed()
    piece_locations = [i for i in xrange(0, len(pieces))]
    numb_unplaced_pieces = len(pieces)
    for piece in pieces:
        piece.rotation = PuzzlePieceRotation.random_rotation()

        # Set the piece location randomly
        idx = random.randint(0, numb_unplaced_pieces - 1)
        loc = piece_locations[idx]
        piece.puzzle_location = PuzzleLocation(DEFAULT_PUZZLE_ID, loc // output_width, loc % output_width)

        # Swap out the used location
        numb_unplaced_pieces -= 1
        piece_locations[idx] = piece_locations[numb_unplaced_pieces]

    return Puzzle.reconstruct_from_pieces(pieces, display_image=False)


def divisors(n):
    """
    Builds a list of divisors for a specified integer

    Args:
        n (int): An integer number

    Returns (List[int]): List of divisors for the specified number
    """
    return sorted(list(itertools.chain.from_iterable([[i, n//i] for i in xrange(1, int(n**0.5) + 1) if n % i == 0])))

if __name__ == '__main__':
    # Setup the logger
    config.setup_logging()

    # Build the randomized image then save it to a file.
    images = ["3300_1.jpg",
              config.build_mcgill_540_piece_filename(7),
              config.build_pomeranz_805_piece_filename(19),
              config.build_pomeranz_805_piece_filename(14),
              config.build_pomeranz_805_piece_filename(10)]
    randomized_puzzle = multi_image_randomizer(images)
    randomized_puzzle.save_to_file("randomized_image.jpg")
