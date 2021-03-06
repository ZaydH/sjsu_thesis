"""
Best Buddy Analyzer for Normal Images
"""
import cStringIO
import numpy as np
import sys
import logging

# noinspection PyUnresolvedReferences
import time

from hammoudeh_puzzle import config
from hammoudeh_puzzle.pickle_helper import PickleHelper
from hammoudeh_puzzle.puzzle_importer import Puzzle, PuzzleType, PieceSideBestBuddyAccuracyResult, PuzzleSolver
from hammoudeh_puzzle.puzzle_piece import PuzzlePiece, PuzzlePieceRotation, top_level_calculate_asymmetric_distance
from hammoudeh_puzzle.puzzle_piece import PuzzlePieceSide
from paikin_tal_solver.inter_piece_distance import InterPieceDistance


class ImageBestBuddyStatistics(object):
    """
    Class used to get the best buddy accuracy statistics for any image.
    """

    def __init__(self, image_file_path, piece_width, puzzle_type, distance_function):

        # Store the information about the input image
        self._filename_root = Puzzle.get_filename_without_extension(image_file_path)
        self._file_extension = Puzzle.get_file_extension(image_file_path)
        # File extension should not include the period
        assert "." not in self._file_extension

        logging.info("Performing best buddy analysis for image: %s" % self._filename_root)

        self.puzzle_type = puzzle_type

        # Consider both interior and exterior best buddies.
        self._numb_wrong_exterior_bb = 0
        self._total_numb_interior_bb = 0
        self._numb_wrong_interior_bb = 0

        # Build a puzzle
        self._puzzle = Puzzle(0, image_file_path, piece_width)

        self.numb_pieces = self._puzzle.numb_pieces

        # Get the piece IDs
        self._puzzle.assign_all_piece_id_numbers_to_original_id()
        self._puzzle.assign_all_pieces_to_original_location()
        self._puzzle.assign_all_pieces_to_same_rotation(PuzzlePieceRotation.degree_0)

        # Calculate the inter-piece distance
        self._interpiece_distance = InterPieceDistance(self._puzzle.pieces, distance_function, puzzle_type)

        # Get the link between number of test buddies and accuracy
        self._numb_best_buddies_versus_accuracy = np.zeros((PuzzlePieceSide.get_numb_sides() + 1,
                                                            PuzzlePieceSide.get_numb_sides() + 1,
                                                            PuzzlePieceSide.get_numb_sides() + 1),
                                                           np.uint32)
        # Store the location of each piece
        self._piece_locations, _ = self._puzzle.build_placed_piece_info()

    def calculate_results(self):
        """
        Calculates the best buddy accuracy results.
        """

        # Calculate the best buddy information for each piece.
        for piece in self._puzzle.pieces:
            self.analyze_piece_best_buddy_info(piece)

        # Output the best buddy accuracy image.
        self.output_results_image()

        # Clear up the memory
        self._puzzle = None
        self._interpiece_distance = None

    @property
    def filename_root(self):
        """
        Returns the file name of the original image used without file extension of path information.

        Returns (str):
            Filename of the original image with the file extension and file path removed.
        """
        return self.filename_root

    @property
    def file_extension(self):
        """
        Returns the file extension (e.g. "jpg", "bmp", "png", etc.)

        Returns (str):
            File extension of the original image
        """
        return self._file_extension

    def analyze_piece_best_buddy_info(self, piece):
        """
        Analyze the best buddy information for a single piece.

        Args:
            piece (PuzzlePiece): Puzzle piece whose best buddy info will be analyzed
        """

        # Get the neighbor location and sides
        neighbor_loc_and_sides = piece.get_neighbor_locations_and_sides()
        original_neighbor_id_and_side = piece.original_neighbor_id_numbers_and_sides

        # Initialize the counters for the piece on the total number of best best buddies and how many are wrong
        numb_piece_bb = 0
        numb_wrong_interior_bb = 0
        numb_wrong_exterior_bb = 0

        # Reset the image coloring
        piece.reset_image_coloring_for_polygons()

        # Iterate through all sides
        for i in xrange(PuzzlePieceSide.get_numb_sides()):

            # Get the neighbor location
            (neighbor_loc, piece_side) = neighbor_loc_and_sides[i]
            neighbor_id_and_side = original_neighbor_id_and_side[i]

            # Assert neighbor and piece side are complementary
            if neighbor_id_and_side is not None:
                (neighbor_id, neighbor_side) = neighbor_id_and_side
                assert piece_side == neighbor_side
            else:
                neighbor_id = -sys.maxint

            # Get the best buddy information for the piece.
            bb_info = self._interpiece_distance.best_buddies(piece.id_number, piece_side)
            if not bb_info:
                piece.results_image_polygon_coloring(piece_side, PieceSideBestBuddyAccuracyResult.no_best_buddy)
                continue
            # Increment the best buddy count
            numb_piece_bb += 1

            # Check if there is a neighbor
            if (neighbor_loc[0] < 0 or neighbor_loc[0] >= self._puzzle.grid_size[0]
                    or neighbor_loc[1] < 0 or neighbor_loc[1] >= self._puzzle.grid_size[1]
                    or self._piece_locations[neighbor_loc] == Puzzle.MISSING_PIECE_PUZZLE_INFO_VALUE):
                # If the neighboring cell is empty and it has a best buddy, it is wrong
                self._numb_wrong_exterior_bb += 1
                numb_wrong_exterior_bb += 1
                piece.results_image_polygon_coloring(piece_side, PieceSideBestBuddyAccuracyResult.wrong_best_buddy)

            # Piece has a neighbor
            else:
                # Increment interior best buddy count
                self._total_numb_interior_bb += 1
                # TODO currently only supports single best buddy
                bb_info = bb_info[0]
                if neighbor_id != bb_info[0] or piece_side.complementary_side != bb_info[1]:
                    numb_wrong_interior_bb += 1
                    self._numb_wrong_interior_bb += 1
                    piece.results_image_polygon_coloring(piece_side, PieceSideBestBuddyAccuracyResult.wrong_best_buddy)
                else:
                    piece.results_image_polygon_coloring(piece_side, PieceSideBestBuddyAccuracyResult.correct_best_buddy)
        # Update the master data structure showing the best buddy distribution
        numpy_index = ImageBestBuddyStatistics.best_buddies_versus_accuracy_tuple(numb_piece_bb,
                                                                                  numb_wrong_interior_bb,
                                                                                  numb_wrong_exterior_bb)
        # Increment the statistics
        self._numb_best_buddies_versus_accuracy[numpy_index] += 1

    @staticmethod
    def best_buddies_versus_accuracy_tuple(numb_bb, numb_wrong_interior_bb, numb_wrong_exterior_bb):
        """

        Args:
            numb_bb (int): Total number of best buddies for a piece
            numb_wrong_interior_bb (int): Number of best buddies for a piece that were wrong on an internal
              location (i.e. where it had a neighbor)
            numb_wrong_exterior_bb (int): Number of best buddies for a piece that were wrong when it had no neighbor

        Returns (Tuple[int]):
            Tuple for accessing the numpy array

        """
        assert numb_bb >= numb_wrong_interior_bb + numb_wrong_exterior_bb
        return numb_bb, numb_wrong_interior_bb, numb_wrong_exterior_bb

    def print_results(self):
        """
        Prints the best buddy results to the console.
        """

        string_io = cStringIO.StringIO()
        print >>string_io, "\nBest Buddy Results for Image:\t" + self._filename_root
        print >>string_io, "\tFile Extension:\t" + self._file_extension
        print >>string_io, "\tNumber of Pieces:\t%d" % self.numb_pieces

        # Total number of best buddies
        print >>string_io, "\tTotal Number of Best Buddies:\t%d" % self.total_number_of_best_buddies
        print >>string_io, "\tTotal Best Buddy Accuracy:\t\t%1.2f%%" % (100 * self.total_accuracy)

        print >>string_io, "\tBest Buddy Density:\t\t\t\t%1.2f%%" % (100 * self.density)
        print >>string_io, "\tInterior Best Buddy Accuracy:\t%1.2f%%" % (100 * self.interior_accuracy)
        print >>string_io, ""
        print >>string_io, "\tNumber of Wrong Interior Best Buddies:\t%d" % self._numb_wrong_interior_bb
        print >>string_io, "\tNumber of Wrong Exterior Best Buddies:\t%d" % self._numb_wrong_exterior_bb

        # log the result
        logging.info(string_io.getvalue())
        string_io.close()

    @property
    def density(self):
        """
        Calculates the best buddy density for the original image.  It is defined as the total number of best
        buddies divided by the total number of possible best buddies (i.e. number of pieces multiplied by the number
        of sides per piece).

        Returns (float):
            Best buddy density across the entire image.

        """
        return 1.0 * self.total_number_of_best_buddies / (self.numb_pieces * PuzzlePieceSide.get_numb_sides())

    @property
    def total_number_of_best_buddies(self):
        """
        Gets the total of best buddies (both interior/exterior and right/wrong).

        Returns (int):
            Number of best buddies in the image
        """
        return self._numb_wrong_exterior_bb + self._total_numb_interior_bb

    @property
    def total_accuracy(self):
        """
        Gets the best buddy accuracy across the entire image  It is defined as:

        :math:`accuracy = 1 - (numb_wrong_interior_bb + numb_wrong_exterior_bb)/(total_numb_best_buddy)`

        Returns (float):
            Best buddy accuracy for the entire image
        """
        return 1 - 1.0 * (self._numb_wrong_interior_bb + self._numb_wrong_exterior_bb) / self.total_number_of_best_buddies

    @property
    def interior_accuracy(self):
        """
        Gets the best buddy accuracy considering only interior best buddies.  It is defined as:

        :math:`interior_accuracy = 1 - (numb_wrong_interior_bb)/(total_numb_interior_best_buddy)`

        Returns (float):
            Best buddy accuracy considering only interior pieces.
        """
        return 1 - 1.0 * self._numb_wrong_interior_bb / self._total_numb_interior_bb

    def output_results_image(self):
        """
        Creates an image showing the best buddy accuracy distribution of an image.
        """
        # Create a time stamp for the results
        timestamp = time.time()
        # Build the original filename
        orig_img_filename = self._filename_root + "." + self._file_extension

        descriptor = "image_best_buddies"
        output_filename = Puzzle.make_image_filename(PuzzleSolver.PaikinTal, [orig_img_filename], descriptor,
                                                     Puzzle.OUTPUT_IMAGE_DIRECTORY, self.puzzle_type,
                                                     timestamp, orig_img_filename=orig_img_filename)
        # Stores the results to a file.
        self._puzzle.build_puzzle_image(use_results_coloring=True)
        self._puzzle.save_to_file(output_filename)


def run_best_buddies_analyzer(image_file, piece_width, puzzle_type):
    """
    Runs the Best Buddy Analyzer on an image and outputs the best buddy results and generates a results image.

    Args:
        image_file (str): Path to an image file
        piece_width (int): Width of a puzzle piece in number of pixels
        puzzle_type (PuzzleType): Type of the puzzle to solve

    """
    pickle_file = PickleHelper.build_filename("bb_accuracy", [image_file], puzzle_type)

    filename_with_image_folder = config.add_image_folder_path([image_file])
    bb_results = ImageBestBuddyStatistics(filename_with_image_folder[0], piece_width, puzzle_type,
                                          top_level_calculate_asymmetric_distance)

    PickleHelper.exporter(bb_results, pickle_file)

    # Calculate the print the results.
    bb_results = PickleHelper.importer(pickle_file)
    bb_results.calculate_results()
    bb_results.print_results()


if __name__ == '__main__':
    config.setup_logging()

    # run_best_buddies_analyzer("." + os.sep + "images" + os.sep + "duck.bmp", 28, PuzzleType.type2)
    #
    # run_best_buddies_analyzer("." + os.sep + "images" + os.sep + "muffins_300x200.jpg", 28, PuzzleType.type2)
    #
    # run_best_buddies_analyzer("." + os.sep + "images" + os.sep + "cat_sleeping_boy.jpg", 28, PuzzleType.type2)

    puzzle_ids = [12, 8, 13]
    for puzzle_id in puzzle_ids:
        filename = config.build_pomeranz_805_piece_filename(puzzle_id)
        run_best_buddies_analyzer(filename, 28, PuzzleType.type2)
