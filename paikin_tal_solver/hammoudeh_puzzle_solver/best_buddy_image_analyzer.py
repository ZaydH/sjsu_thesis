"""
Best Buddy Analyzer for Normal Images
"""

import numpy
import sys

# noinspection PyUnresolvedReferences
from hammoudeh_puzzle_solver.puzzle_importer import Puzzle, PuzzleType, PickleHelper
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePiece, PuzzlePieceRotation
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePieceSide
from paikin_tal_solver.inter_piece_distance import InterPieceDistance


class ImageBestBuddyStatistics(object):
    """
    Class used to get the best buddy accuracy statistics for any image.
    """

    # Location to export pickle files to.
    PICKLE_DIRECTORY = ".\\pickle_files\\"

    def __init__(self, image_filepath, piece_width, puzzle_type, distance_function):

        # Store the information about the input image
        self._filename_root = Puzzle.get_filename_without_extension(image_filepath)
        self._file_extension = Puzzle.get_file_extension(image_filepath)
        # File extension should not include the period
        assert "." not in self._file_extension

        self.puzzle_type = puzzle_type

        # Consider both interior and exterior best buddies.
        self._numb_wrong_exterior_bb = 0
        self._total_numb_interior_bb = 0
        self._numb_wrong_interior_bb = 0

        # Build a puzzle
        self._puzzle = Puzzle(0, image_filepath, piece_width)

        # Get the piece IDs
        self._puzzle.assign_all_piece_id_numbers_to_original_id()
        self._puzzle.assign_all_pieces_to_original_location()
        self._puzzle.assign_all_pieces_to_same_rotation(PuzzlePieceRotation.degree_0)

        # Calculate the inter-piece distance
        self._interpiece_distance = InterPieceDistance(self._puzzle.pieces, distance_function, puzzle_type)

        # Get the link between number of test buddies and accuracy
        self._numb_best_buddies_versus_accuracy = numpy.zeros((PuzzlePieceSide.get_numb_sides() + 1,
                                                               PuzzlePieceSide.get_numb_sides() + 1,
                                                               PuzzlePieceSide.get_numb_sides() + 1),
                                                              numpy.uint32)
        # Store the location of each piece
        self._piece_locations, _ = self._puzzle.build_placed_piece_info()

    def calculate_results(self):
        """
        Calculates the best buddy accuracy results.
        """

        # Calculate the best buddy information for each piece.
        for piece in self._puzzle.pieces:
            self.analyze_piece_best_buddy_info(piece)

        # Clear up the memory
        self._puzzle = None
        self._interpiece_distance = None

    @property
    def filename_root(self):
        """
        Returns the file name of the original image used without file extension of path information.

        Returns (str): Filename of the original image with the file extension and file path removed.
        """
        return self.filename_root

    @property
    def file_extension(self):
        """
        Returns the file extension (e.g. "jpg", "bmp", "png", etc.)

        Returns (str): File extension of the original image
        """
        return self.filename_root

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

        # Iterate through all sides
        for i in xrange(0, PuzzlePieceSide.get_numb_sides()):

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

            # Piece has a neighbor
            else:
                # Increment interior best buddy count
                self._total_numb_interior_bb += 1
                if neighbor_id != self._piece_locations[neighbor_loc]:
                    numb_wrong_interior_bb += 1
                    self._numb_wrong_interior_bb += 1
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
              locaton (i.e. where it had a neighbor)
            numb_wrong_exterior_bb (int): Number of best buddies for a piece that were wrong when it had no neighbor

        Returns (Tuple[int]): Tuple for accessing the numpy array

        """
        assert numb_bb >= numb_wrong_interior_bb + numb_wrong_exterior_bb
        return numb_bb, numb_wrong_interior_bb, numb_wrong_exterior_bb

    def print_results(self):
        """
        Prints the best buddy results to the console.
        """
        print "Best Buddy Results for Image:\t" + self._filename_root
        print "\tFile Extension:\t" + self._file_extension

        # Total number of best buddies
        print "\tTotal Number of Best Buddies:\t%d" % self.total_number_of_best_buddies
        print "\tBest Buddy Accuracy\t%1.2f%%" % (100 * self.accuracy)
        print ""
        print "\tNumber of Wrong Interior Best Buddies:\t%d" % self._numb_wrong_interior_bb
        print "\tNumber of Wrong Exterior Best Buddies:\t%d" % self._numb_wrong_exterior_bb

    @property
    def total_number_of_best_buddies(self):
        """
        Gets the total of best buddies (both interior/exterior and right/wrong).

        Returns (int): Best buddy count
        """
        return self._numb_wrong_exterior_bb + self._numb_wrong_interior_bb

    @property
    def accuracy(self):
        """
        Gets the accuracy of the best buddy placer.  It is defined as:

        :math:`accuracy = (numb_wrong_interior_bb + numb_wrong_exterior_bb)/(total_numb_best_buddy)`

        Returns (float): Best buddy accuracy
        """
        accuracy = 1.0 * (self._numb_wrong_interior_bb + self._numb_wrong_exterior_bb)
        return accuracy / self.total_number_of_best_buddies


if __name__ == '__main__':

    pickle_file = ImageBestBuddyStatistics.PICKLE_DIRECTORY + "bb_accuracy.pk"

    # # bb_results = ImageBestBuddyStatistics(".\\images\\muffins_300x200.jpg", 28, PuzzleType.type2,
    # #                                       PuzzlePiece.calculate_asymmetric_distance)
    # bb_results = ImageBestBuddyStatistics(".\\images\\7.jpg", 28, PuzzleType.type2,
    #                                       PuzzlePiece.calculate_asymmetric_distance)
    # PickleHelper.exporter(bb_results, pickle_file)

    # Calculate the print the results.
    bb_results = PickleHelper.importer(pickle_file)
    bb_results.calculate_results()
    bb_results.print_results()
