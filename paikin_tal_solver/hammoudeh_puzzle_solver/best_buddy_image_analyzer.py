import numpy

from hammoudeh_puzzle_solver.puzzle_importer import Puzzle
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePiece
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePieceSide
from paikin_tal_solver.inter_piece_distance import InterPieceDistance


class ImageBestBuddyAccuracy(object):

    def __init__(self, image_filepath, piece_width, puzzle_type, distance_function):

        # Store the information about the input image
        self._filename_root = Puzzle.get_filename_without_extension(image_filepath)
        self._file_extension = Puzzle.get_file_extension(image_filepath)
        # File extension should not include the period
        assert "." not in self._file_extension

        self.puzzle_type = puzzle_type

        # Consider both interior and exterior best buddies.
        self._total_numb_exterior_bb = 0
        self._numb_wrong_exterior_bb = 0
        self._total_numb_interior_bb = 0
        self._numb_wrong_interior_bb = 0

        # Build a puzzle
        temp_puzzle = Puzzle(0, image_filepath, piece_width, puzzle_type)

        # Get the piece IDs
        temp_puzzle.assign_all_piece_id_numbers_to_original_id()
        temp_puzzle.assign_all_pieces_to_original_location()

        # Calculate the inter-piece distance
        self._interpiece_distance = InterPieceDistance(temp_puzzle.pieces, distance_function, puzzle_type)

        # Get the link between number of test buddies and accuracy
        self._numb_best_buddies_versus_accuracy = numpy.zeros(PuzzlePieceSide.get_numb_sides() + 1,
                                                              PuzzlePieceSide.get_numb_sides() + 1)
        # Store the location of each piece
        self._piece_locations, _ = temp_puzzle.build_placed_piece_info()

        # Clear the interpiece distance in case this will be pickled.
        self._interpiece_distance = None

    def filename_root(self):
        """
        Returns the file name of the original image used without file extension of path information.

        Returns (str): Filename of the original image with the file extension and file path removed.
        """
        return self.filename_root

    def file_extension(self):
        """
        Returns the file extension (e.g.

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







