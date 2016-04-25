import numpy

from hammoudeh_puzzle_solver.puzzle_importer import Puzzle, PuzzleType
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePiece, PuzzlePieceRotation
from hammoudeh_puzzle_solver.puzzle_piece import PuzzlePieceSide
from paikin_tal_solver.inter_piece_distance import InterPieceDistance


class ImageBestBuddyStatistics(object):
    """
    Class used to get the best buddy accuracy statistics for any image.
    """

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
        temp_puzzle = Puzzle(0, image_filepath, piece_width)

        # Get the piece IDs
        temp_puzzle.assign_all_piece_id_numbers_to_original_id()
        temp_puzzle.assign_all_pieces_to_original_location()
        temp_puzzle.assign_all_pieces_to_same_rotation(PuzzlePieceRotation.degree_0)

        # Calculate the inter-piece distance
        self._interpiece_distance = InterPieceDistance(temp_puzzle.pieces, distance_function, puzzle_type)

        # Get the link between number of test buddies and accuracy
        self._numb_best_buddies_versus_accuracy = numpy.zeros((PuzzlePieceSide.get_numb_sides() + 1,
                                                               PuzzlePieceSide.get_numb_sides() + 1,
                                                               PuzzlePieceSide.get_numb_sides() + 1),
                                                              numpy.uint32)
        # Store the location of each piece
        self._piece_locations, _ = temp_puzzle.build_placed_piece_info()

        # Clear the inter-piece distance in case this will be pickled.
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
        original_neighbor_id_and_side = piece.original_neighbor_id_numbers_and_sides()

        # Initialize the counters for the piece on the total number of best best buddies and how many are wrong
        numb_piece_bb = 0
        numb_wrong_interior_bb = 0
        numb_wrong_exterior_bb = 0

        # Iterate through all sides
        for i in xrange(0, neighbor_loc_and_sides - 1):

            # Get the neighbor location
            (neighbor_loc, piece_side) = neighbor_loc_and_sides[i]
            (neighbor_id, neighbor_side) = original_neighbor_id_and_side[i]

            # Assert neighbor and piece side are complementary
            assert piece_side.complementary_side == neighbor_side

            # Get the best buddy information for the piece.
            bb_info = self._interpiece_distance.best_buddies(piece.id_number, piece_side)
            if not bb_info:
                continue
            # Increment the best buddy count
            numb_piece_bb += 1

            # Check if there is a neighbor
            if self._piece_locations[neighbor_loc] == Puzzle.MISSING_PIECE_PUZZLE_INFO_VALUE:
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


if __name__ == '__main__':
    bb_results = ImageBestBuddyStatistics(".\\images\\muffins_300x200.jpg", 28, PuzzleType.type2,
                                          PuzzlePiece.calculate_asymmetric_distance)
    # bb_results = ImageBestBuddyStatistics(".\\images\\7.jpg", 28, PuzzleType.type2,
    #                                       PuzzlePiece.calculate_asymmetric_distance)
    x = 1












