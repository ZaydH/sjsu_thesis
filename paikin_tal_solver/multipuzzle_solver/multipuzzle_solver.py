import cStringIO
import copy
import logging
import time

import numpy as np

from hammoudeh_puzzle import config
from hammoudeh_puzzle import puzzle_importer
from hammoudeh_puzzle import solver_helper
from hammoudeh_puzzle.pickle_helper import PickleHelper
from hammoudeh_puzzle.puzzle_importer import Puzzle
from hammoudeh_puzzle.puzzle_piece import PuzzlePiece
from hammoudeh_puzzle.solver_helper import print_elapsed_time
from hierarchical_clustering import HierarchicalClustering
from paikin_tal_solver.solver import PaikinTalSolver


class StitchingPieceInfo(object):
    def __init__(self, piece_id, segment_numb):
        self._piece_id = piece_id
        self._segment_numb = segment_numb

        self._total_numb_segments = None

        self._total_numb_solved_pieces = 0
        self._solver_piece_segments = []
        self._solver_piece_without_segment = []

        self._segment_overlap_coefficient = []

        # Ensure no duplicate pieces
        if config.PERFORM_ASSERT_CHECKS:
            self._all_solver_pieces = {}

    @property
    def piece_id(self):
        """
        Gets the piece identification number of the stitching piece

        Returns (int):
            Piece identification number
        """
        return self._piece_id

    @property
    def segment_numb(self):
        """
        Gets the identification number of the segment where the specified piece is location.

        Returns (int):
            Piece identification number
        """
        return self._segment_numb

    def add_solver_piece(self, solved_piece_id, piece_segment_id):
        """
        After the solver is run using the stitching piece as the seed, pieces from the solved puzzle are added to the
        stitching piece information object.

        Args:
            solved_piece_id (int): Identification number of piece in stitching piece solver

            piece_segment_id (int): Identification number of the segment where the piece with identification
                number of "solved_piece_id" was assigned in initial segmentation.  This value is "None" if the piece
                has no associated segment.
        """
        self._total_numb_solved_pieces += 1

        # Ensure no duplicate pieces.
        if config.PERFORM_ASSERT_CHECKS:
            key = PuzzlePiece.create_key(solved_piece_id)
            assert key not in self._all_solver_pieces
            self._all_solver_pieces[key] = solved_piece_id

        # Handle the case where the piece was not assigned to any segment initially
        if piece_segment_id is None:
            self._solver_piece_without_segment.append(solved_piece_id)
            return

        while len(self._solver_piece_segments) < piece_segment_id + 1:
            self._solver_piece_segments.append([])
        self._solver_piece_segments[piece_segment_id].append(solved_piece_id)

    @property
    def total_numb_pieces_in_solved_result(self):
        """
        Gets the total number of pieces in the solved result of this stitching piece.

        Returns (int):
            Number of pieces that were in the solved result of this stitching piece
        """
        return self._total_numb_solved_pieces

    def log_piece_to_segment_mapping(self, total_numb_segments=None):
        """
        This logs the breakdown of pieces by mapping them to segments.

        The user can optionally specify the total number of segments in the puzzle and all will be logged.  If this is
        not specified, then only the maximum segment number this stitching piece has a piece from is logged.

        Args:
            total_numb_segments (int):  Total number of segments found
        """

        string_io = cStringIO.StringIO()
        print >> string_io, "Stitching Piece Solver Result Breakdown by Segment"
        print >> string_io, "Segment ID #%d" % self._segment_numb
        print >> string_io, "Stitching Piece ID #%d\n" % self._piece_id

        # Handle the case of no segment
        pieces_no_segment = len(self._solver_piece_without_segment)
        print >> string_io, "%d pieces (%d%%) have no associated segment." % (pieces_no_segment,
                                                                              self._get_percent_in_segment(pieces_no_segment))

        # Determine number of segments to log with option to log more than in the list
        numb_segments_in_info = len(self._solver_piece_segments)
        numb_segments = max(numb_segments_in_info, total_numb_segments) if total_numb_segments is not None else numb_segments_in_info
        for segment_cnt in xrange(0, numb_segments):
            if segment_cnt >= numb_segments_in_info:
                pieces_in_segment = 0
            else:
                pieces_in_segment = len(self._solver_piece_segments[segment_cnt])

            print >> string_io, "%d pieces (%d%%) are from segment #%d" % (pieces_in_segment,
                                                                           self._get_percent_in_segment(pieces_in_segment),
                                                                           segment_cnt)

        logging.info(string_io.getvalue())
        string_io.close()

    def _get_percent_in_segment(self, piece_count_in_segment):
        """
        Converts piece count to the number of pieces in the segment.

        Args:
            piece_count_in_segment (int): Number of pieces in the segment

        Returns (float):
            Percent representation with decimal on the number of pieces in the segment.
        """
        return round(100.0 * piece_count_in_segment / self._total_numb_solved_pieces, 1)

    def calculate_overlap_coefficient(self, size_of_each_segment):
        """
        For solved puzzle of this stitching piece, this function calculates and returns the overlap coefficient.

        Args:
            size_of_each_segment (List[int]): Number of pieces in each segment

        Returns (List[float]):
            Overlap coefficient for this stitching piece to all other segments
        """

        if len(self._solver_piece_segments) > len(size_of_each_segment):
            raise ValueError("List containing each segment's size has too few elements for number of segments")

        # Iterate through each segment and calculate the overlap coefficient.
        self._segment_overlap_coefficient = []
        for segment_cnt in xrange(0, len(size_of_each_segment)):
            other_segment_size = size_of_each_segment[segment_cnt]

            # Handle case where segment number is greater than the maximum segment number for any piece in solved
            # puzzle for this stitching piece
            if segment_cnt >= len(self._solver_piece_segments):
                pieces_from_other_segment = 0
            else:
                pieces_from_other_segment = len(self._solver_piece_segments[segment_cnt])

            overlap = 1.0 * pieces_from_other_segment / min(self._total_numb_solved_pieces, other_segment_size)
            self._segment_overlap_coefficient.append(overlap)

        return copy.copy(self._segment_overlap_coefficient)


class MultiPuzzleSolver(object):

    _MINIMUM_SEGMENT_SIZE = 10

    # Variables related to the saving of puzzle images.
    _SAVE_EACH_SINGLE_PUZZLE_RESULT_TO_AN_IMAGE_FILE = True
    _SAVE_SELECTED_SEGMENTS_TO_AN_IMAGE_FILE = True
    _SAVE_STITCHING_PIECE_SOLVER_RESULT_TO_AN_IMAGE_FILE = True
    _SAVE_FINAL_PUZZLE_IMAGES = True

    # File descriptors for pickle export
    _POST_SEGMENTATION_PUZZLE_PLACEMENT_FILE_DESCRIPTOR = "go_to_segmentation_in_round_%d"
    _POST_SEGMENTATION_COMPLETED_PICKLE_FILE_DESCRIPTOR = "post_initial_segmentation"
    _POST_STITCHING_PIECE_SOLVING_PICKLE_FILE_DESCRIPTOR = "post_stitching_piece_solving"
    _POST_SIMILARITY_MATRIX_CALCULATION_PICKLE_FILE_DESCRIPTOR = "post_similarity_matrix"
    _POST_HIERARCHICAL_CLUSTERING_PICKLE_FILE_DESCRIPTOR = "post_hierarchical_clustering"
    _POST_SELECT_STARTING_PIECES_PICKLE_FILE_DESCRIPTOR = "post_select_starting_pieces"

    # Pickle Related variables
    _ALLOW_POST_SEGMENTATION_PLACEMENT_PICKLE_EXPORT = True
    _ALLOW_POST_SEGMENTATION_ROUND_PICKLE_EXPORT = False
    _ALLOW_POST_SEGMENTATION_COMPLETED_PICKLE_EXPORT = False
    _ALLOW_POST_STITCHING_PIECE_SOLVING_PICKLE_EXPORT = False
    _ALLOW_POST_SIMILARITY_MATRIX_CALCULATION_PICKLE_EXPORT = False
    _ALLOW_POST_HIERARCHICAL_CLUSTERING_PICKLE_EXPORT = False
    _ALLOW_POST_SELECT_STARTING_PIECES_PICKLE_EXPORT = False

    def __init__(self, image_filenames, pieces, distance_function, puzzle_type):
        """
        Constructor for the Multi-Puzzle Solver.

        Args:
            image_filenames (List[str]): Name of the image files.
            pieces (List[PuzzlePiece])): List of puzzle pieces
            distance_function: Calculates the distance between two PuzzlePiece objects.
            puzzle_type (PuzzleType): Type of Paikin Tal Puzzle
        """
        self._numb_pieces = len(pieces)

        self._segments = []
        # This maps piece identification numbers to segments.
        self._piece_id_to_segment_map = {}

        self._stitching_pieces = None

        # Variables used for creating the output image files
        self._start_timestamp = time.time()
        self._image_filenames = image_filenames

        self._numb_segmentation_rounds = None

        # Build the Paikin Tal Solver
        self._paikin_tal_solver = PaikinTalSolver(pieces, distance_function, puzzle_type=puzzle_type)

        # Input to the hierarchical clustering algorithm.
        self._asymmetric_overlap_matrix = None
        self._segment_similarity_matrix = None

        # Output from the hierarchical clustering algorithm
        self._segment_clusters = None
        self._final_cluster_similarity_matrix = None

        # Built from the hierarchical clustering output
        # Input to the final solver.
        self._final_starting_pieces = None

        self._final_puzzles = None

    def run(self):
        """
        Executes all steps involved in the multipuzzle solver.

        Returns (List[Puzzle]):
            Puzzle solutions from the solver.
        """

        logging.info("Multipuzzle Solver Started")

        self._find_initial_segments()

        self._perform_stitching_piece_solving()

        self._build_similarity_matrix()

        self._perform_hierarchical_clustering()

        self._select_starting_pieces_from_clusters()

        self._perform_placement_with_final_seed_pieces()

        self._final_puzzles = self._build_output_puzzles()

        logging.info("Multipuzzle Solver Complete")
        print_elapsed_time(self._start_timestamp, "entire multipuzzle solver.")
        return self._final_puzzles

    def _find_initial_segments(self, skip_initialization=False, go_directly_to_segmentation=False):
        """
        Through iterative single puzzle placing, this function finds a set of segments.

        Args:
            skip_initialization (bool): Skip the initial segments setup.
            go_directly_to_segmentation (bool): Skips right to segmentation.  This feature is largely for debug of
                segmentation via pickle.
        """

        if not skip_initialization and not go_directly_to_segmentation:
            self._paikin_tal_solver.allow_placement_of_all_pieces()

            self._numb_segmentation_rounds = 0

        # Essentially a Do-While loop
        while True:

            time_segmentation_round_began = time.time()
            if not go_directly_to_segmentation:
                self._numb_segmentation_rounds += 1

                logging.info("Beginning segmentation round #%d" % self._numb_segmentation_rounds)

                # In first iteration, the solver settings are still default.
                if self._numb_segmentation_rounds > 1:
                    self._paikin_tal_solver.restore_initial_placer_settings_and_distances()

                # Perform placement as if there is only a single puzzle
                self._paikin_tal_solver.run_single_puzzle_solver()

                if MultiPuzzleSolver._ALLOW_POST_SEGMENTATION_PLACEMENT_PICKLE_EXPORT:
                    self._pickle_export_after_segmentation_puzzle_placement()
            # Proceed with placement as normal.
            go_directly_to_segmentation = False

            # Get the segments from this iteration of the loop
            solved_segments = self._paikin_tal_solver.segment(perform_segment_cleaning=True)
            max_segment_size = self._process_solved_segments(solved_segments[0])

            if MultiPuzzleSolver._SAVE_EACH_SINGLE_PUZZLE_RESULT_TO_AN_IMAGE_FILE:
                self._save_single_solved_puzzle_to_file(self._numb_segmentation_rounds)

            logging.info("Beginning segmentation round #%d" % self._numb_segmentation_rounds)
            solver_helper.print_elapsed_time(time_segmentation_round_began,
                                             "segmentation round #%d" % self._numb_segmentation_rounds)

            if MultiPuzzleSolver._ALLOW_POST_SEGMENTATION_ROUND_PICKLE_EXPORT:
                self._pickle_export_after_segmentation_round()

            # Stop segmenting if no pieces left or maximum segment size is less than the minimum
            if max_segment_size < MultiPuzzleSolver._MINIMUM_SEGMENT_SIZE \
                    or self._numb_pieces - len(self._piece_id_to_segment_map) < MultiPuzzleSolver._MINIMUM_SEGMENT_SIZE:
                break

        # Re-allow all pieces to be placed.
        self._paikin_tal_solver.allow_placement_of_all_pieces()
        self._paikin_tal_solver.reset_all_pieces_placement()

        self._log_segmentation_results()

        if MultiPuzzleSolver._ALLOW_POST_SEGMENTATION_COMPLETED_PICKLE_EXPORT:
            self._pickle_export_after_all_segmentation_completed()

    def _perform_stitching_piece_solving(self):
        """
        Runs a single puzzle, reduced size solver for each of the stitching pieces.
        """

        self._stitching_pieces = self._get_stitching_pieces()
        if config.PERFORM_ASSERT_CHECKS:
            assert len(self._stitching_pieces) == len(self._segments)

        # Iterate through all the stitching pieces in all segments and run the solver on them to determine
        # affinity between segments.
        for segment_cnt in xrange(0, len(self._stitching_pieces)):
            for stitching_piece_cnt in xrange(0, len(self._stitching_pieces[segment_cnt])):
                stitching_piece = self._stitching_pieces[segment_cnt][stitching_piece_cnt]

                logging.info("Beginning placement of stitching piece #%d for segment #%d" % (stitching_piece.piece_id,
                                                                                             segment_cnt))

                # For each stitching piece in each segment run the solver and see the results.
                self._paikin_tal_solver.restore_initial_placer_settings_and_distances()
                self._paikin_tal_solver.run_stitching_piece_solver(stitching_piece.piece_id)

                numb_pieces_placed = self._numb_pieces - self._paikin_tal_solver.numb_unplaced_valid_pieces
                logging.info("Number of Pieces Placed: %d\n" % numb_pieces_placed)

                # Determine the segment each stitching piece belongs to.
                self._process_stitching_piece_solver_result(segment_cnt, stitching_piece_cnt)

                if MultiPuzzleSolver._SAVE_STITCHING_PIECE_SOLVER_RESULT_TO_AN_IMAGE_FILE:
                    self._save_stitching_piece_solved_puzzle_to_file(stitching_piece)

        if MultiPuzzleSolver._ALLOW_POST_STITCHING_PIECE_SOLVING_PICKLE_EXPORT:
            self._pickle_export_after_stitching_piece_solving()

    def _process_stitching_piece_solver_result(self, segment_numb, stitching_piece_numb):
        """
        After the solver is run for a stitching piece, this function process those results including adding the
        pieces in the solved result to the StitchingPieceInfo object.  This includes tracking the segment to which
        each piece in the solved output belows (if any).

        Args:
            segment_numb (int): Identification number of the segment whose stitching piece is being processed

            stitching_piece_numb (int): Index of the stitching piece in the associated segment list
        """
        solved_puzzle, _ = self._paikin_tal_solver.get_solved_puzzles()

        single_puzzle_id_number = 0
        for piece in solved_puzzle[single_puzzle_id_number]:
            piece_id = piece.id_number
            # Get segment associated with the piece if any
            try:
                piece_segment_numb = self._piece_id_to_segment_map[PuzzlePiece.create_key(piece_id)]
            except KeyError:
                piece_segment_numb = None
            # Add the piece to the stitching piece information
            self._stitching_pieces[segment_numb][stitching_piece_numb].add_solver_piece(piece_id,
                                                                                        piece_segment_numb)

        self._stitching_pieces[segment_numb][stitching_piece_numb].log_piece_to_segment_mapping(len(self._segments))

    def _build_similarity_matrix(self):
        """
        Creates the asymmetric overlap and segment similarity matrices for use in the hierarchical clustering.
        """

        logging.info("Reminder on the image files:")
        puzzle_importer.log_puzzle_filenames(self._image_filenames)

        # Build the similarity matrix.  Worst similarity is 0
        numb_segments = len(self._segments)
        self._asymmetric_overlap_matrix = np.full((numb_segments, numb_segments), fill_value=-1, dtype=np.float)

        # Calculate asymmetric overlap for each segment
        numb_pieces_in_each_segment = [segment.numb_pieces for segment in self._segments]
        for segment_i in xrange(0, numb_segments):

            # Iterate through all the stitching pieces in this segment and calculate the asymmetric overlap
            for stitching_piece_info in self._stitching_pieces[segment_i]:
                overlap = stitching_piece_info.calculate_overlap_coefficient(numb_pieces_in_each_segment)

                # Get the max overlap between pairs of segments
                for segment_j in xrange(0, numb_segments):

                    if segment_i == segment_j:
                        continue
                    # Update if the new value is greater
                    if self._asymmetric_overlap_matrix[segment_i, segment_j] < overlap[segment_j]:
                        self._asymmetric_overlap_matrix[segment_i, segment_j] = overlap[segment_j]
        MultiPuzzleSolver._log_numpy_matrix("Asymmetric Segment Overlap Matrix:", self._asymmetric_overlap_matrix)

        # Calculate the similarity matrix
        self._segment_similarity_matrix = np.full((numb_segments, numb_segments), fill_value=-1, dtype=np.float)
        for segment_i in xrange(0, numb_segments):
            for segment_j in xrange(segment_i + 1, numb_segments):
                similarity = self._asymmetric_overlap_matrix[segment_i, segment_j] \
                             + self._asymmetric_overlap_matrix[segment_j, segment_i]
                similarity /= 2
                self._segment_similarity_matrix[segment_i, segment_j] = similarity
        MultiPuzzleSolver._log_numpy_matrix("Segment Similarity Matrix", self._segment_similarity_matrix)

        if MultiPuzzleSolver._ALLOW_POST_SIMILARITY_MATRIX_CALCULATION_PICKLE_EXPORT:
            self._pickle_export_after_similarity_matrix_calculation()

    @staticmethod
    def _log_numpy_matrix(matrix_description_message, numpy_matrix):
        """
        Helper function for logging Numpy matrix values.

        Args:
            matrix_description_message (str): Description message to go with the matrix printing
            numpy_matrix (np[float]): Numpy array to be logged
        """
        string_io = cStringIO.StringIO()
        print >> string_io, matrix_description_message
        print >> string_io, numpy_matrix
        logging.critical(string_io.getvalue())
        string_io.close()

    def _perform_placement_with_final_seed_pieces(self):
        """
        This is run after the final set of seed pieces have been found.  It runs the final solver segments the piece
        into disjoint sets.  The number of disjoint sets is based off the number of seed pieces selected.
        """
        self._paikin_tal_solver.restore_initial_placer_settings_and_distances()
        self._paikin_tal_solver.allow_placement_of_all_pieces()

        self._paikin_tal_solver.run_solver_with_specified_seeds(self._final_starting_pieces)

    def _build_output_puzzles(self):
        """
        Constructs the final output puzzles that are to be returned by the multiple puzzle solver.

        Returns (List[Puzzle]):
            List of the final solved puzzles.
        """
        solved_puzzles, _ = self._paikin_tal_solver.get_solved_puzzles()

        # Merge the pieces into a set of solved puzzles
        output_puzzles = [Puzzle.reconstruct_from_pieces(solved_puzzles[i], i) for i in xrange(0, len(solved_puzzles))]

        # Optionally export the solved image files.
        if MultiPuzzleSolver._SAVE_FINAL_PUZZLE_IMAGES:
            self._output_reconstructed_puzzle_image_files(output_puzzles)

        return output_puzzles

    def _output_reconstructed_puzzle_image_files(self, output_puzzles):
        """
        Saves images files for each of the final reconstructed output puzzles.

        Args:
            output_puzzles (List[Puzzle]): Set of final solved puzzles
        """
        for puzzle in output_puzzles:
            filename = Puzzle.make_image_filename(self._image_filenames,
                                                  "multipuzzle_reconstructed",
                                                  Puzzle.OUTPUT_IMAGE_DIRECTORY,
                                                  self._paikin_tal_solver.puzzle_type,
                                                  self._start_timestamp,
                                                  puzzle_id=puzzle.id_number)
            Puzzle.save_to_file(puzzle, filename)

    def _pickle_export_after_segmentation_round(self):
        """
        Export the entire multipuzzle solver via pickle.
        """
        self._local_pickle_expert_helper("segment_round_%d" % self._numb_segmentation_rounds)

    def _pickle_export_after_segmentation_puzzle_placement(self):
        """
        Export the entire multipuzzle solver via pickle.
        """
        self._local_pickle_expert_helper(MultiPuzzleSolver._POST_SEGMENTATION_PUZZLE_PLACEMENT_FILE_DESCRIPTOR
                                         % self._numb_segmentation_rounds)

    def _pickle_export_after_all_segmentation_completed(self):
        """
        Exports the multipuzzle solver after segmentation is completed.
        """
        self._local_pickle_expert_helper(MultiPuzzleSolver._POST_SEGMENTATION_COMPLETED_PICKLE_FILE_DESCRIPTOR)

    def _pickle_export_after_stitching_piece_solving(self):
        """
        Exports the multipuzzle solver after segmentation is completed.
        """
        self._local_pickle_expert_helper(MultiPuzzleSolver._POST_STITCHING_PIECE_SOLVING_PICKLE_FILE_DESCRIPTOR)

    def _pickle_export_after_similarity_matrix_calculation(self):
        """
        Exports the multipuzzle solver after the similarity matrix is calculated.
        """
        self._local_pickle_expert_helper(MultiPuzzleSolver._POST_SIMILARITY_MATRIX_CALCULATION_PICKLE_FILE_DESCRIPTOR)

    def _pickle_export_after_hierarchical_clustering(self):
        """
        Exports the multipuzzle solver after hierarchical clustering is completed.
        """
        self._local_pickle_expert_helper(MultiPuzzleSolver._POST_HIERARCHICAL_CLUSTERING_PICKLE_FILE_DESCRIPTOR)

    def _pickle_export_after_select_starting_pieces(self):
        """
        Exports the multipuzzle solver after hierarchical clustering is completed.
        """
        self._local_pickle_expert_helper(MultiPuzzleSolver._POST_SELECT_STARTING_PIECES_PICKLE_FILE_DESCRIPTOR)

    def _local_pickle_expert_helper(self, pickle_file_descriptor):
        """
        Helper function that handles the pickle export for a specific file description.

        Args:
            pickle_file_descriptor (str): File descriptor for the pickle file
        """
        pickle_filename = PickleHelper.build_filename(pickle_file_descriptor,
                                                      self._image_filenames,
                                                      self._paikin_tal_solver.puzzle_type)
        PickleHelper.exporter(self, pickle_filename)

    def _make_image_filename(self, filename_descriptor):
        """
        Creates an image file name with the specified descriptor included.

        Args:
            filename_descriptor (str): Filename descriptor for the output file

        Returns (str):
            Standardized filename with directory
        """
        return Puzzle.make_image_filename(self._image_filenames, filename_descriptor, Puzzle.OUTPUT_IMAGE_DIRECTORY,
                                          self._paikin_tal_solver.puzzle_type, self._start_timestamp)

    def _save_single_solved_puzzle_to_file(self, segmentation_round):
        """
        Saves the solved image when perform single puzzle solving to a file.

        Args:
            segmentation_round (int): iteration count for the segmentation
        """
        solved_puzzle, _ = self._paikin_tal_solver.get_solved_puzzles()
        # Reconstruct the puzzle
        new_puzzle = Puzzle.reconstruct_from_pieces(solved_puzzle[0], 0)

        # Store the reconstructed segmented image
        max_numb_zero_padding = 4
        filename_descriptor = "single_puzzle_round_" + str(segmentation_round).zfill(max_numb_zero_padding)
        new_puzzle.save_to_file(self._make_image_filename(filename_descriptor))

        # Store the best buddy image.
        filename_descriptor += "_best_buddy_acc"
        output_filename = self._make_image_filename(filename_descriptor)
        self._paikin_tal_solver.best_buddy_accuracy.output_results_images(self._image_filenames, [new_puzzle],
                                                                          self._paikin_tal_solver.puzzle_type,
                                                                          self._start_timestamp,
                                                                          output_filenames=[output_filename])

    def _save_stitching_piece_solved_puzzle_to_file(self, stitching_piece_segment_info):
        """
        Saves the solved image when perform single puzzle solving to a file.

        Args:
            stitching_piece_segment_info (StitchingPieceInfo): Information on the stitching piece information.
        """
        solved_puzzle, _ = self._paikin_tal_solver.get_solved_puzzles()
        # Reconstruct the puzzle
        new_puzzle = Puzzle.reconstruct_from_pieces(solved_puzzle[0], 0)

        max_numb_zero_padding = 4

        # Store the reconstructed image
        segment_id = stitching_piece_segment_info.segment_numb
        filename_descriptor = "segment_" + str(segment_id).zfill(max_numb_zero_padding)

        stitching_piece_id = stitching_piece_segment_info.piece_id
        filename_descriptor += "_stitching_piece_id_" + str(stitching_piece_id).zfill(max_numb_zero_padding)

        # Build the filename and output to a file
        new_puzzle.save_to_file(self._make_image_filename(filename_descriptor))

    def _process_solved_segments(self, solved_segments):
        """
        Processes all the solved segments.  It selects those segments from the solved puzzle that will be used
        by the solver to determine the seed pieces.

        <b>Note</b>: All pieces that exist in the puzzle may not be in the solved segment(s) since some
            pieces may be iteratively excluded.

        Args:
            solved_segments (List[PuzzleSegment]): A list of the segments found by the single puzzle solver.

        Returns (int): Maximum segment size
        """
        # Get the maximum segment size
        max_segment_size = max([segment.numb_pieces for segment in solved_segments])

        for segment in solved_segments:
            if segment.numb_pieces >= max_segment_size / 2 \
                    and segment.numb_pieces >= MultiPuzzleSolver._MINIMUM_SEGMENT_SIZE:
                self._select_segment_for_solver(segment)

        return max_segment_size

    def _select_segment_for_solver(self, selected_segment):
        """
        Segment is selected for use by the solver.

        Args:
            selected_segment (PuzzleSegment): Segment selected to be used by the solver
        """
        initial_segment_id = selected_segment.id_number

        # Add the segment to the list
        selected_segment.update_segment_for_multipuzzle_solver(len(self._segments))
        self._segments.append(selected_segment)

        # Store the segment
        for piece_id in selected_segment.get_piece_ids():
            self._paikin_tal_solver.disallow_piece_placement(piece_id)
            # Store the mapping of piece to segment.
            key = PuzzlePiece.create_key(piece_id)
            self._piece_id_to_segment_map[key] = selected_segment.id_number

        logging.info("Saved segment #%d has %d pieces." % (selected_segment.id_number, selected_segment.numb_pieces))

        # Optionally output the segment image to a file.
        if MultiPuzzleSolver._SAVE_SELECTED_SEGMENTS_TO_AN_IMAGE_FILE:
            zfill_width = 4
            filename_descriptor = "segment_number_" + str(selected_segment.id_number).zfill(zfill_width)
            filename_descriptor += "_puzzle_round_" + str(self._numb_segmentation_rounds).zfill(zfill_width)

            single_puzzle_id = 0
            self._paikin_tal_solver.save_segment_to_image_file(single_puzzle_id, initial_segment_id,
                                                               filename_descriptor, self._image_filenames,
                                                               self._start_timestamp)

    def _get_stitching_pieces(self):
        """
        Iterates through all of the segments found by the initial solver, and builds a list of the piece identification
        numbers for all of the stitching pieces.

        Returns (List[List[StitchingPieceSegment]]): List of stitching pieces for each segment.
        """
        all_stitching_pieces = []
        existing_piece_ids = {}
        for segment_id_numb in xrange(0, len(self._segments)):

            # Verify the identification number matches what is stored in the array
            if config.PERFORM_ASSERT_CHECKS:
                assert segment_id_numb == self._segments[segment_id_numb].id_number

            # Separate the stitching pieces by segments
            all_stitching_pieces.append([])
            segment_stitching_pieces = self._segments[segment_id_numb].select_pieces_for_segment_stitching()
            for segmentation_piece_id in segment_stitching_pieces:
                all_stitching_pieces[segment_id_numb].append(StitchingPieceInfo(segmentation_piece_id, segment_id_numb))

                # Verify no duplicate stitching pieces
                if config.PERFORM_ASSERT_CHECKS:
                    key = PuzzlePiece.create_key(segmentation_piece_id)
                    assert key not in existing_piece_ids
                    existing_piece_ids[key] = segmentation_piece_id

        return all_stitching_pieces

    def _perform_hierarchical_clustering(self):
        """
        Performs hierarchical clustering to merge the puzzle segments.
        """
        # Perform the hierarchical clustering output.
        temp_similarity_matrix = copy.copy(self._segment_similarity_matrix)
        hierarchical_results = HierarchicalClustering.run(self._segments, temp_similarity_matrix)
        self._segment_clusters = hierarchical_results[0]
        self._final_cluster_similarity_matrix = hierarchical_results[1]

        self._log_clustering_result()

        if MultiPuzzleSolver._ALLOW_POST_HIERARCHICAL_CLUSTERING_PICKLE_EXPORT:
            self._pickle_export_after_hierarchical_clustering()

    def _select_starting_pieces_from_clusters(self):
        """
        From the segment clusters, this function builds the starting pieces for each of the output puzzles.
        """

        piece_to_cluster_map = {}
        cluster_has_seed = {}
        for cluster in self._segment_clusters:
            # Used to determine if each cluster has a starting piece
            cluster_has_seed[MultiPuzzleSolver.create_cluster_key(cluster.id_number)] = False

            # Build a map from piece ID number to cluster
            for piece_id in cluster.get_pieces():
                key = PuzzlePiece.create_key(piece_id)

                # Verify no duplicates
                if config.PERFORM_ASSERT_CHECKS:
                    assert key not in piece_to_cluster_map
                piece_to_cluster_map[key] = cluster.id_number

        # Get the starting piece ordering
        starting_piece_ordering = self._paikin_tal_solver.get_initial_starting_piece_ordering()

        # Find the start pieces.
        self._final_starting_pieces = []
        starting_piece_cnt = 0
        while len(self._final_starting_pieces) != len(self._segment_clusters):

            # Get the cluster number (if any) for the starting piece candidate
            starting_piece_candidate = starting_piece_ordering[starting_piece_cnt]
            piece_key = PuzzlePiece.create_key(starting_piece_candidate)
            try:
                cluster_number = piece_to_cluster_map[piece_key]
                cluster_key = MultiPuzzleSolver.create_cluster_key(cluster_number)

                # If the cluster has no seed, use this seed piece
                if not cluster_has_seed[cluster_key]:
                    self._final_starting_pieces.append(starting_piece_candidate)
                    cluster_has_seed[cluster_key] = True

                    # Log the clustering information
                    piece_initial_puzzle = self._paikin_tal_solver.get_piece_original_puzzle_id(starting_piece_candidate)
                    logging.info("Seed piece for cluster %d is piece id %d from initial puzzle %d" % (cluster_number,
                                                                                                      starting_piece_candidate,
                                                                                                      piece_initial_puzzle))
            except KeyError:
                pass
            starting_piece_cnt += 1

        if MultiPuzzleSolver._ALLOW_POST_SELECT_STARTING_PIECES_PICKLE_EXPORT:
            self._pickle_export_after_select_starting_pieces()

    @staticmethod
    def create_cluster_key(cluster_id):
        """
        Creates and returns a key for a cluster for use in a dictionary.
        Args:
            cluster_id (int): Cluster identification number

        Returns (str): Key for the cluster
        """
        return str(cluster_id)

    def _log_clustering_result(self):
        """
        Logs information on the hierarchical clusters found during the hierarchical clustering.
        """
        logging.info("Re-logging segment details for reference.")
        self._log_segmentation_results()

        string_io = cStringIO.StringIO()
        print >>string_io, "Number of Clusters: %d" % len(self._segment_clusters)
        # Log the segments in cluster
        for cluster in self._segment_clusters:
            # Build a list of segments in the cluster.
            pieces_per_input_puzzle = [0] * len(self._image_filenames)
            segment_list = ""
            for segment_id in cluster.get_segments():
                if segment_list:
                    segment_list += ", "
                # Get the pieces in the cluster from the original puzzle
                segment_list += str(segment_id)
                for piece_id in self._segments[segment_id].get_piece_ids():
                    original_puzzle_id = self._paikin_tal_solver.get_piece_original_puzzle_id(piece_id)
                    pieces_per_input_puzzle[original_puzzle_id] += 1

            # Print the number of pieces from each puzzle.
            print >> string_io, ("\tCluster #%d contains segments: " + segment_list) % cluster.id_number
            for puzzle_id in xrange(0, len(self._image_filenames)):
                print >> string_io, "\t\tPuzzle #%d Piece Count: %d" % (puzzle_id, pieces_per_input_puzzle[puzzle_id])
            print >> string_io, ""

        logging.critical(string_io.getvalue())
        string_io.close()

    def _log_segmentation_results(self):
        """
        Logs information on the hierarchical clusters found during the hierarchical clustering.
        """
        string_io = cStringIO.StringIO()

        print >>string_io, "Number of Segments: %d" % len(self._segments)
        for segment in self._segments:
            print >> string_io, "\tSegment #%d contains %d pieces." % (segment.id_number, len(segment.get_piece_ids()))
            # Get the pieces in the cluster from the original puzzle
            pieces_per_input_puzzle = [0] * len(self._image_filenames)
            for piece_id in segment.get_piece_ids():
                original_puzzle_id = self._paikin_tal_solver.get_piece_original_puzzle_id(piece_id)
                pieces_per_input_puzzle[original_puzzle_id] += 1

            # Print the number of pieces from each puzzle.
            for puzzle_id in xrange(0, len(self._image_filenames)):
                print >> string_io, "\t\tPuzzle #%d Piece Count: %d" % (puzzle_id, pieces_per_input_puzzle[puzzle_id])
            print >> string_io, ""

        logging.critical(string_io.getvalue())
        string_io.close()

    def reset_timestamp(self):
        """
        Resets the time stamp for the puzzle solver for debug functions.
        """
        self._start_timestamp = time.time()

    @staticmethod
    def run_imported_segmentation_round(image_filenames, puzzle_type, segmentation_round_numb):
        """
        Debug method that imports a pickle file for the specified image files, puzzle type, and segmentation round
        and then runs the initial segmentation starting after the specified round.

        Args:
            image_filenames (List[str]): List of paths to image file names
            puzzle_type (PuzzleType): Solver puzzle type
            segmentation_round_numb (int): Segmentation round number
        """

        pickle_file_descriptor = "segment_round_%d" % segmentation_round_numb
        pickle_filename = PickleHelper.build_filename(pickle_file_descriptor, image_filenames, puzzle_type)

        solver = PickleHelper.importer(pickle_filename)
        # noinspection PyProtectedMember
        solver.reset_timestamp()

        # noinspection PyProtectedMember
        solver._find_initial_segments(skip_initialization=True)

    @staticmethod
    def run_imported_segmentation_experiment(image_filenames, puzzle_type, segmentation_round_numb):
        """
        Debug method that imports a pickle file for the specified image files, puzzle type, and segmentation round
        and then runs the initial segmentation starting after the specified round.

        Args:
            image_filenames (List[str]): List of paths to image file names
            puzzle_type (PuzzleType): Solver puzzle type
            segmentation_round_numb (int): Segmentation round number
        """

        file_descriptor = MultiPuzzleSolver._POST_SEGMENTATION_PUZZLE_PLACEMENT_FILE_DESCRIPTOR % segmentation_round_numb
        pickle_filename = PickleHelper.build_filename(file_descriptor, image_filenames, puzzle_type)

        solver = PickleHelper.importer(pickle_filename)
        # noinspection PyProtectedMember
        solver.reset_timestamp()

        # noinspection PyProtectedMember
        solver._find_initial_segments(go_directly_to_segmentation=True)

    @staticmethod
    def run_imported_stitching_piece_solving(image_filenames, puzzle_type):
        """
        Debug method that is used to test the stitching piece solving.

        Args:
            image_filenames (List[str]): List of paths to image file names
            puzzle_type (PuzzleType): Solver puzzle type
        """
        pickle_filename = PickleHelper.build_filename(MultiPuzzleSolver._POST_SEGMENTATION_COMPLETED_PICKLE_FILE_DESCRIPTOR,
                                                      image_filenames, puzzle_type)
        solver = PickleHelper.importer(pickle_filename)
        # noinspection PyProtectedMember
        solver.reset_timestamp()

        # noinspection PyProtectedMember
        solver._perform_stitching_piece_solving()

    @staticmethod
    def run_imported_similarity_matrix_calculation(image_filenames, puzzle_type):
        """
        Debug method that imports a pickle file for the specified image files, puzzle type, and segmentation round
        and then runs the initial segmentation starting after the specified round.

        Args:
            image_filenames (List[str]): List of paths to image file names
            puzzle_type (PuzzleType): Solver puzzle type
        """
        pickle_file_descriptor = MultiPuzzleSolver._POST_STITCHING_PIECE_SOLVING_PICKLE_FILE_DESCRIPTOR
        pickle_filename = PickleHelper.build_filename(pickle_file_descriptor, image_filenames, puzzle_type)

        solver = PickleHelper.importer(pickle_filename)
        # noinspection PyProtectedMember
        solver.reset_timestamp()

        # noinspection PyProtectedMember
        solver._build_similarity_matrix()

    @staticmethod
    def run_imported_hierarchical_clustering(image_filenames, puzzle_type):
        """
        Debug method that imports a pickle file after the similarity matrix was built.  It then runs hierarchical
        clustering on that imported pickle file.

        Note: The pickle file name is built from the image file names and the puzzle type.

        Args:
            image_filenames (List[str]): List of paths to image file names
            puzzle_type (PuzzleType): Solver puzzle type
        """
        pickle_file_descriptor = MultiPuzzleSolver._POST_SIMILARITY_MATRIX_CALCULATION_PICKLE_FILE_DESCRIPTOR
        pickle_filename = PickleHelper.build_filename(pickle_file_descriptor, image_filenames, puzzle_type)

        solver = PickleHelper.importer(pickle_filename)
        solver.reset_timestamp()

        # noinspection PyProtectedMember
        solver._perform_hierarchical_clustering()

    @staticmethod
    def run_imported_select_starting_pieces(image_filenames, puzzle_type):
        """
        Debug method that imports a pickle file after hierarchical clustering was completed.  It then selects the
        seed pieces as inputs to the solver..

        Note: The pickle file name is built from the image file names and the puzzle type.

        Args:
            image_filenames (List[str]): List of paths to image file names
            puzzle_type (PuzzleType): Solver puzzle type
        """
        pickle_file_descriptor = MultiPuzzleSolver._POST_HIERARCHICAL_CLUSTERING_PICKLE_FILE_DESCRIPTOR
        pickle_filename = PickleHelper.build_filename(pickle_file_descriptor, image_filenames, puzzle_type)

        solver = PickleHelper.importer(pickle_filename)
        solver.reset_timestamp()

        # noinspection PyProtectedMember
        solver._select_starting_pieces_from_clusters()

    @staticmethod
    def run_imported_final_puzzle_solving(image_filenames, puzzle_type):
        """
        Debug method that imports a pickle file after hierarchical clustering was completed.  It then selects the
        seed pieces as inputs to the solver..

        Note: The pickle file name is built from the image file names and the puzzle type.

        Args:
            image_filenames (List[str]): List of paths to image file names
            puzzle_type (PuzzleType): Solver puzzle type
        """
        pickle_file_descriptor = MultiPuzzleSolver._POST_SELECT_STARTING_PIECES_PICKLE_FILE_DESCRIPTOR
        pickle_filename = PickleHelper.build_filename(pickle_file_descriptor, image_filenames, puzzle_type)

        solver = PickleHelper.importer(pickle_filename)
        solver.reset_timestamp()

        # noinspection PyProtectedMember
        solver._perform_placement_with_final_seed_pieces()
        # noinspection PyProtectedMember
        solver._final_puzzles = solver._build_output_puzzles()
