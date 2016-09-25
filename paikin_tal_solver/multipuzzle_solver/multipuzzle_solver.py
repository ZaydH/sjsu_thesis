import time

from hammoudeh_puzzle.puzzle_importer import Puzzle
from paikin_tal_solver.solver import PaikinTalSolver


class MultiPuzzleSolver(object):

    _MINIMUM_SEGMENT_SIZE = 10

    _SAVE_EACH_SINGLE_PUZZLE_RESULT_TO_A_FILE = False

    def __init__(self, pieces, distance_function, puzzle_type=None):
        """
        Constructor for the Multi-Puzzle Solver.

        Args:
            pieces (List[PuzzlePiece])): List of puzzle pieces
            distance_function: Calculates the distance between two PuzzlePiece objects.
            puzzle_type (PuzzleType): Type of Paikin Tal Puzzle
        """
        self._start_timestamp = time.time()

        self._numb_pieces = len(pieces)

        self._segments = []

        # This maps piece identification numbers to segments.
        self._piece_id_to_segment_map = {}

        # Build the Paikin Tal Solver
        self._paikin_tal_solver = PaikinTalSolver(pieces, distance_function, puzzle_type=puzzle_type)

    def run(self):

        self._find_initial_segments()

    def _find_initial_segments(self):
        """
        Through iterative single puzzle placing, this function finds a set of segments.
        """
        self._paikin_tal_solver.allow_placement_of_all_pieces()

        segmentation_round = 0

        # Essentially a Do-While loop
        while True:
            segmentation_round += 1

            # Perform placement as if there is only a single puzzle
            self._paikin_tal_solver.run_single_puzzle_solver()

            # Get the segments from this iteration of the loop
            solved_segments = self._paikin_tal_solver.segment()
            max_segment_size = self._process_solved_segments(solved_segments)

            if MultiPuzzleSolver._SAVE_EACH_SINGLE_PUZZLE_RESULT_TO_A_FILE:
                self._save_single_solved_puzzle_to_file(segmentation_round)

            if max_segment_size < MultiPuzzleSolver._MINIMUM_SEGMENT_SIZE:
                break

    def _save_single_solved_puzzle_to_file(self, segmentation_round):
        """
        Saves the solved image to a file.

        Args:
            segmentation_round (int): iteration count for the segmentation
        """
        solved_puzzle, _ = self._paikin_tal_solver.get_solved_puzzles()
        # Reconstruct the puzzle
        new_puzzle = Puzzle.reconstruct_from_pieces(solved_puzzle[0], 0)

        # Store the reconstructed image
        max_numb_zero_padding = 4
        filename_descriptor = "single_puzzle_round" + str(segmentation_round).zfill(max_numb_zero_padding)
        filename = Puzzle.make_image_filename(filename_descriptor, Puzzle.OUTPUT_IMAGE_DIRECTORY,
                                              self._paikin_tal_solver.puzzle_type, self._start_timestamp,
                                              puzzle_id=str(0))
        new_puzzle.save_to_file(filename)

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
            if segment.numb_pieces >= max_segment_size / 2:
                self._select_segment_for_solver(segment)

        return max_segment_size

    def _select_segment_for_solver(self, selected_segment):
        """
        Segment is selected for use by the solver.

        Args:
            selected_segment (PuzzleSegment): Segment selected to be used by the solver
        """
        self._segments.append(selected_segment)
        for piece_id in selected_segment.get_piece_ids():
            self._paikin_tal_solver.disallow_piece_placement(piece_id)

    def _place_identification_pieces(self):
        pass

    def _continue_finding_initial_segments(self):
        pass
