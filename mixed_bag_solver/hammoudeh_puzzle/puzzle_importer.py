"""Jigsaw Puzzle Object
"""
import cStringIO
import copy
import datetime
import logging
import math
import os
import random

import cv2  # OpenCV
import numpy as np
from enum import Enum

from hammoudeh_puzzle import config
from hammoudeh_puzzle.puzzle_piece import PuzzlePiece, PuzzlePieceRotation, PuzzlePieceSide, SolidColor


class PuzzleSolver(Enum):
    PaikinTal = "paikin_tal"
    MultiPuzzle = "multipuzzle"


class DirectAccuracyResultType(Enum):
    AccuracyScore = 1
    DifferentPuzzleScore = 2
    WrongRotationScore = 3
    WrongLocationScore = 4
    MissingPieceScore = 5


class ResultAccuracyMetric(Enum):
    StandardDirectAccuracy = 1
    ModifiedDirectAccuracy = 2
    ModifiedNeighborAccuracy = 3


class NeighborAccuracyErrorType(Enum):
    MissingPieceCount = 1
    DifferentPuzzleCount = 2


class PuzzleType(Enum):
    """
    Type of the puzzle to solve.  Type 1 has no piece rotation while type 2 allows piece rotation.
    """

    type1 = 1
    type2 = 2


class PuzzleDimensions(object):
    """
    Stores the information regarding the puzzle dimensions including its top left and bottom right corners
    as well as its total size.
    """

    def __init__(self, puzzle_id, starting_point):
        self.puzzle_id = puzzle_id
        self.top_left = [starting_point[0], starting_point[1]]
        self.bottom_right = [starting_point[0], starting_point[1]]
        self.total_size = (1, 1)

    def update_dimensions(self):
        """
        Puzzle Dimensions Updater

        Updates the total dimensions of the puzzle.
        """
        self.total_size = (self.bottom_right[0] - self.top_left[0] + 1,
                           self.bottom_right[1] - self.top_left[1] + 1)


class PuzzleResultsCollection(object):
    """
    Stores all the puzzle results information in a single collection.
    """

    _PERFORM_ASSERT_CHECKS = True

    def __init__(self, puzzle_solver_type, puzzle_type, solved_puzzles, image_file_paths):
        """
        Constructs the puzzle results information.

        Args:
            puzzle_solver_type (PuzzleSolver): Type of Solver
            solved_puzzles (List[List[PuzzlePieces]]): List of pieces in each of the puzzles output by the solver
            image_file_paths (List[Str])): Names of the image files.
        """

        self._puzzle_solver_type = puzzle_solver_type
        self._puzzle_type = puzzle_type
        self._image_filenames = image_file_paths
        self._numb_output_puzzles = len(solved_puzzles)

        self._puzzle_results = []

        # Iterate through all the solved puzzles
        for set_of_pieces in solved_puzzles:
            # Iterate through all of the pieces in the puzzle
            for piece in set_of_pieces:

                # Iterate through all the pieces
                puzzle_exists = False
                for i in range(0, len(self._puzzle_results)):
                    # Check if the puzzle ID matches this set of results information.
                    if piece.original_puzzle_id == self._puzzle_results[i].original_id_number:
                        puzzle_exists = True
                        self._puzzle_results[i].numb_pieces += 1
                        continue

                # If the puzzle does not exist, then create a results information
                if not puzzle_exists:
                    new_puzzle = PuzzleResultsInformation(piece.original_puzzle_id,
                                                          image_file_paths[piece.original_puzzle_id])
                    new_puzzle.numb_pieces = 1
                    self._puzzle_results.append(new_puzzle)

        # Sort by original puzzle id
        self._puzzle_results = sorted(self._puzzle_results, key=lambda result: result.original_id_number)

    def calculate_accuracies(self, solved_puzzles):
        """
        Results Accuracy Calculator

        Calculates the standard direct, modified direct, and modified neighbor accuracies for a set of solved
        puzzles.

        Args:
            solved_puzzles (List[Puzzle]): A set of solved puzzles

        """

        # Go through all of the original puzzles
        for i in xrange(0, len(self._puzzle_results)):
            for puzzle in solved_puzzles:
                # Update the puzzle results
                self._puzzle_results[i].resolve_direct_accuracies(puzzle)
                self._puzzle_results[i].resolve_neighbor_accuracies(puzzle)

    @property
    def results(self):
        """
        Puzzle Results Accessor

        Returns (PuzzleResultsInformation):
            Puzzle results information for a single puzzle.
        """
        return self._puzzle_results

    def output_results_images(self, puzzle_solver_type, input_image_filenames, solved_puzzles, puzzle_type, timestamp):
        """
        Creates images showing the results of the image.

        Args:
            puzzle_solver_type (PuzzleSolver): Type of solver being run
            input_image_filenames (List[str]): List of image files and directories.
            solved_puzzles (List[Puzzle]): A set of solved puzzles.
            puzzle_type (PuzzleType): Type of the solved image
            timestamp (float): Timestamp

        """
        # Standard direct accuracy first then modified
        for original_puzzle_id in xrange(0, len(self._puzzle_results)):

            # Get the individual results and puzzle
            results = self._puzzle_results[original_puzzle_id]

            for i in xrange(0, 2):

                # Select the accuracy type
                direct_acc = None
                descriptor = ""
                if i == 0:
                    direct_acc = results.standard_direct_accuracy
                    descriptor = "std_direct_acc"
                elif i == 1:
                    direct_acc = results.modified_direct_accuracy
                    descriptor = "mod_direct_acc"
                # Verify a direct accuracy was selected
                assert direct_acc is not None

                # Get the solved puzzle that best matches this original image.
                solved_puzzle = solved_puzzles[direct_acc.solved_puzzle_id]

                # Verify the original puzzle id matches this set of direct accuracy
                if PuzzleResultsCollection._PERFORM_ASSERT_CHECKS:
                    assert direct_acc.original_puzzle_id == original_puzzle_id

                # Set the piece coloring using the selected accuracy type
                solved_puzzle.set_piece_color_for_direct_accuracy(direct_acc)

                # Determine whether the puzzle id should be used in the filename
                solved_puzzle_id = solved_puzzle.id_number if len(self._puzzle_results) > 1 else None
                output_filename = Puzzle.make_image_filename(puzzle_solver_type, input_image_filenames, descriptor,
                                                             Puzzle.OUTPUT_IMAGE_DIRECTORY, puzzle_type,
                                                             timestamp, orig_img_filename=results.original_filename,
                                                             puzzle_id=solved_puzzle_id)
                # Stores the results to a file.
                solved_puzzle.build_puzzle_image(use_results_coloring=True)
                solved_puzzle.save_to_file(output_filename)

            # Get the individual results and puzzle
            neighbor_acc = results.modified_neighbor_accuracy

            # Iterate each puzzle piece and set its color for each side
            solved_puzzle = solved_puzzles[neighbor_acc.solved_puzzle_id]

            # Verify the original puzzle id matches this set of direct accuracy
            if PuzzleResultsCollection._PERFORM_ASSERT_CHECKS:
                assert neighbor_acc.original_puzzle_id == original_puzzle_id

            # Go through and color each piece
            for piece in solved_puzzle.pieces:
                piece.reset_image_coloring_for_polygons()
                for side in PuzzlePieceSide.get_all_sides():
                    coloring = neighbor_acc.get_piece_side_result(piece.id_number, side)
                    piece.results_image_polygon_coloring(side, coloring)

            # Define the output filename
            descriptor = "neighbor_acc"
            solved_puzzle_id = solved_puzzle.id_number if len(self._puzzle_results) > 1 else None
            output_filename = Puzzle.make_image_filename(puzzle_solver_type, input_image_filenames, descriptor,
                                                         Puzzle.OUTPUT_IMAGE_DIRECTORY, puzzle_type,
                                                         timestamp, orig_img_filename=results.original_filename,
                                                         puzzle_id=solved_puzzle_id)

            # Stores the results to a file.
            solved_puzzle.build_puzzle_image(use_results_coloring=True)
            solved_puzzle.save_to_file(output_filename)

    def save_results_to_file(self):
        """
        Saves the results information to a CSV file for analysis.
        """

        # Build the header line.
        if not os.path.exists(config.RESULTS_FILE):
            PuzzleResultsCollection._write_results_file_header_line()

        with open(config.RESULTS_FILE, "a") as results_stream:

            # Close the previous solver
            results_stream.write('\n')

            results_stream.write(self._puzzle_solver_type.value)

            results_stream.write("," + str(self._puzzle_type.value))

            # Write the image file names
            results_stream.write(",")
            for (idx, image_file) in enumerate(self._image_filenames):
                if idx > 0:
                    results_stream.write(" | ")
                results_stream.write(Puzzle.get_filename_without_extension(image_file))

            # Print information on the puzzle counts
            numb_input_puzzles = len(self._image_filenames)
            numb_solved_puzzles = self._numb_output_puzzles
            for count_info in [numb_input_puzzles, numb_solved_puzzles, numb_input_puzzles - numb_solved_puzzles]:
                results_stream.write("," + str(count_info))

            # Print the Puzzle Results Information
            for puzzle_result in self._puzzle_results:
                results_stream.write(",Input Puzzle #" + str(puzzle_result.original_id_number))
                results_stream.write("," + str(puzzle_result.numb_pieces))

                # Add the standard and modified direct accuracy score information
                accuracy_metrics = [ResultAccuracyMetric.StandardDirectAccuracy, ResultAccuracyMetric.ModifiedDirectAccuracy]
                for acc_metric in accuracy_metrics:
                    results_stream.write(",Solved Puzzle #" + str(puzzle_result.get_accuracy_solved_puzzle_id(acc_metric)))
                    results_stream.write("," + str(puzzle_result.get_direct_accuracy_solved_puzzle_piece_count(acc_metric)))

                    # Define the direct accuracy score parameters to be printed
                    scores = [DirectAccuracyResultType.AccuracyScore, DirectAccuracyResultType.DifferentPuzzleScore,
                              DirectAccuracyResultType.WrongLocationScore, DirectAccuracyResultType.WrongRotationScore,
                              DirectAccuracyResultType.MissingPieceScore]
                    for score in scores:
                        results_stream.write("," + str(puzzle_result.get_direct_accuracy_score(acc_metric, score)))

                # Add the neighbor accuracy information
                solved_puzzle_id = puzzle_result.get_accuracy_solved_puzzle_id(ResultAccuracyMetric.ModifiedNeighborAccuracy)
                results_stream.write(",Solved Puzzle#" + str(solved_puzzle_id))

                # Print the piece count for the solved puzzle
                solved_puzzle_piece_count = puzzle_result.modified_neighbor_accuracy.total_numb_pieces_in_solved_puzzle
                results_stream.write("," + str(solved_puzzle_piece_count))

                # Print result information
                results_stream.write("," + str(puzzle_result.get_neighbor_accuracy_score()))
                neighbor_error_types = [NeighborAccuracyErrorType.MissingPieceCount, NeighborAccuracyErrorType.DifferentPuzzleCount]
                for error_type in neighbor_error_types:
                    results_stream.write("," + str(puzzle_result.get_neighbor_error_piece_count(error_type)))

    @staticmethod
    def _write_results_file_header_line():
        """
        Writes the header line for the results file
        """
        with open(config.RESULTS_FILE, "w") as results_stream:
            headers = ["Solver Type", "Puzzle Type", "Image Filenames", "Number of Input Puzzles", "Number of Output Puzzles",
                       "Diff Between Output and Expected Puzzle Count", "Input Puzzle #", "Input Piece Count",
                       "Standard Direct Accuracy Puzzle #", "Standard Direct Accuracy Solved Puzzle Piece Count",
                       "Standard Direct Accuracy Score", "Standard Direct Accuracy Different Puzzle",
                       "Standard Direct Accuracy Wrong Location", "Standard Direct Accuracy Wrong Rotation",
                       "Standard Direct Accuracy Piece Missing Count", "Modified Direct Accuracy Puzzle #",
                       "Modified Direct Accuracy Solved Puzzle Piece Count", "Modified Direct Accuracy Score",
                       "Modified Direct Accuracy Different Puzzle", "Modified Direct Accuracy Wrong Location",
                       "Modified Direct Accuracy Wrong Rotation", "Modified Direct Accuracy Piece Missing Count",
                       "Modified Neighbor Accuracy Puzzle #", "Modified Neighbor Accuracy Solved Puzzle Piece Count",
                       "Modified Neighbor Accuracy Score", "Modified Neighbor Accuracy Missing Piece Count",
                       "Modified Neighbor Accuracy Edges from Different Puzzle"]
            # Print all header fields
            for (idx, field) in enumerate(headers):
                # Comma separate
                if idx > 0:
                    results_stream.write(",")
                # Write the field
                results_stream.write(field)

    def print_results(self):
        """
        Solver Accuracy Results Printer

        Prints the accuracy results of a solver to the console.
        """

        string_io = cStringIO.StringIO()
        # Iterate through each puzzle and print that puzzle's results
        for results in self._puzzle_results:
            # Print the header line
            print >>string_io, "Original Filename: %s" % results.original_filename
            print >>string_io, "Puzzle Identification Number: " + str(results.original_id_number) + "\n"

            # Print the standard accuracy information
            for accuracy_type in [ResultAccuracyMetric.StandardDirectAccuracy, ResultAccuracyMetric.ModifiedDirectAccuracy]:

                # Select the type of direct accuracy to print >>string_io, .
                if accuracy_type == ResultAccuracyMetric.StandardDirectAccuracy:
                    acc_name = "\tStandard"
                    direct_acc = results.standard_direct_accuracy
                elif accuracy_type == ResultAccuracyMetric.ModifiedDirectAccuracy:
                    acc_name = "\tModified"
                    direct_acc = results.modified_direct_accuracy

                # Print the selected direct accuracy type
                numb_pieces_in_original_puzzle = results.numb_pieces
                piece_count_weight = direct_acc.numb_different_puzzle + numb_pieces_in_original_puzzle
                print >>string_io, "\tSolved Puzzle ID #%d" % direct_acc.solved_puzzle_id
                print >>string_io, acc_name + " Direct Accuracy:\t\t%d/%d\t(%3.2f%%)" % (direct_acc.numb_correct_placements,
                                                                                         piece_count_weight,
                                                                                         100.0 * direct_acc.numb_correct_placements / piece_count_weight)
                print >>string_io, acc_name + " Numb from Diff Puzzle:\t%d/%d\t(%3.2f%%)" % (direct_acc.numb_different_puzzle,
                                                                                             piece_count_weight,
                                                                                             100.0 * direct_acc.numb_different_puzzle / piece_count_weight)
                print >>string_io, acc_name + " Numb Wrong Location:\t%d/%d\t(%3.2f%%)" % (direct_acc.numb_wrong_location,
                                                                                           piece_count_weight,
                                                                                           100.0 * direct_acc.numb_wrong_location / piece_count_weight)
                print >>string_io, acc_name + " Numb Wrong Rotation:\t%d/%d\t(%3.2f%%)" % (direct_acc.numb_wrong_rotation,
                                                                                           piece_count_weight,
                                                                                           100.0 * direct_acc.numb_wrong_rotation / piece_count_weight)
                # Calculate the number of missing pieces
                numb_pieces_missing = (numb_pieces_in_original_puzzle
                                       - direct_acc.numb_pieces_from_original_puzzle_in_solved_puzzle)
                print >>string_io, acc_name + " Numb Pieces Missing:\t%d/%d\t(%3.2f%%)" % (numb_pieces_missing,
                                                                                           numb_pieces_in_original_puzzle,
                                                                                           100.0 * numb_pieces_missing / numb_pieces_in_original_puzzle)
                # Print a new line to separate the results
                print >>string_io, ""

            # Print the modified neighbor accuracy
            numb_pieces_in_original_puzzle = results.numb_pieces
            neighbor_acc = results.modified_neighbor_accuracy
            neighbor_count_weight = neighbor_acc.numb_pieces_in_original_puzzle + neighbor_acc.wrong_puzzle_id
            neighbor_count_weight *= PuzzlePieceSide.get_numb_sides()
            print >>string_io, "\tSolved Puzzle ID #%d" % neighbor_acc.solved_puzzle_id
            print >>string_io, "\tNeighbor Accuracy:\t\t%d/%d\t(%3.2f%%)" % (neighbor_acc.correct_neighbor_count,
                                                                             neighbor_count_weight,
                                                                             100.0 * neighbor_acc.correct_neighbor_count / neighbor_count_weight)
            numb_missing_pieces = (numb_pieces_in_original_puzzle
                                   - neighbor_acc.numb_pieces_from_original_puzzle_in_solved_puzzle)
            print >>string_io, "\tNumb Missing Pieces:\t%d/%d\t(%3.2f%%)" % (numb_missing_pieces,
                                                                             results.numb_pieces,
                                                                             100.0 * numb_missing_pieces / results.numb_pieces)
            numb_from_wrong_puzzle = neighbor_acc.wrong_puzzle_id
            numb_pieces_in_puzzle = neighbor_acc.total_numb_pieces_in_solved_puzzle
            print >>string_io, "\tNumb from Diff Puzzle:\t%d/%d\t(%3.2f%%)" % (numb_from_wrong_puzzle,
                                                                               numb_pieces_in_puzzle,
                                                                               100.0 * numb_from_wrong_puzzle / numb_pieces_in_puzzle)
            # Print a new line to separate the results
            print >>string_io, ""

        # Log the result
        logging.info(string_io.getvalue())
        string_io.close()


class PuzzleResultsInformation(object):
    """
    Encapsulates all of the accuracy results information for a puzzle.
    """

    _PERFORM_ASSERT_CHECKS = config.PERFORM_ASSERT_CHECKS

    def __init__(self, puzzle_id, original_img_filename):

        # Store the number of pieces and the puzzle id
        self.original_id_number = puzzle_id
        self._numb_pieces = 0
        self._original_img_filename = original_img_filename

        # Define the attributes for the standard accuracy.
        self.standard_direct_accuracy = None
        self.modified_direct_accuracy = None
        self.modified_neighbor_accuracy = None

    @property
    def numb_pieces(self):
        """
        Gets the piece count for an implicit puzzle id

        Returns (int):
            Number of pieces in original puzzle with this puzzle id

        """
        return self._numb_pieces

    @numb_pieces.setter
    def numb_pieces(self, value):
        self._numb_pieces = value

    def get_direct_accuracy_solved_puzzle_piece_count(self, accuracy_metric):
        """
        Get the number of pieces in the direct accuracy selected puzzle.

        Args:
            accuracy_metric (ResultAccuracyMetric): Direct accuracy metric (e.g., Standard, Modified) being used

        Returns (int): Number of pieces in the solved puzzle
        """
        if accuracy_metric == ResultAccuracyMetric.ModifiedDirectAccuracy:
            direct_acc = self.modified_direct_accuracy
        elif accuracy_metric == ResultAccuracyMetric.StandardDirectAccuracy:
            direct_acc = self.standard_direct_accuracy
        else:
            raise ValueError("Invalid accuracy metric.")

        # Calculate the direct accuracy
        numb_pieces_in_original_puzzle = self.numb_pieces
        return direct_acc.numb_different_puzzle + numb_pieces_in_original_puzzle

    def get_direct_accuracy_score(self, accuracy_metric, result_type):
        """
        Gets either the modified or direct accuracy score for a solved solution.

        Args:
            accuracy_metric (ResultAccuracyMetric): Direct accuracy metric (e.g., Standard, Modified) being used
            result_type (DirectAccuracyResultType): Specific direct accuracy metric being used

        Returns (float): Accuracy score bounded between 0.0 and 1.0
        """
        if accuracy_metric == ResultAccuracyMetric.ModifiedDirectAccuracy:
            direct_acc = self.modified_direct_accuracy
        elif accuracy_metric == ResultAccuracyMetric.StandardDirectAccuracy:
            direct_acc = self.standard_direct_accuracy
        else:
            raise ValueError("Invalid accuracy metric.")

        # Calculate the direct accuracy
        numb_pieces_in_original_puzzle = self.numb_pieces
        piece_count_weight = direct_acc.numb_different_puzzle + numb_pieces_in_original_puzzle

        # Determine the accuracy type to access
        if result_type == DirectAccuracyResultType.AccuracyScore:
            count = direct_acc.numb_correct_placements
        elif result_type == DirectAccuracyResultType.DifferentPuzzleScore:
            count = direct_acc.numb_different_puzzle
        elif result_type == DirectAccuracyResultType.WrongLocationScore:
            count = direct_acc.numb_wrong_location
        elif result_type == DirectAccuracyResultType.WrongRotationScore:
            count = direct_acc.numb_wrong_rotation
        elif result_type == DirectAccuracyResultType.MissingPieceScore:
            numb_pieces_in_original_puzzle = direct_acc.numb_pieces_in_original_puzzle
            return numb_pieces_in_original_puzzle - direct_acc.numb_pieces_from_original_puzzle_in_solved_puzzle
        else:
            raise ValueError("Invalid accuracy result type.")

        return 1.0 * count / piece_count_weight

    def get_neighbor_accuracy_score(self):
        """
        Gets the neighbor accuracy score for the puzzle

        Returns (float): Accuracy score
        """
        neighbor_acc = self.modified_neighbor_accuracy
        neighbor_count_weight = neighbor_acc.numb_pieces_in_original_puzzle + neighbor_acc.wrong_puzzle_id
        neighbor_count_weight *= PuzzlePieceSide.get_numb_sides()
        return 1.0 * neighbor_acc.correct_neighbor_count / neighbor_count_weight

    def get_neighbor_error_piece_count(self, result_type):
        """
        Gets a specified type of neighbor inaccuracy.

        Args:
            result_type (NeighborAccuracyErrorType): Error type for the neighbor accuracy to consider

        Returns (int): Number of pieces that fall into the error category
        """
        numb_pieces_in_original_puzzle = self.numb_pieces
        neighbor_acc = self.modified_neighbor_accuracy

        if result_type == NeighborAccuracyErrorType.MissingPieceCount:
            return numb_pieces_in_original_puzzle - neighbor_acc.numb_pieces_from_original_puzzle_in_solved_puzzle
        elif result_type == NeighborAccuracyErrorType.DifferentPuzzleCount:
            return neighbor_acc.wrong_puzzle_id
        else:
            raise ValueError("Invalid neighbor accuracy error type.")

    def get_accuracy_solved_puzzle_id(self, accuracy_type):
        """
        Gets either the modified or direct accuracy score for a solved solution.

        Args:
            accuracy_type (ResultAccuracyMetric): Type of accuracy being accessed.

        Returns (float): Accuracy score bounded between 0.0 and 1.0
        """
        if accuracy_type == ResultAccuracyMetric.ModifiedDirectAccuracy:
            acc_info = self.modified_direct_accuracy
        elif accuracy_type == ResultAccuracyMetric.StandardDirectAccuracy:
            acc_info = self.standard_direct_accuracy
        elif accuracy_type == ResultAccuracyMetric.ModifiedNeighborAccuracy:
            acc_info = self.modified_neighbor_accuracy
        else:
            raise ValueError("Invalid Accuracy Type.")
        return acc_info.solved_puzzle_id

    def resolve_neighbor_accuracies(self, solved_puzzle):
        """
        Neighbor Accuracy Resolved

        This function is used to calculate the neighbor accuracy of a solved puzzle and compare it to
        the results of previously calculated accuracies to select the best one.

        Args:
            solved_puzzle (Puzzle): A solved Puzzle
        """

        # Placed piece array
        placed_piece_matrix, rotation_matrix = solved_puzzle.build_placed_piece_info()

        # Create a temporary neighbor accuracy info
        neighbor_accuracy_info = ModifiedNeighborAccuracy(self.original_id_number, solved_puzzle.id_number, self.numb_pieces)

        # Iterate through the set of pieces
        for piece in solved_puzzle.pieces:

            original_neighbor_id_and_sides = piece.original_neighbor_id_numbers_and_sides
            # Sort the sides of the neighbor location to match the original order.
            neighbor_location_and_sides = piece.get_neighbor_locations_and_sides()
            neighbor_location_and_sides = sorted(neighbor_location_and_sides, key=lambda tup: tup[1].value)

            # Perform a check of the piece location information
            if PuzzleResultsInformation._PERFORM_ASSERT_CHECKS:
                assert len(neighbor_location_and_sides) == len(original_neighbor_id_and_sides)
                # Verify the side in each element is the same
                for i in xrange(0, len(neighbor_location_and_sides)):
                    assert neighbor_location_and_sides[i][1] == original_neighbor_id_and_sides[i][1]

            # Iterate through all sides and check the
            for side_numb in xrange(0, len(neighbor_location_and_sides)):

                side = PuzzlePieceSide(side_numb)

                # Verify the puzzle identification numbers match.  If not, mark all as wrong then go to next piece
                if piece.original_puzzle_id != self.original_id_number:
                    # neighbor_accuracy_info.wrong_puzzle_id += 1
                    neighbor_accuracy_info.add_wrong_puzzle_id(piece.id_number, side)
                    continue

                # Extract the placed piece ID.  If a cell is empty or does not exist, mark it as None
                neighbor_loc = neighbor_location_and_sides[side_numb][0]
                # Check the location is invalid.  If it is, mark the location as None.
                if (neighbor_loc[0] < 0 or neighbor_loc[1] < 0
                        or neighbor_loc[0] >= solved_puzzle.grid_size[0]
                        or neighbor_loc[1] >= solved_puzzle.grid_size[1]):

                    placed_piece_id = None
                # Handle dimensions that are off the edge of the board
                else:
                    # Get the placed piece number
                    placed_piece_id = placed_piece_matrix[neighbor_loc]
                    # Handle missing pieces
                    placed_piece_id = placed_piece_id if placed_piece_id >= 0 else None

                # Check if the neighbor if the neighbor
                if(placed_piece_id == original_neighbor_id_and_sides[side_numb][0]
                   and (original_neighbor_id_and_sides[side_numb][0] is None or
                        piece.rotation.value == rotation_matrix[neighbor_location_and_sides[side_numb][0]])):

                    # neighbor_accuracy_info.correct_neighbor_count += 1
                    neighbor_accuracy_info.add_correct_neighbor(piece.id_number, side)

                # Mark neighbor as incorrect
                else:
                    # neighbor_accuracy_info.wrong_neighbor_count += 1
                    neighbor_accuracy_info.add_wrong_neighbor(piece.id_number, side)

        # Update the best accuracy should be updated.
        if ModifiedNeighborAccuracy.check_if_update_neighbor_accuracy(self.modified_neighbor_accuracy,
                                                                      neighbor_accuracy_info):
            self.modified_neighbor_accuracy = neighbor_accuracy_info

    def resolve_direct_accuracies(self, puzzle):
        """
        Direct Accuracy Resolver

        In the case of multiple puzzles, there will be multiple direct accuracies for each

        Args:
            puzzle (Puzzle): A solved puzzle
        """

        # Get the standard direct accuracy
        new_direct_accuracy = puzzle.determine_standard_direct_accuracy(self.original_id_number, self.numb_pieces)

        # Update the stored standard direct accuracy if applicable
        if DirectAccuracyPuzzleResults.check_if_update_direct_accuracy(self.standard_direct_accuracy,
                                                                       new_direct_accuracy):
            self.standard_direct_accuracy = new_direct_accuracy

        # Modified direct accuracy is a more complicated procedure so it runs separately
        self._resolve_modified_direct_accuracy(puzzle)

    def _resolve_modified_direct_accuracy(self, puzzle):
        """
        Modified Direct Accuracy Resolver

        Specially resolves the modified direct accuracy.  This is required as as the upper left coordinate(s) need
        to be found.

        Simplified Algorithm Flow:
        * Find the piece(s) that have the minimum MANHATTAN distance from the upper left corner of the puzzle (i.e.
        location (0, 0)

        * Use any possible piece location that is within that manhattan distance as a reference point for calculating
        the direct accuracy.

        Args:
            puzzle (Puzzle): A puzzle object whose direct accuracy is being determined.

        """

        # Placed piece array
        placed_piece_matrix, _ = puzzle.build_placed_piece_info()

        # Do a Breadth first search to define the possible candidates for modified direct method
        frontier_set = [(0, 0)]
        explored_set = []
        found_dist = None
        # Continue searching until either
        while found_dist is None or (frontier_set and frontier_set[0][0] + frontier_set[0][1] <= found_dist):
            # Pop an element from the front of the list and add it to the explored set
            next_loc = frontier_set.pop(0)
            explored_set.append(next_loc)

            # Check if there is a piece in the current location
            if found_dist is None and placed_piece_matrix[next_loc] != -1:
                found_dist = next_loc[0] + next_loc[1]  # Use the manhattan distance
            else:
                # Move one piece down
                down_loc = (next_loc[0] + 1, next_loc[1])
                # Only need to check first dimension since only one that changed
                if down_loc[0] < puzzle.grid_size[0] and down_loc not in explored_set and down_loc not in frontier_set:
                    frontier_set.append(down_loc)

                # Move one piece to the right
                right_loc = (next_loc[0], next_loc[1] + 1)
                # Only need to check second dimension since only one that changed
                if right_loc[1] < puzzle.grid_size[1] and right_loc not in explored_set and right_loc not in frontier_set:
                    frontier_set.append(right_loc)

        # For all upper left coordinate candidates, determine the modified direct accuracy.
        for possible_upper_left in explored_set:
            modified_direct_accuracy = puzzle.determine_modified_direct_accuracy(self.original_id_number,
                                                                                 possible_upper_left,
                                                                                 self.numb_pieces)
            # Update the standard direct accuracy
            if DirectAccuracyPuzzleResults.check_if_update_direct_accuracy(self.modified_direct_accuracy,
                                                                           modified_direct_accuracy):
                self.modified_direct_accuracy = modified_direct_accuracy

    @property
    def original_filename(self):
        """
        Property for accessing an image's filename.

        Returns (str):
            Original filename of the input image

        """
        return self._original_img_filename


class PieceDirectAccuracyResult(Enum):
    """
    Enumerated type used to represent the direct accuracy results for a single individual piece.  Each enumerated
    value is a tuple representing the color of the puzzle piece in BGR format.
    """
    different_puzzle = (255, 0, 0)  # Blue
    correct_placement = (0, 204, 0)  # Green
    wrong_location = (0, 0, 255)  # Red
    wrong_rotation = (51, 153, 255)  # Orange


class DirectAccuracyPuzzleResults(object):
    """
    Structure used for managing puzzle placement results.
    """

    def __init__(self, original_puzzle_id, solved_puzzle_id, numb_pieces_in_original_puzzle):
        self._orig_puzzle_id = original_puzzle_id
        self._solved_puzzle_id = solved_puzzle_id
        self._different_puzzle = {}
        self.numb_pieces_in_original_puzzle = numb_pieces_in_original_puzzle
        self._wrong_location = {}
        self._wrong_rotation = {}
        self._correct_placement = {}

    def get_piece_result(self, piece_id):
        """
        Gets the direct accuracy results (e.g. wrong puzzle, wrong id, wrong rotation, etc.) for an individual
        piece.

        Args:
            piece_id (int): Identification number of the piece

        Returns (PieceDirectAccuracyResult):
            Direct accuracy result for an individual piece
        """
        key = str(piece_id)
        if key in self._correct_placement:
            return PieceDirectAccuracyResult.correct_placement

        if key in self._wrong_rotation:
            return PieceDirectAccuracyResult.wrong_rotation

        if key in self._wrong_location:
            return PieceDirectAccuracyResult.wrong_location

        if key in self._different_puzzle:
            return PieceDirectAccuracyResult.different_puzzle

        # Piece does not exist in the results so raise an error
        raise ValueError("Piece id: \"%d\" does not exist in this result set." % piece_id)

    @property
    def original_puzzle_id(self):
        """
        Direct Accuracy Original Puzzle ID Number Accessor

        Returns (int):
            The puzzle ID associated with the ORIGINAL set of puzzle results
        """
        return self._orig_puzzle_id

    @property
    def solved_puzzle_id(self):
        """
        Direct Accuracy Solved Puzzle ID Number Accessor

        Returns (int):
            The puzzle ID associated with the SOLVED set of puzzle pieces
        """
        return self._solved_puzzle_id

    def add_wrong_location(self, piece):
        """
        Wrong Piece Location Tracker

        Adds a piece that was assigned to the wrong location number to the tracker.

        Args:
            piece (PuzzlePiece): Piece placed in the wrong LOCATION
        """
        self._wrong_location[str(piece.id_number)] = piece

    def add_different_puzzle(self, piece):
        """
        Wrong Puzzle ID Tracker

        Adds a piece that was assigned to a DIFFERENT PUZZLE ID number to the tracker.

        Args:
            piece (PuzzlePiece): Puzzle Piece that was placed with the different PUZZLE IDENTIFICATION NUMBER
        """
        self._different_puzzle[str(piece.id_number)] = piece

    def add_wrong_rotation(self, piece):
        """
        Wrong Piece Rotation Tracker

        Adds a piece that had the wrong rotation

        Args:
            piece (PuzzlePiece): Puzzle Piece that was placed with the wrong Rotation (i.e. not 0 degrees)
        """
        self._wrong_rotation[str(piece.id_number)] = piece

    def add_correct_placement(self, piece):
        """
        Correctly Placed Piece Tracker

        Adds a piece that has been placed correctly

        Args:
            piece (PuzzlePiece): Puzzle Piece that was placed CORRECTLY.
        """
        self._correct_placement[str(piece.id_number)] = piece

    @property
    def weighted_accuracy(self):
        """
        Calculates and returns the weighted accuracy score for this puzzle.

        Returns (float):
            Weighted accuracy score for this puzzle solution

        """
        return 1.0 * self.numb_correct_placements / (self.numb_pieces_in_original_puzzle
                                                     + self.numb_different_puzzle)

    @property
    def numb_correct_placements(self):
        """
        Number of Pieces Placed Correctly Property

        Gets the number of pieces placed correctly.

        Returns (int):
            Number of pieces correctly placed with the right puzzle ID, location, and rotation.
        """
        return len(self._correct_placement)

    @property
    def numb_wrong_location(self):
        """
        Number of Pieces Placed in the Wrong Location

        Gets the number of pieces placed in the wrong LOCATION.

        Returns (int):
            Number of pieces placed in the wrong location.
        """
        return len(self._wrong_location)

    @property
    def numb_wrong_rotation(self):
        """
        Number of Pieces with the Wrong Rotation

        Gets the number of pieces placed with the wrong ROTATION.

        Returns (int):
            Number of pieces placed with the incorrect rotation (i.e. not 0 degrees)
        """
        return len(self._wrong_rotation)

    @property
    def numb_different_puzzle(self):
        """
        Number of Pieces in the Wrong Puzzle

        Gets the number of pieces placed in entirely the wrong puzzle.

        Returns (int):
            Number of pieces placed in the wrong puzzle
        """
        return len(self._different_puzzle)

    @property
    def total_numb_pieces_in_solved_puzzle(self):
        """
        Total Number of Pieces in the solved image

        Returns (int):
            Total number of pieces (both with expected puzzle id and wrong puzzle id(s))

        """
        return self.numb_pieces_from_original_puzzle_in_solved_puzzle + self.numb_different_puzzle

    @property
    def numb_pieces_from_original_puzzle_in_solved_puzzle(self):
        """
        Number of pieces from the original puzzle in the solved result

        Returns (int):
            Only the number of included pieces

        """
        return self.numb_correct_placements + self.numb_wrong_location + self.numb_wrong_rotation

    @staticmethod
    def check_if_update_direct_accuracy(current_best_direct_accuracy, new_direct_accuracy):
        """
        Determines whether the current best direct accuracy should be replaced with a newly calculated direct
        accuracy.

        Args:
            current_best_direct_accuracy (DirectAccuracyPuzzleResults): The best direct accuracy results
            found so far.

            new_direct_accuracy (DirectAccuracyPuzzleResults): The newly calculated best direct accuracy

        Returns (bool):
            True if the current best direct accuracy should be replaced with the new direct accuracy and False
            otherwise.
        """

        # If no best direct accuracy found so far, then just return True
        if current_best_direct_accuracy is None:
            return True

        # Get the information on the current best result
        best_numb_included_pieces = current_best_direct_accuracy.numb_pieces_from_original_puzzle_in_solved_puzzle
        best_accuracy = current_best_direct_accuracy.weighted_accuracy

        # Get the information on the new direct accuracy result
        new_numb_included_pieces = new_direct_accuracy.numb_pieces_from_original_puzzle_in_solved_puzzle
        new_accuracy = new_direct_accuracy.weighted_accuracy

        # Update the standard direct accuracy if applicable
        if (best_accuracy < new_accuracy or
                (best_accuracy == new_accuracy and best_numb_included_pieces < new_numb_included_pieces)):
            return True
        else:
            return False


class BestBuddyResultsCollection(object):
    """
    Stores the best buddy result accuracies in a single object.
    """

    _PERFORM_ASSERT_CHECKS = True

    def __init__(self):
        self._best_buddy_accuracy = []

    def create_best_buddy_accuracy_for_new_puzzle(self, puzzle_id):
        """
        Creates a best buddy accuracy for a new puzzle.

        Args:
            puzzle_id (int): Creates a new best buddy accuracy for a puzzle

        """
        new_bb_acc = BestBuddyAccuracy(puzzle_id)
        self._best_buddy_accuracy.append(new_bb_acc)

    @property
    def best_buddy_accuracy(self):
        """
        Property to get access to all the best buddy information.

        Returns (List[BestBuddyAccuracy]):
            The best buddy accuracy objects for all puzzles.

        """
        return self._best_buddy_accuracy

    def __iter__(self):
        """
        Allows for the iteration over a set of best buddy objects.

        Returns(List[BestBuddyAccuracy]):
            The best buddy accuracy objects for all puzzles.

        """
        for i in xrange(0, len(self._best_buddy_accuracy)):
            yield self._best_buddy_accuracy[i]

    def __getitem__(self, key):
        """
        Allows for the indexing of a Best buddy accuracy class

        Args:
            key (int): Puzzle id

        Returns(List[BestBuddyAccuracy]):
            The best buddy accuracy objects for all puzzles.

        """
        return self._best_buddy_accuracy[key]

    def total_best_buddy_count(self):
        """
        Gets the total number of best buddies represented by this collection.

        Returns (int):
            Total number of best buddies in the collection

        """
        bb_count = 0
        for best_buddy_acc in self._best_buddy_accuracy:
            bb_count += best_buddy_acc.numb_open_best_buddies
            bb_count += best_buddy_acc.numb_wrong_best_buddies
            bb_count += best_buddy_acc.numb_correct_best_buddies
            bb_count += best_buddy_acc.numb_excluded_best_buddies
        return bb_count

    def print_results(self):
        """
        Prints the best buddy accuracy information to the log.
        """
        # Use a cStringIO as essentially a string buffer
        string_io = cStringIO.StringIO()
        print >>string_io, "Best Buddy Accuracy Information:"
        for bb_acc in self._best_buddy_accuracy:
            print >>string_io, str(bb_acc) + "\n"
        # Write the to log.
        logging.debug(string_io.getvalue())
        string_io.close()

    def output_results_images(self, puzzle_solver_type, all_input_image_filenames, solved_puzzles, puzzle_type,
                              timestamp, orig_img_filename=None, output_filenames=None):
        """
        Converts the results information to a data visualization to see where there are right and wrong
        best buddies.

        Args:
            puzzle_solver_type (PuzzleSolver): Type of solver being run
            all_input_image_filenames (List[str]): List of input image files and paths.
            solved_puzzles (List[Puzzle]): List of puzzles.
            puzzle_type (PuzzleType): Type of the solved puzzle.
            timestamp (float): Timestamp as a floating point number.  Converted to a string by this function.
            orig_img_filename (Optional str): Filename of the original image
            output_filenames (List[str]): Name of the output image files
        """

        # Iterate through each possible and print its BB accuracy information.
        for puzzle_id in xrange(0, len(self._best_buddy_accuracy)):

            # Get the individual results and puzzle
            puzzle = solved_puzzles[puzzle_id]
            bb_acc = self._best_buddy_accuracy[puzzle_id]

            # Ensure that the puzzle ids match
            if BestBuddyResultsCollection._PERFORM_ASSERT_CHECKS:
                assert bb_acc.puzzle_id == puzzle.id_number

            # Iterate each puzzle piece and set its color for each side
            for piece in puzzle.pieces:
                piece.reset_image_coloring_for_polygons()
                for side in PuzzlePieceSide.get_all_sides():
                    coloring = bb_acc.get_piece_side_result(piece.id_number, side)
                    piece.results_image_polygon_coloring(side, coloring)

            # Determine whether the puzzle id should be used in the filename
            filename_puzzle_id = puzzle_id if orig_img_filename is None else None
            descriptor = "best_buddy_acc"
            if output_filenames is None:
                output_filename = Puzzle.make_image_filename(puzzle_solver_type, all_input_image_filenames, descriptor,
                                                             Puzzle.OUTPUT_IMAGE_DIRECTORY, puzzle_type,
                                                             timestamp, orig_img_filename=orig_img_filename,
                                                             puzzle_id=filename_puzzle_id)
            else:
                output_filename = output_filenames[puzzle_id]
            # Stores the results to a file.
            puzzle.build_puzzle_image(use_results_coloring=True)
            puzzle.save_to_file(output_filename)


class PieceSideBestBuddyAccuracyResult(Enum):
    """
    Enumerated type used to represent the direct accuracy results for a single individual piece.  Each enumerated
    value is a tuple representing the color of the puzzle piece in BGR format.
    """
    wrong_best_buddy = (0, 0, 255)  # Red
    correct_best_buddy = (0, 204, 0)  # Green
    no_best_buddy = (255, 255, 255)  # White
    open_best_buddy = (255, 0, 0)  # Blue


class BestBuddyAccuracy(object):
    """
    Store the best buddy accuracy information for a single puzzle
    """

    _PERFORM_ASSERT_CHECK = True

    def __init__(self, puzzle_id):
        self.puzzle_id = puzzle_id
        self._open_best_buddies = {}
        self._wrong_best_buddies = {}
        self._correct_best_buddies = {}
        self._excluded_best_buddies = {}

    def add_excluded_best_buddy(self, piece_id, side):
        """
        Adds an excluded best buddies pairing.

        Args:
            piece_id (int): Piece identification number of the as of yet unpaired best buddy
            side (PuzzlePieceSide): Side of the unpaired best buddy
        """
        BestBuddyAccuracy.add_piece_side_tuple_to_dict(self._excluded_best_buddies, piece_id, side)

    def add_open_best_buddy(self, piece_id, side):
        """
        Deletes an unpaired open best buddy from the list

        Args:
            piece_id (int): Piece identification number of the as of yet unpaired best buddy
            side (PuzzlePieceSide): Side of the unpaired best buddy
        """
        BestBuddyAccuracy.add_piece_side_tuple_to_dict(self._open_best_buddies, piece_id, side)

    def delete_open_best_buddy(self, piece_id, side):
        """
        Adds an unpaired open best buddy to the list

        Args:
            piece_id (int): Piece identification number of the as of yet unpaired best buddy
            side (PuzzlePieceSide): Side of the unpaired best buddy
        """
        if self.exists_open_best_buddy(piece_id, side):
            key = BestBuddyAccuracy.piece_side_tuple_key(piece_id, side)
            del self._open_best_buddies[key]

    def exists_open_best_buddy(self, piece_id, side):
        """
        Checks if a pairing of a piece identification number and side exists in the pool of open best buddies.

        Args:
            piece_id (int): Piece identification number
            side (PuzzlePieceSide): Possible neighbor side

        Returns (bool):
            True if the key is in the dictionary and False otherwise.

        """
        return BestBuddyAccuracy.check_if_piece_side_tuple_in_dict(self._open_best_buddies, piece_id, side)

    def exists_wrong_best_buddy(self, piece_id, side):
        """
        Checks if a pairing of a piece identification number and side exists in the pool of WRONG best buddies.

        Args:
            piece_id (int): Piece identification number
            side (PuzzlePieceSide): Possible neighbor side

        Returns (bool):
            True if the key is in the dictionary and False otherwise.

        """
        return BestBuddyAccuracy.check_if_piece_side_tuple_in_dict(self._wrong_best_buddies, piece_id, side)

    def add_wrong_best_buddy(self, piece_id, side):
        """
        Adds a new piece and side to the list of wrong best buddies.

            Args:
                piece_id (int): Identification number of the piece with the wrong best buddy
                side (PuzzlePieceSide): Side of the piece with the wrong best buddy
        """
        BestBuddyAccuracy.add_piece_side_tuple_to_dict(self._wrong_best_buddies, piece_id, side)

    def exists_correct_best_buddy(self, piece_id, side):
        """
        Checks if a pairing of a piece identification number and side exists in the pool of CORRECT best buddies.

        Args:
            piece_id (int): Piece identification number
            side (PuzzlePieceSide): Possible neighbor side

        Returns (bool):
            True if the key is in the dictionary and False otherwise.

        """
        return BestBuddyAccuracy.check_if_piece_side_tuple_in_dict(self._correct_best_buddies, piece_id, side)

    def add_correct_best_buddy(self, piece_id, side):
        """
        Side of the piece where the BB is being referred.

        Args:
            piece_id (int): Identification number ofr the piece of interest
            side (PuzzlePieceSide): Puzzle piece side of reference for the correct best buddy
        """
        BestBuddyAccuracy.add_piece_side_tuple_to_dict(self._correct_best_buddies, piece_id, side)

    def get_piece_side_result(self, piece_id, side):
        """
        Gets the best buddy result for the combination of puzzle piece and side.

        Args:
            piece_id (int): Identification number of the piece
            side (PuzzlePieceSide): Side of the puzzle piece of puzzle piece of interest

        Returns (PieceSideBestBuddyAccuracyResult):
            Best buddy accuracy result
        """
        if self.exists_wrong_best_buddy(piece_id, side):
            return PieceSideBestBuddyAccuracyResult.wrong_best_buddy

        if self.exists_correct_best_buddy(piece_id, side):
            return PieceSideBestBuddyAccuracyResult.correct_best_buddy

        if self.exists_open_best_buddy(piece_id, side):
            return PieceSideBestBuddyAccuracyResult.open_best_buddy

        # Piece does not have a best buddy
        return PieceSideBestBuddyAccuracyResult.no_best_buddy

    @staticmethod
    def add_piece_side_tuple_to_dict(bb_dict, piece_id, side):
        """
        Adds a best buddy information to the specified best buddy dictionary.

        Args:
            bb_dict (dict): Dictionary containing best buddy information
            piece_id (int): Identification number of the piece
            side (PuzzlePieceSide): Side of the piece that is referred to for best buddy.
        """
        key = BestBuddyAccuracy.piece_side_tuple_key(piece_id, side)
        bb_dict[key] = (piece_id, side)

    @staticmethod
    def check_if_piece_side_tuple_in_dict(bb_dict, piece_id, side):
        """
        Checks whether the piece and side exists in the specified best buddy dictionary.

        Args:
            bb_dict (dict): Best buddy information dictionary
            piece_id (int): Identification number for the piece
            side (PuzzlePieceSide): Side of the piece where the BB is being referred.

        Returns (bool):
            True if the pairing of piece_id and side exists in the BB dictionary.
        """
        key = BestBuddyAccuracy.piece_side_tuple_key(piece_id, side)
        return key in bb_dict

    @staticmethod
    def piece_side_tuple_key(piece_id, side):
        """
        Creates a unique dictionary key for storing the piece side tuple.

        Args:
            piece_id (int):
            side (PuzzlePieceSide):

        Returns (str):
            Dictionary key in the form "<piece_id>_<side_int_value>"
        """
        return str(piece_id) + "_" + str(side.value)

    @property
    def numb_excluded_best_buddies(self):
        """
        Property to get the total number of best buddies where one of the pieces has been excluded.

        Returns (int):
            Total number of best buddies whose best buddies have not yet been placed.

        """
        return len(self._excluded_best_buddies)

    @property
    def numb_open_best_buddies(self):
        """
        Property to get the total number of best buddies where one of the pair has not been placed.

        Returns (int):
            Total number of best buddies whose best buddies have not yet been placed.

        """
        return len(self._open_best_buddies)

    @property
    def numb_correct_best_buddies(self):
        """
        Gets the number of correct best buddies who are next to their best buddy

        Returns (int):
            Total number of correct best buddies

        """
        return len(self._correct_best_buddies)

    @property
    def numb_wrong_best_buddies(self):
        """
        Gets the number of correct best buddies who are NOT next to their best buddy

        Returns (int):
            Total number of WRONG best buddies

        """
        return len(self._wrong_best_buddies)

    def __unicode__(self):
        """
        Constructs the best buddy accuracy information as a string

        Returns (string):
            Best Buddy accuracy as a string
        """
        return "Best Buddy Info for Solved Puzzle #%s\n" % self.puzzle_id \
               + "\tNumb Open Best Buddies:\t\t%s\n" % self.numb_open_best_buddies \
               + "\tNumb Correct Best Buddies:\t%s\n" % self.numb_correct_best_buddies \
               + "\tNumb Wrong Best Buddies:\t%s" % self.numb_wrong_best_buddies

    def __str__(self):
        return unicode(self).encode('utf-8')


class PieceSideNeighborAccuracyResult(Enum):
    """
    Defines the color for tuples of piece ids and sides according to the neighbor accuracy metric.
    """
    correct_neighbor = (0, 204, 0)  # Green
    wrong_neighbor = (0, 0, 255)  # Red
    different_puzzle_id = (255, 0, 0)  # Blue


class ModifiedNeighborAccuracy(object):
    """
    Encapsulating structure for the modified neighbor based accuracy approach.
    """

    def __init__(self, original_puzzle_id, solved_puzzle_id, number_of_pieces):
        self._original_puzzle_id = original_puzzle_id
        self._solved_puzzle_id = solved_puzzle_id
        self._actual_number_of_pieces = number_of_pieces

        self._wrong_puzzle_id = {}
        self._correct_neighbors = {}
        self._wrong_neighbors = {}

    def get_piece_side_result(self, piece_id, side):
        """
        Gets the best buddy result for the combination of puzzle piece and side.

        Args:
            piece_id (int): Identification number of the piece
            side (PuzzlePieceSide): Side of the puzzle piece of puzzle piece of interest

        Returns (PieceSideBestBuddyAccuracyResult):
            Best buddy accuracy result
        """
        if self.exists_wrong_puzzle_id(piece_id, side):
            return PieceSideNeighborAccuracyResult.different_puzzle_id

        if self.exists_correct_neighbor(piece_id, side):
            return PieceSideNeighborAccuracyResult.correct_neighbor

        if self.exists_wrong_neighbor(piece_id, side):
            return PieceSideNeighborAccuracyResult.wrong_neighbor

        # Piece does not have a best buddy
        raise ValueError("Pairing of piece id \"%s\" and side \"%s\" does not exist in this puzzle" % (piece_id,
                                                                                                       side.side_name))

    def add_wrong_puzzle_id(self, piece_id, side):
        """
        Stores information on piece assigned to the wrong puzzle ID.

        Args:
            piece_id (int): Identification number of the wrong puzzle piece
            side (PuzzlePieceSide): Side of the puzzle piece with the wrong neighbor

        """
        BestBuddyAccuracy.add_piece_side_tuple_to_dict(self._wrong_puzzle_id, piece_id, side)

    def exists_wrong_puzzle_id(self, piece_id, side):
        """
        Determines whether the pairing of the piece id and the side are in the WRONG PUZZLE ID list.

        Args:
            piece_id (int): Identification number of the wrong puzzle piece
            side (PuzzlePieceSide): Side of the puzzle piece which is being checked for a WRONG PUZZLE ID

        """
        return BestBuddyAccuracy.check_if_piece_side_tuple_in_dict(self._wrong_puzzle_id, piece_id, side)

    @property
    def wrong_puzzle_id(self):
        """
        Gets the number of pieces assigned to the wrong puzzle

        Returns (int):
            Number pieces in the wrong puzzle

        """
        return len(self._wrong_puzzle_id)

    def add_correct_neighbor(self, piece_id, side):
        """
        Stores information on a CORRECT neighbor

        Args:
            piece_id (int): Identification number of the CORRECT puzzle piece
            side (PuzzlePieceSide): Side of the puzzle piece with the CORRECT neighbor

        """
        BestBuddyAccuracy.add_piece_side_tuple_to_dict(self._correct_neighbors, piece_id, side)

    def exists_correct_neighbor(self, piece_id, side):
        """
        Determines whether the pairing of the piece id and the side are in the CORRECT neighbors list.

        Args:
            piece_id (int): Identification number of the wrong puzzle piece
            side (PuzzlePieceSide): Side of the puzzle piece which is being checked for a CORRECT neighbor

        """
        return BestBuddyAccuracy.check_if_piece_side_tuple_in_dict(self._correct_neighbors, piece_id, side)

    @property
    def correct_neighbor_count(self):
        """
        Gets the CORRECT neighbor count for this puzzle.

        Returns (int):
            Number of CORRECT neighbors in the puzzle

        """
        return len(self._correct_neighbors)

    def add_wrong_neighbor(self, piece_id, side):
        """
        Stores information on a wrong neighbor

        Args:
            piece_id (int): Identification number of the wrong puzzle piece
            side (PuzzlePieceSide): Side of the puzzle piece with the wrong neighbor

        """
        BestBuddyAccuracy.add_piece_side_tuple_to_dict(self._wrong_neighbors, piece_id, side)

    def exists_wrong_neighbor(self, piece_id, side):
        """
        Determines whether the pairing of the piece id and the side are in the WRONG neighbors list.

        Args:
            piece_id (int): Identification number of the wrong puzzle piece
            side (PuzzlePieceSide): Side of the puzzle piece which is being checked for a WRONG neighbor

        """
        return BestBuddyAccuracy.check_if_piece_side_tuple_in_dict(self._wrong_neighbors, piece_id, side)

    @property
    def wrong_neighbor_count(self):
        """
        Gets the wrong neighbor count for this puzzle.

        Returns (int):
            Number of wrong neighbors in the puzzle

        """
        return len(self._wrong_neighbors)

    @property
    def original_puzzle_id(self):
        """
        Original/Input Puzzle ID Number Property

        This method is used to access puzzle identification number information of the original/input puzzle.

        Returns (int):
            Original puzzle identification number associated with this set of modified neighbor accuracy information.
        """
        return self._original_puzzle_id

    @property
    def solved_puzzle_id(self):
        """
        Solved Puzzle ID Number Property

        This method is used to access puzzle identification number information of the puzzle that was output
        by the solved.

        Returns (int):
            Solved puzzle identification number associated with this set of modified neighbor accuracy information.
        """
        return self._solved_puzzle_id

    @property
    def total_numb_pieces_in_solved_puzzle(self):
        """
        Number of Included Pieces

        This function returns the number of puzzle pieces in the solved image.

        Returns (int):
            Number of pieces in the solved
        """
        return self.numb_pieces_from_original_puzzle_in_solved_puzzle + self.wrong_puzzle_id

    @property
    def numb_pieces_in_original_puzzle(self):
        """
        Property to extract the number of pieces in the original (input) puzzle.

        Returns (int):
            Piece count in the original, input puzzle

        """
        return self._actual_number_of_pieces

    @property
    def numb_pieces_from_original_puzzle_in_solved_puzzle(self):
        """
        Number of pieces from the original puzzle in the solved result

        Returns (int):
            Only the number of included pieces

        """
        return (self.correct_neighbor_count + self.wrong_neighbor_count) / PuzzlePieceSide.get_numb_sides()

    @staticmethod
    def check_if_update_neighbor_accuracy(current_best_neighbor_accuracy, new_neighbor_accuracy):
        """
        Determines whether the current best direct accuracy should be replaced with a newly calculated direct
        accuracy.

        Args:
            current_best_neighbor_accuracy (ModifiedNeighborAccuracy): The best direct accuracy results
            found so far.

            new_neighbor_accuracy (ModifiedNeighborAccuracy): The newly calculated best direct accuracy

        Returns (bool):
            True if accuracy should be updated and False otherwise.
        """

        # If no best neighbor accuracy found so far, then just return True
        if current_best_neighbor_accuracy is None:
            return True

        # Get the information on the current best result
        best_numb_correct = current_best_neighbor_accuracy.correct_neighbor_count
        best_accuracy = current_best_neighbor_accuracy.weighted_accuracy

        # Get the information on the new direct accuracy result
        new_numb_correct = new_neighbor_accuracy.correct_neighbor_count
        new_accuracy = new_neighbor_accuracy.weighted_accuracy

        # Update the standard direct accuracy if applicable
        if (best_accuracy < new_accuracy or
                (best_accuracy == new_accuracy and best_numb_correct < new_numb_correct)):
            return True
        else:
            return False

    @property
    def weighted_accuracy(self):
        """
        Calculates and returns the weighted neighbor accuracy

        Returns (float):
            Weighted neighbor accuracy.
        """
        accuracy = 1.0 * self.correct_neighbor_count / (self._actual_number_of_pieces + self.wrong_puzzle_id)
        accuracy /= PuzzlePieceSide.get_numb_sides()
        return accuracy


class Puzzle(object):
    """
    Puzzle Object represents a single Jigsaw Puzzle.  It can import a puzzle from an image file and
    create the puzzle pieces.
    """

    print_debug_messages = True

    # Used to store the Puzzle result images.
    OUTPUT_IMAGE_DIRECTORY = "." + os.sep + "solved" + os.sep

    # DEFAULT_PIECE_WIDTH = 28  # Width of a puzzle in pixels
    DEFAULT_PIECE_WIDTH = 25  # Width of a puzzle in pixels

    # Optionally perform assertion checks.
    _PERFORM_ASSERT_CHECKS = True

    # Define the number of dimensions in the BGR space (i.e. blue, green, red)
    NUMBER_BGR_DIMENSIONS = 3

    # Define the border when generating the accuracy images.
    export_with_border = True
    border_width = 3
    border_outer_stripe_width = 1

    # Value stored when building the puzzle info when no piece is present in a given location
    MISSING_PIECE_PUZZLE_INFO_VALUE = -1

    def __init__(self, id_number, image_filename=None, piece_width=None, starting_piece_id=0):
        """Puzzle Constructor

        Constructor that will optionally load an image into the puzzle as well.

        Args:
            id_number (int): ID number for the image.  It is used for multiple image puzzles.
            image_filename (Optional str): File path of the image to load
            piece_width (Optional int): Width of a puzzle piece in pixels
            starting_piece_id (int): Identification number for the first piece in the puzzle.  If not specified,
                it default to 0.

        Returns (Puzzle):
            Puzzle divided into pieces based off the source image and the specified parameters.
        """
        # Internal Pillow Image object.
        self._id = id_number
        self._img = None
        self._img_LAB = None

        # Initialize the puzzle information.
        self._grid_size = None
        self._piece_width = piece_width if piece_width is not None else Puzzle.DEFAULT_PIECE_WIDTH
        self._img_width = None
        self._img_height = None

        # Define the upper left coordinate
        self._upper_left = (0, 0)

        # No pieces for the puzzle yet.
        self._pieces = []
        # self._id_number_to_index_map = {}

        if image_filename is None:
            self._filename = ""
            return

        # Stores the image file and then loads it.
        self._filename = image_filename
        self._load_puzzle_image()

        self._segments = []

        # Make image pieces.
        self.make_pieces(starting_piece_id)

    def _load_puzzle_image(self):
        """Puzzle Image Loader

        Loads the puzzle image file a specified filename.  Loads the specified puzzle image into memory.
        It also stores information on the puzzle dimensions (e.g. width, height) into the puzzle object.

        """

        # If the filename does not exist, then raise an error.
        if not os.path.exists(self._filename):
            raise ValueError("Invalid \"%s\" value.  File does not exist" % self._filename)

        self._img = cv2.imread(self._filename)  # Note this imports in BGR format not RGB.
        if self._img is None:
            raise IOError("Unable to load the image at the specified location \"%s\"." % self._filename)

        # Get the image dimensions.
        self._img_height, self._img_width = self._img.shape[:2]

        # Make a LAB version of the image.
        self._img_LAB = cv2.cvtColor(self._img, cv2.COLOR_BGR2LAB)

    def make_pieces(self, starting_id_numb=0):
        """
        Puzzle Generator

        Given a puzzle, this function turns the puzzle into a set of pieces.

        **Note:** When creating the pieces, some of the source image may need to be discarded
        if the image size is not evenly divisible by the number of pieces specified
        as parameters to this function.

        Args:
            starting_id_numb (Optional int): Identification number of the first piece in the puzzle.  If it is not
            specified it defaults to 0.
        """
        # Calculate the piece information.
        numb_cols = int(math.floor(self._img_width / self.piece_width))  # Floor in python returns a float
        numb_rows = int(math.floor(self._img_height / self.piece_width))  # Floor in python returns a float
        if numb_cols == 0 or numb_rows == 0:
            raise ValueError("Image size is too small for the image.  Check your setup")

        # Store the grid size.
        self._grid_size = (numb_rows, numb_cols)

        # Store the original width and height and recalculate the new width and height.
        original_width = self._img_width
        original_height = self._img_height
        self._img_width = numb_cols * self.piece_width
        self._img_height = numb_rows * self.piece_width

        # Shave off the edge of the image LAB and BGR images
        puzzle_upper_left = ((original_height - self._img_height) / 2, (original_width - self._img_width) / 2)
        self._img = Puzzle.extract_subimage(self._img, puzzle_upper_left, (self._img_height, self._img_width))
        self._img_LAB = Puzzle.extract_subimage(self._img_LAB, puzzle_upper_left, (self._img_height, self._img_width))

        # Break the board into pieces.
        piece_id = starting_id_numb
        piece_size = (self.piece_width, self.piece_width)
        self._pieces = []  # Create an empty array to hold the puzzle pieces.
        for row in range(0, numb_rows):
            for col in range(0, numb_cols):
                # No longer consider upper left since board shrunk above
                piece_upper_left = (row * piece_size[0], col * piece_size[1])
                piece_img = Puzzle.extract_subimage(self._img_LAB, piece_upper_left, piece_size)

                # Create the puzzle piece and assign to the location.
                location = (row, col)
                self._pieces.append(PuzzlePiece(self._id, location, piece_img,
                                                piece_id=piece_id, puzzle_grid_size=self._grid_size))
                # Increment the piece identification number
                piece_id += 1

    @property
    def numb_pieces(self):
        """
        Gets the number of pieces.  Note missing pieces are not counted in this statistic.

        Returns (int):
            Number of pieces in the puzzle

        """
        return len(self._pieces)

    @property
    def id_number(self):
        """
        Puzzle Identification Number

        Gets the identification number for a puzzle.

        Returns (int):
            Identification number for the puzzle
        """
        return self._id

    @property
    def pieces(self):
        """
        Gets all of the pieces in this puzzle.

        Returns (List[PuzzlePiece]):
            A list of the pieces in the puzzle
        """
        return self._pieces

    @property
    def piece_width(self):
        """
        Gets the size of a puzzle piece.

        Returns (int):
            Height/width of each piece in number of pixels.

        """
        return self._piece_width

    @staticmethod
    def reconstruct_from_pieces(pieces, id_numb=-1, display_image=False):
        """
        Constructs a puzzle from a set of pieces.

        Args:
            pieces ([PuzzlePiece]): Set of puzzle pieces that comprise the puzzle.
            id_numb (Optional int): Identification number for the puzzle
            display_image (Optional bool): Select whether to display the image at the end of reconstruction

        Returns (Puzzle):
            Puzzle constructed from the pieces.
        """

        if len(pieces) == 0:
            raise ValueError("Error: Each puzzle must have at least one piece.")

        # Create the puzzle to return.  Give it an ID number.
        output_puzzle = Puzzle(id_numb)
        output_puzzle._id = id_numb

        # Create a copy of the pieces.
        output_puzzle._pieces = copy.deepcopy(pieces)

        # Get the first piece and use its information
        first_piece = output_puzzle._pieces[0]
        output_puzzle._piece_width = first_piece.width

        # Find the min and max row and column.
        (min_row, max_row, min_col, max_col) = output_puzzle.get_min_and_max_row_and_columns(True)

        # Normalize each piece's locations based off all the pieces in the board.
        for piece in output_puzzle._pieces:
            loc = piece.location
            piece.location = (loc[0] - min_row, loc[1] - min_col)
        output_puzzle.reset_upper_left_location()

        # Construct the puzzle image.
        output_puzzle.build_puzzle_image()

        # Convert the image to LAB format.
        output_puzzle._img_LAB = cv2.cvtColor(output_puzzle._img, cv2.COLOR_BGR2LAB)
        if display_image:
            Puzzle.display_image(output_puzzle._img)

        return output_puzzle

    # def _get_piece_by_id_number(self, piece_id_number):
    #     """
    #     Pieces in a puzzle are stored in a list.  This function allows for constant time look up
    #     of a puzzle piece based off its identification number.
    #
    #     Args:
    #         piece_id_number (int): Identification number of the puzzle piece
    #
    #     Returns (PuzzlePiece): Puzzle piece with the specified identification number.
    #     """
    #
    #     # Build the indexing of piece
    #     if not self._id_number_to_index_map:
    #         for i in xrange(0, self.numb_pieces):
    #             self._id_number_to_index_map[str(self._pieces[i].id_numb)] = i
    #
    #     # Return the piece with the specified index number.
    #     pieces_array_index = self._id_number_to_index_map[str(piece_id_number)]
    #     return self._pieces[pieces_array_index]

    def save_segment_color_image(self, filename):
        """
        Saves a copy of the image segments to the specified output file.

        Args:
            filename (string): Filename and path the image will be output to.
        """
        # Build the output image using the segment coloring
        self.build_puzzle_image(use_segment_coloring=True)
        # Save the image to a file.
        Puzzle._save_to_file(filename, self._img)

    def build_puzzle_image(self, use_results_coloring=False, use_segment_coloring=False):
        """
        Makes a puzzle image.

        Args:
            use_results_coloring (Optional bool): If set to true, each piece's image is not based off the original
              image but what was stored based off the results.
            use_segment_coloring (Optional bool): Use the segment coloring scheme to build the puzzle image

        Returns (Puzzle):
            Puzzle constructed from the pieces.
        """
        if Puzzle._PERFORM_ASSERT_CHECKS:
            assert not use_results_coloring or not use_segment_coloring

        # noinspection PyTypeChecker
        self._img = PuzzlePiece.create_solid_image(SolidColor.black, self._img_width, self._img_height)

        # Insert the pieces into the puzzle
        for piece in self._pieces:
            self.insert_piece_into_image(piece, use_results_coloring, use_segment_coloring)

    def reset_upper_left_location(self):
        """
        Reset Upper Left Location

        "Upper Left" refers to the coordinate of the upper-leftmost piece.  By running this function, you
        update this reference point to (0, 0)
        """
        self._upper_left = (0, 0)

    def determine_standard_direct_accuracy(self, expected_puzzle_id, numb_pieces_in_original_puzzle):
        """
        Standard Direct Accuracy Finder

        Determines the accuracy of the placement using the standard direct accuracy method.

        Args:
            expected_puzzle_id (int): Expected puzzle identification number
            numb_pieces_in_original_puzzle (int): Number of pieces in the original puzzle

        Returns (DirectAccuracyPuzzleResults):
            Information regarding the direct accuracy of the placement

        """
        return self.determine_modified_direct_accuracy(expected_puzzle_id, (0, 0), numb_pieces_in_original_puzzle)

    def determine_modified_direct_accuracy(self, expected_puzzle_id, upper_left, numb_pieces_in_original_puzzle):
        """
        Modified Direct Accuracy Finder

        Determines the accuracy of the placement using the modified direct accuracy method where by the piece accuracy
        is determined by something other than the exact upper-most left piece (i.e. location (0, 0)).

        Args:
            expected_puzzle_id (int): Expected puzzle identification number
            upper_left(Tuple[int]): In the direct method, the upper-left most location is the origin.  In the
            "Modified Direct Method", this can be a tuple in the format (row, column).
            numb_pieces_in_original_puzzle (int): Number of pieces in the original puzzle

        Returns (DirectAccuracyPuzzleResults):
            Information regarding the modified direct accuracy of the placement

        """
        # Optionally perform the assertion checks
        if Puzzle._PERFORM_ASSERT_CHECKS:
            assert self._upper_left == (0, 0)  # Upper left should be normalized to zero.

        # Determine the accuracy assuming the upper left is in the normal location (i.e. (0,0))
        accuracy_info = DirectAccuracyPuzzleResults(expected_puzzle_id, self.id_number, numb_pieces_in_original_puzzle)

        # Iterate through each piece and determine its accuracy results.
        for piece in self._pieces:

            # Ensure that the puzzle ID matches the requirement
            if piece.original_puzzle_id != expected_puzzle_id:
                accuracy_info.add_different_puzzle(piece)

            # Ensure that the puzzle piece is in the correct location
            elif not piece.is_correctly_placed(upper_left):
                accuracy_info.add_wrong_location(piece)

            # Ensure that the rotation is set to 0
            elif piece.rotation != PuzzlePieceRotation.degree_0:
                accuracy_info.add_wrong_rotation(piece)

            # Piece is correctly placed
            else:
                accuracy_info.add_correct_placement(piece)

        return accuracy_info

    def randomize_puzzle_piece_locations(self):
        """
        Puzzle Piece Location Randomizer

        Randomly assigns puzzle pieces to different locations.
        """

        # Get all locations in the image.
        all_locations = []
        for piece in self._pieces:
            all_locations.append(piece.location)

        # Shuffle the image locations
        random.shuffle(all_locations)

        # Reassign the pieces to random locations
        for i in range(0, len(self._pieces)):
            self._pieces[i].location = all_locations[i]

    def randomize_puzzle_piece_rotations(self):
        """
        Puzzle Piece Rotation Randomizer

        Assigns a random rotation to each piece in the puzzle.
        """
        for piece in self._pieces:
            piece.rotation = PuzzlePieceRotation.random_rotation()

    def set_piece_color_for_direct_accuracy(self, direct_acc_results):
        """
        Colorizes a piece based off the direct accuracy results.

        Args:
            direct_acc_results (DirectAccuracyPuzzleResults): Direct accuracy (either standard or modified) for
            the solved image.

        """
        # Iterate through each puzzle piece and set its piece color
        for piece in self._pieces:
            piece.results_image_coloring = direct_acc_results.get_piece_result(piece.id_number)

    def get_min_and_max_row_and_columns(self, update_board_dimension_information):
        """
        Min/Max Row and Column Finder

        For a given set of pieces, this function returns the minimum and maximum of the columns and rows
        across all of the pieces.

        Args:
            update_board_dimension_information (bool): Selectively update the board dimension information (e.g.
            upper left location, grid_size, puzzle width, puzzle height, etc.).

        Returns ([int]):
            Tuple in the form: (min_row, max_row, min_column, max_column)
        """
        first_piece = self._pieces[0]
        min_row = max_row = first_piece.location[0]
        min_col = max_col = first_piece.location[1]
        for i in range(0, len(self._pieces)):
            # Verify all pieces are the same size
            if Puzzle.print_debug_messages:
                assert(self.piece_width == self._pieces[i].width)
            # Get the location of the piece
            temp_loc = self._pieces[i].location
            # Update the min and max row if needed
            if min_row > temp_loc[0]:
                min_row = temp_loc[0]
            elif max_row < temp_loc[0]:
                max_row = temp_loc[0]
            # Update the min and max column if needed
            if min_col > temp_loc[1]:
                min_col = temp_loc[1]
            elif max_col < temp_loc[1]:
                max_col = temp_loc[1]

        # If the user specified it, update the board dimension information
        if update_board_dimension_information:
            self._upper_left = (min_row, min_col)
            self._grid_size = (max_row - min_row + 1, max_col - min_col + 1)
            # Calculate the size of the image
            self._img_width = self._grid_size[1] * self.piece_width
            self._img_height = self._grid_size[0] * self.piece_width

        # Return the minimum and maximum row/column information
        return min_row, max_row, min_col, max_col

    def assign_all_pieces_to_original_location(self):
        """Piece Correct Assignment

        Assigns each piece to its original location for debug purposes.

        ::Note:: This should NOT be used in a solver; that would be cheating.
        """
        for piece in self._pieces:
            # noinspection PyProtectedMember
            piece._assign_to_original_location()

    def assign_all_pieces_to_same_rotation(self, rotation):
        """
        Pieces Rotation Assigner

        Assigns each piece to a specified, single rotation

        ::Note:: This should NOT be used in a solver; that would be cheating.

        Args:
            rotation (PuzzlePieceRotation): Rotation to assign to all pieces.

        """
        for piece in self._pieces:
            # noinspection PyProtectedMember
            piece.rotation = rotation

    @property
    def grid_size(self):
        """
        Puzzle Grid Size Property

        Used to get the maximum dimensions of the puzzle in number of rows by number of columns.

        Returns ([int)):
            Grid size of the puzzle as a tuple in the form (number_rows, number_columns)

        """
        return self._grid_size

    # noinspection PyUnusedLocal
    @staticmethod
    def create_solid_bgr_image(size, color):
        """
        Solid BGR Image Creator

        Creates a BGR Image (i.e. NumPy) array of the specified size.

        RIGHT NOW ONLY BLACK is supported.

        Args:
            size ([int]): Size of the image in height by width
            color (ImageColor): Solid color of the image.

        Returns:
            NumPy array representing a BGR image of the specified solid color
        """
        dimensions = (size[0], size[1], Puzzle.NUMBER_BGR_DIMENSIONS)
        return np.zeros(dimensions, np.uint8)

    @staticmethod
    def extract_subimage(img, upper_left, size):
        """
        Given an image (in the form of a Numpy array) extract a subimage.

        Args:
            img : Image in the form of a Numpy array.
            upper_left ([int]): upper left location of the image to extract
            size ([int]): Size of the of the sub

        Returns (Numpy[int]):
            Sub image as a Numpy array
        """

        # Calculate the lower right of the image
        img_end = []
        for i in range(0, 2):
            img_end.append(upper_left[i] + size[i])

        # Return the sub image.
        return img[upper_left[0]:img_end[0], upper_left[1]:img_end[1], :]

    def insert_piece_into_image(self, piece, use_results_coloring=False, use_segment_coloring=False):
        """
        Takes a puzzle piece and converts its image into BGR then adds it to the master image.

        Args:
            piece (PuzzlePiece): Puzzle piece to be inserted into the puzzle's image.
            use_results_coloring (Optional bool): If set to true, each piece's image is not based off the original
              image but what was stored based off the results.
            use_segment_coloring (Optional bool): Use the segment colors when generating the image
        """
        if Puzzle._PERFORM_ASSERT_CHECKS:
            assert not use_results_coloring or not use_segment_coloring

        piece_loc = piece.puzzle_location

        # Define the upper left corner of the piece to insert
        upper_left = (piece_loc.row * piece.width, piece_loc.column * piece.width)

        # Determine whether to use the image or something based off the results
        puzzle_img = self._img

        # Use the result coloring
        if use_results_coloring:
            results_coloring = piece.results_image_coloring
            # Verify the piece has actual results information.
            if Puzzle._PERFORM_ASSERT_CHECKS:
                assert results_coloring is not None
            # If the item is not a list, it is solid coloring
            if not isinstance(results_coloring, list):
                piece_img = PuzzlePiece.create_solid_image(results_coloring, piece.width)
            else:
                piece_img = PuzzlePiece.create_side_polygon_image(results_coloring, piece.width)
        # Use the segment coloring.
        elif use_segment_coloring:
            # Verify the piece actually has a segment color
            if Puzzle._PERFORM_ASSERT_CHECKS:
                assert piece.has_segment_color()
            # Use the segment color as the piece image.
            if not piece.is_stitching_piece:
                piece_img = PuzzlePiece.create_solid_image(piece.segment_color, piece.width)
            else:

                piece_img = PuzzlePiece.create_solid_piece_with_identification_mark(piece.segment_color, piece.width)

        # Use the piece's actual image
        else:
            piece_img = piece.bgr_image()

        # Select whether to display the image rotated
        if piece.rotation is None or piece.rotation == PuzzlePieceRotation.degree_0:
            Puzzle.insert_subimage(puzzle_img, upper_left, piece_img)
        else:
            rotated_img = np.rot90(piece_img, (PuzzlePieceRotation.degree_360.value - piece.rotation.value) / 90)
            # Puzzle.display_image(piece_img)
            # Puzzle.display_image(rotated_img)
            Puzzle.insert_subimage(puzzle_img, upper_left, rotated_img)

    @staticmethod
    def get_file_extension(filename):
        """
        Get the file extension of an object.

        Args:
            filename (str): Filename of an object

        Returns (str):
            File extension of the specified filename (without the period)
        """
        import os
        ext = str(os.path.basename(filename)).split('.', 1)[1]
        return ext if ext else None

    @staticmethod
    def get_filename_without_extension(filename_and_path):
        """
        Get the file extension of an object.

        Args:
            filename_and_path (str): Filename of an object

        Returns (str):
            Filename of the object with the path and file extension removed.
        """
        return os.path.splitext(os.path.basename(filename_and_path))[0]

    @staticmethod
    def make_image_filename(puzzle_solver_type, image_filenames, image_descriptor, output_directory, puzzle_type, timestamp,
                            orig_img_filename=None, puzzle_id=None):
        """
        Builds an image file name using a set of standard parameters.

        Args:
            puzzle_solver_type (PuzzleSolver): Type of solver being run
            image_filenames (list[str]): List of image files
            image_descriptor (str): Descriptor of the output puzzle.
            output_directory (str): Path where the file should be output
            puzzle_type (PuzzleType): Type of the puzzle
            timestamp (float): Timestamp to be associated with the file
            orig_img_filename (Optional str): Path to the original file name
            puzzle_id (Optional int): Identification number of the puzzle.

        Returns (str):
            Image file name constructed from the specified parameters.
        """
        filenames_combined = ""
        for img_file in image_filenames:
            if filenames_combined:
                filenames_combined += "."
            filenames_combined += Puzzle.get_filename_without_extension(img_file)

        image_directory = output_directory + puzzle_solver_type.value + "_" + filenames_combined
        ts_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y.%m.%d_%H.%M.%S')
        image_directory += "_type" + str(puzzle_type.value) + "_" + ts_str + "/"

        # Store the reconstructed image
        output_filename = image_directory + image_descriptor + "_type" + str(puzzle_type.value)

        # If a specific filename is specified, use that.
        img_extension = None
        if orig_img_filename is not None:
            img_extension = Puzzle.get_file_extension(orig_img_filename)
            img_root_filename = Puzzle.get_filename_without_extension(orig_img_filename)
            output_filename += "_" + img_root_filename

        if puzzle_id is not None:
            output_filename += "_" + "solved_puzzle_" + ("%04d" % puzzle_id)

        # Add timestamp and file extension
        output_filename += "_" + ts_str
        if img_extension is not None:
            output_filename += "." + img_extension
        else:
            output_filename += ".jpg"

        return output_filename

    @staticmethod
    def insert_subimage(master_img, upper_left, subimage):
        """
        Given an image (in the form of a NumPy array), insert another image into it.

        Args:
            master_img : Image in the form of a NumPy array where the sub-image will be inserted
            upper_left ([int]): upper left location of the the master image where the sub image will be inserted
            subimage ([int]): Sub-image to be inserted into the master image.

        Returns (Numpy[int]):
            Sub image as a NumPy array
        """

        # Verify the upper left input value is valid.
        if Puzzle.print_debug_messages and upper_left[0] < 0 or upper_left[1] < 0:
            raise ValueError("Error: upper left is off the image grid. Row and column must be >=0")

        # Calculate the lower right of the image
        subimage_shape = subimage.shape
        bottom_right = [upper_left[0] + subimage_shape[0], upper_left[1] + subimage_shape[1]]

        # Verify that the shape information is valid.
        if Puzzle.print_debug_messages:
            master_shape = master_img.shape
            assert master_shape[0] >= bottom_right[0] and master_shape[1] >= bottom_right[1]

        # Insert the subimage.
        master_img[upper_left[0]:bottom_right[0], upper_left[1]:bottom_right[1], :] = subimage

    @staticmethod
    def display_image(img):
        """
        Displays the image in a window for debug viewing.

        Args:
            img: OpenCV image in the form of a Numpy array

        """
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def _save_to_file(filename, img):
        """
        Save Image to a File

        Saves any NumPy array to an image file.

        Args:
            filename (str): Filename and path to save the OpenCV image.
            img: OpenCV image in the form of a Numpy array
        """
        # If the file directory does not exist create it.
        file_directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.isdir(file_directory):
            print "Creating solved image directory: \"%s\"." % file_directory
            os.makedirs(file_directory)

        # Write the image to disk.
        cv2.imwrite(filename, img)

    def save_to_file(self, filename):
        """
        Save Puzzle to a File

        Saves a puzzle to the specified file name.

        Args:
            filename (str): Filename and path to save the OpenCV image.
        """
        Puzzle._save_to_file(filename, self._img)

    def build_placed_piece_info(self):
        """
        Placed Piece Info Builder

        For a puzzle, this function builds two Numpy 2D arrays. They are:


        1. Piece Locations - A Numpy 2D matrix showing the PUZZLE PIECE ID NUMBER in each puzzle location.  If a puzzle
        piece location has no assigned piece, then the cell is filled with the Puzzle class's static property
        "MISSING_PIECE_PUZZLE_INFO_VALUE"

        2. Piece Rotations - A Numpy 2D matrix showing the ROTATION VALUE NUMBER the puzzle piece in each puzzle
        location.  If no puzzle piece is assigned to a specific location, then the cell is filled in with the
        Puzzle class's static property "MISSING_PIECE_PUZZLE_INFO_VALUE"

        Returns (Tuple[Numpy[int]]):
            Location of each puzzle piece in the grid
        """

        # Check whether the upper location is (0, 0)
        if Puzzle._PERFORM_ASSERT_CHECKS:
            assert self._upper_left == (0, 0)

        # Build a NumPy array that is by default "None" for each cell.
        placed_piece_matrix = np.full(self._grid_size, -1, np.int32)
        placed_piece_rotation = np.full(self._grid_size, -1, np.int32)

        # For each element in the array,
        for piece in self._pieces:
            placed_piece_matrix[piece.location] = piece.original_piece_id
            placed_piece_rotation[piece.location] = piece.rotation.value

        # Return the built NumPy array
        return placed_piece_matrix, placed_piece_rotation

    @staticmethod
    def get_side_of_primary_adjacent_to_other_piece(primary_piece_location, other_piece_location):
        """
        Given two adjacent pieces (i.e. a primary piece and an other piece), return the side of the primary
        piece that is adjacent (i.e. touching) the other piece.

        Args:
            primary_piece_location (PuzzleLocation): Location of the primary piece
            other_piece_location (PuzzleLocation): Location of the other piece

        Returns (PuzzlePieceSide):
            Side of the primary piece adjacent to the other piece.
        """

        diff_row = primary_piece_location.location[0] - other_piece_location.location[0]
        diff_col = primary_piece_location.location[1] - other_piece_location.location[1]

        # Verify the locations are actually adjacent
        # noinspection PyProtectedMember
        if Puzzle._PERFORM_ASSERT_CHECKS:
            assert primary_piece_location.puzzle_id == other_piece_location.puzzle_id
            # Verify the locations are exactly one space away
            assert abs(diff_row) + abs(diff_col) == 1

        # Determine the side of the adjacent piece.
        if diff_row == 1:
            return PuzzlePieceSide.top
        if diff_row == -1:
            return PuzzlePieceSide.bottom
        if diff_col == 1:
            return PuzzlePieceSide.left
        if diff_col == -1:
            return PuzzlePieceSide.right

    def assign_all_piece_id_numbers_to_original_id(self):
        """
        Sets the piece ID number to its original ID number.  Should not generally be used in a puzzle solver.
        """
        for piece in self._pieces:
            piece.id_number = piece.original_piece_id

    @staticmethod
    def get_combined_pieces_multiple_images(image_files, piece_width):
        """
        Builds a set of puzzle pieces from across a set of image files.
        Args:
            image_files (List[str]): Paths to image files
            piece_width (int): Width of a puzzle piece in number of pixels

        Returns (List[PuzzlePieces]): Pieces combined from across the images/
        """
        combined_pieces = []
        puzzles = []
        for i in range(0, len(image_files)):
            # Define the identification number of the first piece
            starting_piece_id = len(combined_pieces)

            # Build the puzzle and add it to the list of puzzles
            new_puzzle = Puzzle(i, image_files[i], piece_width, starting_piece_id)
            puzzles.append(new_puzzle)

            # Concatenate to the list of all pieces.
            combined_pieces += puzzles[i].pieces
        return combined_pieces, puzzles

    @staticmethod
    def output_results_information_and_puzzles(puzzle_solver_type, image_files, paikin_tal_solver,
                                               solved_puzzles, timestamp, export_segment_images=False):
        """
        Generates the results information of the solved puzzles.

        Args:
            puzzle_solver_type (PuzzleSolver): Type of Solver

            image_files (List[str]): List of paths to the input image files.

            paikin_tal_solver (PaikinTalSolver): A completed Paikin & Tal solver object

            solved_puzzles (List[List[PuzzlePieces]]): A list of lists of PuzzlePieces.  The pieces
             are organized based off their output puzzle from the solver.

            timestamp (int): Timestamp to be used for indicating puzzle image information.

            export_segment_images (bool): True if export_segment_images should be exported and False otherwise.
        """

        # Iterate through all the puzzles.  Reconstruct them and get their accuracies.
        output_puzzles = []
        for puzzle_pieces in solved_puzzles:
            # Get the first piece of the puzzle and extract information on it.
            first_piece = puzzle_pieces[0]
            puzzle_id = first_piece.puzzle_id

            # Reconstruct the puzzle
            new_puzzle = Puzzle.reconstruct_from_pieces(puzzle_pieces, puzzle_id)

            # Store the reconstructed image
            filename_descriptor = "reconstructed"
            if len(image_files) == 1:
                orig_img_filename = image_files[0]
                puzzle_id_filename = None
            else:
                orig_img_filename = None
                puzzle_id_filename = puzzle_id
            filename = Puzzle.make_image_filename(puzzle_solver_type, image_files, filename_descriptor,
                                                  Puzzle.OUTPUT_IMAGE_DIRECTORY, paikin_tal_solver.puzzle_type, timestamp,
                                                  orig_img_filename=orig_img_filename, puzzle_id=puzzle_id_filename)
            new_puzzle.save_to_file(filename)

            if export_segment_images:
                # Save the segmented file image
                filename_descriptor = "segmented"
                segment_filename = Puzzle.make_image_filename(puzzle_solver_type, image_files, filename_descriptor,
                                                              Puzzle.OUTPUT_IMAGE_DIRECTORY, paikin_tal_solver.puzzle_type,
                                                              timestamp, orig_img_filename=orig_img_filename,
                                                              puzzle_id=puzzle_id_filename)
                new_puzzle.save_segment_color_image(segment_filename)

            # Append the puzzle to the list
            output_puzzles.append(new_puzzle)

        # Determine the image filename
        orig_img_filename = image_files[0] if len(image_files) == 1 else None
        # Print the best buddy accuracy information
        paikin_tal_solver.best_buddy_accuracy.print_results()
        paikin_tal_solver.best_buddy_accuracy.output_results_images(puzzle_solver_type, image_files, output_puzzles,
                                                                    paikin_tal_solver.puzzle_type,
                                                                    timestamp, orig_img_filename=orig_img_filename)

        # Build the results information collection
        results_information = PuzzleResultsCollection(puzzle_solver_type, paikin_tal_solver.puzzle_type,
                                                      solved_puzzles, image_files)
        # Calculate and print the accuracy results
        results_information.calculate_accuracies(output_puzzles)
        # Print the results to the console
        results_information.print_results()
        # Print the results as image files
        results_information.output_results_images(puzzle_solver_type, image_files, output_puzzles,
                                                  paikin_tal_solver.puzzle_type, timestamp)
        results_information.save_results_to_file()


class PuzzleTester(object):
    """
    Puzzle tester class used for debugging the solver.
    """

    PIECE_WIDTH = 5
    NUMB_PUZZLE_PIECES = 9
    GRID_SIZE = (int(math.sqrt(NUMB_PUZZLE_PIECES)), int(math.sqrt(NUMB_PUZZLE_PIECES)))
    NUMB_PIXEL_DIMENSIONS = 3

    TEST_ARRAY_FIRST_PIXEL_VALUE = 0

    # Get the information on the test image
    TEST_IMAGE_FILENAME = "." + os.sep + "test" + os.sep + "test.jpg"
    TEST_IMAGE_WIDTH = 300
    TEST_IMAGE_HEIGHT = 200

    @staticmethod
    def build_pixel_list(start_value, is_row, reverse_list=False):
        """
        Pixel List Builder

        Given a starting value for the first pixel in the first dimension, this function gets the pixel values
        in an array similar to a call to "get_row_pixels" or "get_column_pixels" for a puzzle piece.

        Args:
            start_value (int): Value of the first (i.e. lowest valued) pixel's first dimension

            is_row (bool): True if building a pixel list for a row and "False" if it is a column.  This is used to
            determine the stepping factor from one pixel to the next.

            reverse_list (bool): If "True", HIGHEST valued pixel dimension is returned in the first index of the list
            and all subsequent pixel values are monotonically DECREASING.  If "False", the LOWEST valued pixel dimension
            is returned in the first index of the list and all subsequent pixel values are monotonically increasing.

        Returns ([int]):
            An array of individual values simulating a set of pixels
        """

        # Determine the pixel to pixel step size
        if is_row:
            pixel_offset = PuzzleTester.NUMB_PIXEL_DIMENSIONS
        else:
            pixel_offset = PuzzleTester.row_to_row_step_size()

        # Build the list of pixel values
        pixels = np.zeros((PuzzleTester.PIECE_WIDTH, PuzzleTester.NUMB_PIXEL_DIMENSIONS), np.int8)
        for i in range(0, PuzzleTester.PIECE_WIDTH):
            pixel_start = start_value + i * pixel_offset
            for j in range(0, PuzzleTester.NUMB_PIXEL_DIMENSIONS):
                pixels[i, j] = pixel_start + j

        # Return the result either reversed or not.
        if reverse_list:
            return pixels[::-1]
        else:
            return pixels

    @staticmethod
    def row_to_row_step_size():
        """
        Row to Row Step Size

        For a given pixel's given dimension, this function returns the number of dimensions between this pixel and
        the matching pixel exactly one row below.

        It is essentially the number of dimensions multiplied by the width of the original image (in pixels).

        Returns (int):
            Offset in dimensions.
        """
        step_size = PuzzleTester.NUMB_PIXEL_DIMENSIONS * PuzzleTester.PIECE_WIDTH * math.sqrt(PuzzleTester.NUMB_PUZZLE_PIECES)
        return int(step_size)

    @staticmethod
    def piece_to_piece_step_size():
        """
        Piece to Piece Step Size

        For a given pixel's given dimension, this function returns the number of dimensions between this pixel and
        the matching pixel exactly one puzzle piece away.

        It is essentially the number of dimensions multiplied by the width of a puzzle piece (in pixels).

        Returns (int):
            Offset in dimensions.
        """
        return PuzzleTester.NUMB_PIXEL_DIMENSIONS * PuzzleTester.PIECE_WIDTH

    @staticmethod
    def build_dummy_puzzle():
        """
        Dummy Puzzle Builder

        Using an image on the disk, this function builds a dummy puzzle using a Numpy array that is manually
        loaded with sequentially increasing pixel values.

        Returns (Puzzle):
            A puzzle where each pixel dimension from left to right sequentially increases by one.
        """

        # Create a puzzle whose image data will be overridden
        puzzle = Puzzle(0, PuzzleTester.TEST_IMAGE_FILENAME)

        # Define the puzzle side
        piece_width = PuzzleTester.PIECE_WIDTH
        numb_pieces = PuzzleTester.NUMB_PUZZLE_PIECES
        numb_dim = PuzzleTester.NUMB_PIXEL_DIMENSIONS

        # Define the array
        dummy_img = np.zeros((int(round(piece_width * math.sqrt(numb_pieces))),
                              int(round(piece_width * math.sqrt(numb_pieces))),
                              numb_dim),
                             dtype=np.int8)
        # populate the array
        val = PuzzleTester.TEST_ARRAY_FIRST_PIXEL_VALUE
        img_shape = dummy_img.shape
        for row in range(0, img_shape[0]):
            for col in range(0, img_shape[1]):
                for dim in range(0, img_shape[2]):
                    dummy_img[row, col, dim] = val
                    val += 1

        # Overwrite the image parameters
        puzzle._img = dummy_img
        puzzle._img_LAB = dummy_img
        puzzle._img_width = img_shape[1]
        puzzle._img_height = img_shape[0]
        puzzle._piece_width = PuzzleTester.PIECE_WIDTH
        puzzle._grid_size = (math.sqrt(PuzzleTester.NUMB_PUZZLE_PIECES), math.sqrt(PuzzleTester.NUMB_PUZZLE_PIECES))

        # Remake the puzzle pieces
        puzzle.make_pieces()
        return puzzle


def log_puzzle_filenames(image_filenames):
    """
    Logs the name of the input image files used for the solver.

    Args:
        image_filenames (List[str]): Path to the input image files.
    """
    string_io = cStringIO.StringIO()
    print >> string_io, "Names of the Image Files:"
    for img_file in image_filenames:
        print >> string_io, "\t%s" % Puzzle.get_filename_without_extension(img_file)

    # log the result
    logging.critical(string_io.getvalue())
    string_io.close()
