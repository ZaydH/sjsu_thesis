import logging
import os
import pickle
import time

from hammoudeh_puzzle import config
from hammoudeh_puzzle.puzzle_importer import Puzzle
from hammoudeh_puzzle.solver_helper import print_elapsed_time


class PickleHelper(object):
    """
    The Pickle Helper class is used to simplify the importing and exporting of objects via the Python Pickle
    Library.
    """

    _PERFORM_ASSERT_CHECKS = config.PERFORM_ASSERT_CHECKS

    _PICKLE_DIRECTORY = ".\\pickle_files\\"

    @staticmethod
    def importer(filename):
        """
        Generic Pickling Importer Method

        Helper method used to import any object from a Pickle file.

        ::Note::: This function does not support objects of type "Puzzle."  They should use the class' specialized
        Pickling functions.

        Args:
            filename (str): Pickle Filename

        Returns:
            The object serialized in the specified filename.

        """
        start_time = time.time()

        # Check the file directory exists
        file_directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.isdir(file_directory):
            raise ValueError("The file directory: \"" + file_directory + "\" does not appear to exist.")

        logging.info("Beginning pickle IMPORT of file: \"" + filename + "\"")
        # import from the pickle file.
        f = open(filename, 'r')
        obj = pickle.load(f)
        f.close()

        logging.info("Completed pickle IMPORT of file: \"" + filename + "\"")
        print_elapsed_time(start_time, "pickle IMPORT of file: \"" + filename + "\"")
        return obj

    @staticmethod
    def exporter(obj, filename):
        """Generic Pickling Exporter Method

        Helper method used to export any object to a Pickle file.

        ::Note::: This function does not support objects of type "Puzzle."  They should use the class' specialized
        Pickling functions.

        Args:
            obj:                Object to be exported to a specified Pickle file.
            filename (str):     Name of the Pickle file.

        """
        start_time = time.time()

        # If the file directory does not exist create it.
        file_directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.isdir(file_directory):
            logging.debug("Creating pickle export directory \"%s\"." % file_directory)
            os.makedirs(file_directory)

        logging.info("Beginning pickle EXPORT of file: \"" + filename + "\"")
        # Dump pickle to the file.
        f = open(filename, 'w')
        pickle.dump(obj, f)
        f.close()

        logging.info("Completed pickle EXPORT to file: \"" + filename + "\"")
        print_elapsed_time(start_time, "pickle EXPORT of file: \"" + filename + "\"")

    ENABLE_PICKLE = True

    @staticmethod
    def build_filename(pickle_descriptor, image_filenames, puzzle_type):
        """
        Creates the filename of the pickle output file.

        Args:
            pickle_descriptor (string): Descriptor of the pickle contents
            image_filenames (List[string]): Image file names
            puzzle_type (PuzzleType): Type of the puzzle being solved

        Returns (str): Pickle filename
        """
        assert PickleHelper.ENABLE_PICKLE
        pickle_root_filename = ""
        for i in range(0, len(image_filenames)):
            # Get the root of the filename (i.e. without path and file extension
            img_root_filename = Puzzle.get_filename_without_extension(image_filenames[i])
            # Append the file name to the information
            pickle_root_filename += "_" + img_root_filename

        pickle_root_filename += "_type_" + str(puzzle_type.value) + ".pk"
        return PickleHelper._PICKLE_DIRECTORY + pickle_descriptor + pickle_root_filename
