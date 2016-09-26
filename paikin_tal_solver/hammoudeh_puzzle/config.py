import logging

DEFAULT_PIECE_WIDTH = 28

PERFORM_ASSERT_CHECKS = True


def setup_logging(filename="solver_driver.log", log_level=logging.DEBUG):
    """
    Configures the logger for process tasks

    Args:
        filename (str): Name of the log file to be generated
        log_level (int): Logger level (e.g. DEBUG, INFO, WARNING)

    """
    data_format = '%m/%d/%Y %I:%M:%S %p'  # Example Time Format - 12/12/2010 11:46:36 AM
    # noinspection SpellCheckingInspection
    logging.basicConfig(filename=filename, level=log_level, format='%(asctime)s -- %(message)s', datefmt=data_format)

    # Also print to stdout
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    logging.info("*********************************** New Run Beginning ***********************************")