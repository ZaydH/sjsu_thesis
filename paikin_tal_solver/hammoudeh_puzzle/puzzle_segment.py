class PuzzleSegment(object):
    """
    This class is used to store a puzzle the information associated with a puzzle segment.
    """

    _PERFORM_ASSERTION_CHECKS = True

    def __init__(self, puzzle_id, id_number):
        """
        Puzzle segment constructor.

        Args:
            id_number (int): Unique identification number for the segment.
        """
        self._puzzle_id = puzzle_id
        self._id_number = id_number
        self._numb_pieces = 0
        self._piece_ids = {}

    @property
    def puzzle_id(self):
        """
        Each segment is only associated with a single solved puzzle.  This property is used to access the puzzle
        identification number of the implicitly associated segment.

        Returns (int): Identification number of the puzzle associated with this segment.

        """
        return self._puzzle_id

    @property
    def id_number(self):
        """
        Property for return the identification number for a puzzle segment

        Returns (int): Segment ID number
        """
        return self._id_number

    @property
    def numb_pieces(self):
        """
        Property that access the nuymber of pieces in the puzzle

        Returns (int): Number of pieces in the segment (minimum one)
        """
        return self._numb_pieces

    def add_piece(self, piece_id):
        """
        Adds a puzzle piece (as defined by the piece's identification number) to the puzzle segment.

        Args:
            piece_id (int): Identification if the puzzle piece to be added to the segment
        """
        key = PuzzleSegment._get_piece_key(piece_id)
        self._piece_ids[key] = piece_id

    def remove_piece(self, piece_id):
        """
        Removes a puzzle piece (as defined by the piece's identification number) from the puzzle segment.

        Args:
            piece_id (int): Identification if the puzzle piece to be removed from the segment

        """

        key = PuzzleSegment._get_piece_key(piece_id)
        # Optionally ensure the key exists before trying to remove it
        if PuzzleSegment._PERFORM_ASSERTION_CHECKS:
            assert key in self._piece_ids
        del self._piece_ids[key]

    @staticmethod
    def _get_piece_key(piece_id):
        """
        Puzzle piece identification numbers are stored in a Python dictinary.  As such, a key is needed to insert
        and remove elements from the dictionary.  This function is used to generate said key.

        Args:
            piece_id (int): Puzzle piece identification number.

        Returns (String): Key associated with the puzzle piece that is used by the PuzzleSegment class.
        """
        return str(piece_id)