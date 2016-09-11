import copy


class PuzzleSegment(object):
    """
    This class is used to store a puzzle the information associated with a puzzle segment.
    """

    _PERFORM_ASSERTION_CHECKS = True

    def __init__(self, puzzle_id, segment_id_number):
        """
        Puzzle segment constructor.

        Args:
            puzzle_id (int): Identification number of the solved puzzle
            segment_id_number (int): Unique identification number for the segment.
        """
        self._puzzle_id = puzzle_id
        self._segment_id_number = segment_id_number
        self._numb_pieces = 0
        self._pieces = {}
        self._seed_piece = None
        self._neighbor_segment_ids = {}

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
        return self._segment_id_number

    @property
    def numb_pieces(self):
        """
        Property that access the nuymber of pieces in the puzzle

        Returns (int): Number of pieces in the segment (minimum one)
        """
        return self._numb_pieces

    def add_piece(self, piece_id, piece_location):
        """
        Adds a puzzle piece (as defined by the piece's identification number) to the puzzle segment.

        Args:
            piece_id (int): Identification if the puzzle piece to be added to the segment
            piece_location (Location): Location of the puzzle piece
        """
        # Store the seed piece special
        if len(self._pieces) == 0:
            self._seed_piece = (piece_id, piece_location)

        # Store all pieces in the piece dictionary
        key = PuzzleSegment._get_piece_key(piece_id)
        self._pieces[key] = (piece_id, piece_location)

    def remove_piece(self, piece_id):
        """
        Removes a puzzle piece (as defined by the piece's identification number) from the puzzle segment.

        Args:
            piece_id (int): Identification if the puzzle piece to be removed from the segment
        """

        key = PuzzleSegment._get_piece_key(piece_id)
        # Optionally ensure the key exists before trying to remove it
        if PuzzleSegment._PERFORM_ASSERTION_CHECKS:
            assert key in self._pieces
        del self._pieces[key]

    def add_neighboring_segment(self, neighbor_segment_id):
        """
        Adds a neighboring segment number for this segment.  Neighboring segment numbers are used to determine the
        degree of each segment and for coloring the visualization.

        Args:
            neighbor_segment_id (int): Identification number of the neighboring segment.
        """
        segment_key = PuzzleSegment._get_segment_key(neighbor_segment_id)

        # Verify that the neighbor ID does not match its own ID
        if PuzzleSegment._PERFORM_ASSERTION_CHECKS:
            assert self._segment_id_number != neighbor_segment_id

        # If neighbor does not exist in the list, add the neighbor to the list
        if segment_key not in self._neighbor_segment_ids:
            self._neighbor_segment_ids[segment_key] = neighbor_segment_id

    def get_neighbor_segment_ids(self):
        """
        Accessor for the segments neighboring this segment.

        Returns (Dict[int]): The identification numbers of all segments that are adjacent to this segment.
        """
        # When getting the neighbor ids in the normal flow, this should not be blank
        if self._PERFORM_ASSERTION_CHECKS:
            assert self.neighbor_degree > 0

        return copy.deepcopy(self._neighbor_segment_ids)

    @property
    def neighbor_degree(self):
        """
        Property for getting the number of segments that are adjacent to this puzzle.

        Returns (int): Number of segments that are adjacent to this puzzle.
        """
        numb_neighbors = len(self._neighbor_segment_ids)

        # Verify the degree is greater than 0
        if PuzzleSegment._PERFORM_ASSERTION_CHECKS:
            assert numb_neighbors > 0

        return numb_neighbors

    @staticmethod
    def _get_piece_key(piece_id):
        """
        Puzzle piece identification numbers are stored in a Python dictionary.  As such, a key is needed to insert
        and remove elements from the dictionary.  This function is used to generate said key.

        Args:
            piece_id (int): Puzzle piece identification number.

        Returns (String): Key associated with the puzzle piece that is used by the PuzzleSegment class.
        """
        return str(piece_id)

    @staticmethod
    def _get_segment_key(segment_id):
        """
        Neighboring segment information is stored in a dictionary.  This function is used create that key identification
        number.

        Args:
            segment_id (int): Identification number of a segment

        Returns (String): Key associated with the puzzle segment identification number.
        """
        return str(segment_id)
