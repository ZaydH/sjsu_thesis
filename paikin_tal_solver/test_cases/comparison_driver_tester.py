"""Unittest Module for the Comparison Driver

.. moduleauthor:: Zayd Hammoudeh <hammoudeh@gmail.com>
"""

import unittest

from solver_comparison_driver import increment_puzzle_id_list


class ComparisonSolverTester(unittest.TestCase):

    def test_puzzle_identification_incrementer(self):
        """
        Verify the identification number incrementer works correctly.
        """
        self.assertTrue([1, 2, 4] == increment_puzzle_id_list([1, 2, 3], 20))

        self.assertTrue([1, 2, 20] == increment_puzzle_id_list([1, 2, 19], 20))

        self.assertTrue([1, 3, 4] == increment_puzzle_id_list([1, 2, 20], 20))

        self.assertTrue([1, 19, 20] == increment_puzzle_id_list([1, 18, 20], 20))

        self.assertTrue([2, 3, 4] == increment_puzzle_id_list([1, 19, 20], 20))

        self.assertTrue([19, 20, 21] == increment_puzzle_id_list([18, 19, 20], 20))
