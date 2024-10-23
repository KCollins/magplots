#!/usr/bin/env python
# Full license can be found in License.md
# Full author list can be found in .zenodo.json file
# DOI:10.5281/zenodo.10703227
# ----------------------------------------------------------------------------
"""Integration and unit test suite for magplots methods."""

import datetime
import unittest

import magplots.magFunctions as mp


class TestFindConjugate(unittest.TestCase):
    """Integration and unit test suite for conj_calc methods."""

    def setUp(self):
        """Initialize the test case by copying over necessary files."""
        self.start = datetime.datetime(2018, 9, 4, 0, 0, 0)
        self.end = datetime.datetime(2018, 9, 5, 0, 0, 0)
        self.magname = "atu"  
        self.resolution = "1sec"

    def tearDown(self):
        """Clean up the test environment."""
        del self.start, self.end, self.magname, self.resolution

    def eval_magfetch(self):
        """Evaluate the `magfetch` function."""
        result = magfetch(self.start, self.end, self.magname, self.resolution)
       
        # Assert that the returned values are close enough to expected values
        self.assertAlmostEqual(result["UT"][1],  datetime(2018, 9, 4, 0, 0, 1, 1000), delta=1)
