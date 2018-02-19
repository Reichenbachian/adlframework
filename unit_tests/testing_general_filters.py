import unittest
import numpy as np
from adlframework.filters.general_filters import min_array_shape
from adlframework.filters.general_filters import ignore_label
from adlframework.filters.general_filters import accept_label


class TestGeneralFilters(unittest.TestCase):
    def test_min_array_shape(self):
        sample = np.array([[1, 2, 3], [3, 4, 1]])
        self.assertTrue(min_array_shape(np.array([[1, 2, 3], [3, 4, 1]]), [1,2]))
        self.assertFalse(min_array_shape(sample, [7,2]))
        self.assertTrue(min_array_shape(sample, [2, 3]))

    def test_ignore_label(self):
        sample = np.array([[8, 2, 3], [3, 4, 9]])
        labelnames = [0, 1]
        self.assertFalse(ignore_label(sample, labelnames))

    def test_accept_label(self):
        sample = np.array([[8, 2, 3], [3, 4, 9]])
        labelnames = [0, 1]
        self.assertFalse(ignore_label(sample, labelnames))

suite = unittest.TestLoader().loadTestsFromTestCase(TestGeneralFilters)
unittest.TextTestRunner(verbosity=2).run(suite)
