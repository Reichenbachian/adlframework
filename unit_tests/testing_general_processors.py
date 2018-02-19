import unittest
import numpy as np
from adlframework.processors.general_processors import to_np_arr
from adlframework.processors.general_processors import map_labels
from adlframework.processors.general_processors import threshold_labels
from adlframework.processors.general_processors import remove_outliers
from adlframework.processors.general_processors import smooth_gaussian_processor
from adlframework.processors.general_processors import amplitude_extractor

class TestGeneralProcessors(unittest.TestCase):

    def test_to_np_arr(self):
        sample = [[1, 2], [0, 0]]
        to_np = to_np_arr(sample)
        self.assertEquals(np.shape(to_np), np.shape(np.array([[1, 2], [0, 0]])))
        self.assertNotEquals(np.shape(to_np), np.shape(np.array([[1, 2], [0]])))

    def test_map_labels(self):
        sample = [[1, 2], [0, 0]]
        with self.assertRaises(AssertionError):
            map_labels(sample, targets = [])
        data, new = map_labels(sample, targets=[1,1])
        self.assertEquals(data, [1, 2])

    def test_threshold_labels(self):
        sample = [[1, 2], [0, 0]]
        with self.assertRaises(AssertionError):
            threshold_labels(sample)
        with self.assertRaises(AssertionError):
            threshold_labels([[1,2], "test"])

    def test_remove_outliers(self):
        sample = [[1, 2, 2, 1, 9], [0, 0, 1, 0, 1]]
        self.assertEquals(remove_outliers(sample, threshold = 2), ([1, 2, 2, 1, 3.0], [0, 0, 1, 0, 1]))
        self.assertEquals(remove_outliers(sample), ([1, 2, 2, 1, 3.0], [0, 0, 1, 0, 1]))

    def test_smooth_gaussian_processor(self):
        sample = [[1, 2, 2, 1, 9], [0, 0, 1, 0, 1]]
        with self.assertRaises(AssertionError):
            smooth_gaussian_processor(sample)
        self.assertEquals(len(smooth_gaussian_processor([0, 1])), 2)

    def test_amplitude_extractor(self):
        amp, label = amplitude_extractor([[0, 0, 0],[0,0, 1]])
        print amp, label
        self.assertEqual(len(amp), 3)


suite = unittest.TestLoader().loadTestsFromTestCase(TestGeneralProcessors)
unittest.TextTestRunner(verbosity=2).run(suite)
