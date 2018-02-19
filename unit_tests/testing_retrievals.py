import unittest
import numpy as np
from adlframework.retrievals.retrieval import Retrieval

class TestRetrieval(unittest.TestCase):

    def test_abstract_class(self):
        with self.assertRaises(TypeError):
            x = Retrieval()

suite = unittest.TestLoader().loadTestsFromTestCase(TestRetrieval)
unittest.TextTestRunner(verbosity=2).run(suite)
