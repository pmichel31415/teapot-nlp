import os.path
import unittest

import sys
teapot_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(teapot_root)
import teapot  # noqa


class TestAPI(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.inputs = ["ba", "bb"]
        self.adv_inputs = ["aa", "ab"]
        self.outputs = ["00", "11"]
        self.adv_outputs = ["01", "01"]
        self.refs = ["10", "10"]

    def test_example_usage(self):
        chrf_scorer = teapot.ChrF()
        teapot.BLEU()
        teapot.METEOR("path/to/meteor.jar", java_command="java -Xmx2G -jar")
        chrf_scorer.score(self.adv_inputs, self.inputs)
        chrf_scorer.rd_score(self.adv_outputs, self.outputs, self.refs)
