import os.path
import unittest

import sys
teapot_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(teapot_root)
from teapot import scorers  # noqa
from teapot import utils  # noqa


class TestBLEU(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.ref = (
            "But what's interesting is the unique cadence of the song, "
            "the rhythm of the dance in every culture."
        )
        self.hyp = (
            "But the remarkable rhythms of the song is interesting, "
            "the pace of dance in all cultures."
        )
        self.scorer = scorers.scorers["bleu"]()

    def test_score(self):
        [score] = self.scorer.score([self.hyp], [self.ref])
        self.assertAlmostEquals(score, 0.04615967330356789)


class TestChrF(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.ref = (
            "But what's interesting is the unique cadence of the song, "
            "the rhythm of the dance in every culture."
        )
        self.hyp = (
            "But the remarkable rhythms of the song is interesting, "
            "the pace of dance in all cultures."
        )
        self.scorer = scorers.scorers["chrf"]()

    def test_score(self):
        [score] = self.scorer.score([self.hyp], [self.ref])
        self.assertAlmostEquals(score, 0.49226509620151965)


class TestMETEOR(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.ref = (
            "But what's interesting is the unique cadence of the song, "
            "the rhythm of the dance in every culture."
        )
        self.hyp = (
            "But the remarkable rhythms of the song is interesting, "
            "the pace of dance in all cultures."
        )
        self.scorer = scorers.scorers["meteor"]("meteor-1.5/meteor-1.5.jar")

    def test_score(self):
        [score] = self.scorer.score([self.hyp], [self.ref], lang="en")
        self.assertAlmostEquals(score, 0.29357174078988885)
