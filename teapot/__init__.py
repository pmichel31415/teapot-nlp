from teapot.scorers import Scorer, register_scorer
from teapot.scorers import BLEU, METEOR, ChrF


__version__ = "0.1"

__all__ = [
    "Scorer",
    "register_scorer",
    "BLEU",
    "METEOR",
    "ChrF",
]
