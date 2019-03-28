import tempfile
import subprocess
import importlib.util

import re
import sacrebleu
from teapot import utils

scorers = {}


def register_scorer(keys, name):

    def register_func(cls):
        if not issubclass(cls, Scorer):
            raise ValueError(f"{cls.__name__} should be a subclass of Scorer")
        keys_list = keys
        if isinstance(keys_list, str):
            keys_list = [keys_list]
        for key in keys_list:
            if key in scorers:
                raise ValueError(f"Scorer {key} already exists")
            scorers[key] = cls
        cls._name = name
        return cls

    return register_func


class Scorer(object):
    _name = "base"

    @property
    def name(self):
        return self._name

    def score(self, hyps, refs, lang=None, check_tok=True):
        """Score a list of hypotheses"""
        if len(hyps) != len(refs):
            raise ValueError(
                "Mismatched input lengths "
                f"{len(hyps)}!={len(refs)}"
            )
        if check_tok:
            utils.check_tokenization(hyps)
            utils.check_tokenization(refs)
        return self.score_corpus(hyps, refs, lang=lang)

    def rd_score(self, hyps, bases, refs, lang=None, check_tok=True):
        """Relative decrease in score"""
        if check_tok:
            utils.check_tokenization(hyps)
            utils.check_tokenization(bases)
            utils.check_tokenization(refs)
        base_scores = self.score(bases, refs, lang=lang, check_tok=False)
        hyp_scores = self.score(hyps, refs, lang=lang, check_tok=False)
        rd_scores = [
            utils.relative_decrease(base_score, hyp_score)
            for base_score, hyp_score in zip(base_scores, hyp_scores)
        ]
        return rd_scores

    def score_corpus(self, hyps, refs, lang=None):
        return [self.score_sentence(hyp, ref, lang=lang)
                for hyp, ref in zip(hyps, refs)]

    def score_sentence(self, hyp, ref, lang=None):
        raise NotImplementedError()

    @classmethod
    def add_args(cls, parser):
        pass

    @classmethod
    def from_args(cls, args):
        return cls()


@register_scorer(["zero_one", "exact_match"], "accuracy")
class ZeroOne(Scorer):

    def score_sentence(self, hyp, ref, lang=None):
        return float(hyp == ref)


@register_scorer("bleu", "BLEU")
class BLEU(Scorer):

    def score_sentence(self, hyp, ref, lang=None):
        return sacrebleu.sentence_bleu(hyp, ref, smooth_value=0.01) / 100


@register_scorer("chrf", "ChrF")
class ChrF(Scorer):

    def score_sentence(self, hyp, ref, lang=None):
        return sacrebleu.sentence_chrf(hyp, ref)


@register_scorer("meteor", "METEOR")
class METEOR(Scorer):

    def __init__(self, meteor_jar, java_command="java -Xmx2G -jar"):
        self.meteor_jar = meteor_jar
        self.java_command = java_command

    def score_sentence(self, hyp, ref, lang=None):
        return self.score_corpus([hyp], [ref], lang=lang)

    def score_corpus(self, hyps, refs, lang=None):
        # Language must be specified
        if lang is None:
            raise ValueError("You need to specify a language for METEOR")
        # Save to temporary files
        hyp_file = tempfile.mktemp(suffix="hyp.txt")
        ref_file = tempfile.mktemp(suffix="ref.txt")
        utils.savetxt(ref_file, refs)
        utils.savetxt(hyp_file, hyps)
        # Call meteor
        meteor_command = f"{self.java_command} {self.meteor_jar}".split()
        meteor_args = [
            hyp_file,
            ref_file,
            "-norm",
            "-l",
            lang,
        ]
        out = ""
        try:
            out = subprocess.check_output(
                meteor_command + meteor_args,
                stderr=subprocess.STDOUT,
                encoding="utf8"
            )
        except subprocess.CalledProcessError:
            command_str = " ".join(meteor_command + meteor_args)
            raise ValueError(f"METEOR command `{command_str}` failed:\n{out}")
        # Parse output
        meteor_shower = []
        segment_score = re.compile(r"^Segment [0-9]* score:[^0-9]*([\.0-9]+)$")
        try:
            for line in out.split("\n"):
                score_match = segment_score.match(line)
                if score_match:
                    meteor_shower.append(float(score_match.group(1)))
        except Exception as e:
            last_lines_output = "\n".join(out.split("\n")[-10:])
            raise ValueError(
                f"There was an error ({e.__class__.__name__}) parsing the "
                "output of the METEOR command, here are the last 10 lines:\n"
                f"{last_lines_output}\nTraceback:\n{e.__traceback__}"
            )
        return meteor_shower

    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group("Meteor related arguments")
        group.add_argument(
            "--meteor-jar",
            metavar="FILE",
            type=str,
            required=True,
            help="Path to the meteor jar"
        )
        group.add_argument(
            "--java-command",
            type=str,
            default="java -Xmx2G -jar",
            help="Java command to run the jar"
        )

    @classmethod
    def from_args(cls, args):
        return cls(args.meteor_jar, java_command=args.java_command)


def read_custom_scorers_source(source_path):
    # Adapted from https://stackoverflow.com/a/67692
    spec = importlib.util.spec_from_file_location("module.name", source_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)


def get_scorer_class(key):
    if key not in scorers:
        raise ValueError(f"Scorer \"{key}\" does not exist")
    return scorers[key]


def scorers_from_args(args):
    return (
        get_scorer_class(args.s_src).from_args(args),
        get_scorer_class(args.s_tgt).from_args(args),
    )
