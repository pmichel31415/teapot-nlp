import teapot

# Register your new scorer in teapot using
# @teapot.register_scorer(key, name)
# Where
#  - `key` is a shorthand for the name of your score, without spaces.
#     This is the value that you'll need to specify with `--score [key]`
#     when you run teapot
#  - `name` can be a longer and more descriptive name
#    (with spaces, uppercase, etc...). This will be used for logging


@teapot.register_scorer("f1", "F-1")
# Then subclass `teapot.Scorer`
class F1(teapot.Scorer):

    def score_sentence(self, hyp, ref, lang=None):
        """Score the similarity between 2 sentences

        This is the only function that you need to implement in most cases.
        It is assumed that the score will be in the [0, 1] interval.
        """
        # Note: it is better to assume that the 2 sentences are not tokenized
        # and perform tokenization yourself, for better reproducibility by
        # others. In any case, TEAPOT will complain (but not fail) if it
        # appears that the data is already tokenize. For the sake of
        # simplicity in this example we just split by space.
        hyp = hyp.split()
        ref = ref.split()
        # Precision and recall
        hyp_words = set(hyp)
        ref_words = set(ref)
        precision = len([word for word in hyp if word in ref_words]) / len(hyp)
        recall = len([word for word in ref if word in hyp_words]) / len(ref)
        # Harmonic mean
        if precision + recall == 0:
            return 0
        else:
            return 2 * precision * recall / (precision + recall)


# Here is a slightly more complicated example
# (this is a bit of a useless scorer that returns the same score for any input)

@teapot.register_scorer("constant", "F-1")
class Constant(teapot.Scorer):

    def __init__(self, value=0.5):
        self.value = value

    # In some case you might want to reimplement `score_corpus` directly,
    # for example if you need to call an external program
    # (see an example for meteor in teapot/scorers.py).
    # However in most cases you won't have to and teapot will just score
    # all sentences.
    def score_corpus(self, hyps, refs, lang=None):
        """This should return a list of independent scores, one for each
        element in hyps, refs (assuming they are of the same size)"""

        return [self.value] * len(hyps)

    @classmethod
    def add_args(cls, parser):
        """Add arguments specific to this scorer to the command line

        Use this if you want to make your scorer configurable.
        This will add arguments to the teapot command."""
        group = parser.add_argument_group("Constant related arguments")
        group.add_argument("--value", type=float, default=0.5,
                           help="Score constant value")

    @classmethod
    def from_args(cls, args):
        """If you added command line arguments with `add_args`, use this
        method to call the constructor using the command line arguments"""
        return cls(value=args.value)
