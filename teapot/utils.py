import sys
from math import sqrt


def itertxt(filename):
    with open(filename, "r") as f:
        for line in f:
            yield line.rstrip()


def loadtxt(filename):
    return list(itertxt(filename))


def savetxt(filename, txt):
    with open(filename, "w") as f:
        for line in txt:
            print(line, file=f)


def stats(x):
    N = len(x)
    x = sorted(x)
    mean = sum(x) / N
    std = sqrt(sum([(x_i - mean) ** 2 for x_i in x]) / max(N - 1, .1))
    percentile_5 = x[int(N * 0.05)]
    percentile_95 = x[int(N * 0.95)]
    return mean, std, percentile_5, percentile_95


def relative_decrease(y, x):
    if y > 0:
        return max(0, y - x) / y
    else:
        return 0


def check_tokenization(sents):
    """Check whether an input text is tokenized (borrowed from sacreBLEU)"""
    too_much = 100
    tokenized_count = 0
    for sent in sents:
        if sent.endswith(' .'):
            tokenized_count += 1
            # Too much is too much
            if tokenized_count == too_much:
                print(
                    f"That's {too_much} lines that end in a "
                    "tokenized period ('.')",
                    file=sys.stderr,
                )
                print(
                    "It looks like you forgot to detokenize your data, "
                    "which may hurt your score and make your results "
                    "difficult to replicate.",
                    file=sys.stderr,
                )
                return too_much
    return tokenized_count
