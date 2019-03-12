# TEAPOT: a Tool for Evaluating Adversarial Perturbations On Text

TEAPOT is a toolkit to evaluate the effectiveness of adversarial perturbations on NLP systems by taking into account the preservation of meaning in the source sentence.

## Usage

```bash
teapot \
    --src example/src.fr \
    --adv-src example/adv.charswap.fr \
    --out example/base.en \
    --adv-out example/adv.charswap.en \
    --ref example/ref.en
```

will output:

```
Source side preservation (ChrF):
Mean:   86.908
Std:    11.622
5%-95%: 64.109-97.683
--------------------------------------------------------------------------------
Target side degradation (ChrF):
Mean:   21.085
Std:    22.106
5%-95%: 0.000-67.162
--------------------------------------------------------------------------------
Success percentage: 65.20 %
```

Alternatively you can specify only `--src` and `--adv-src` (for source side evaluation) *or* only `--out`,`--adv-out` and `--ref` (for target side evaluation).

You can learn more about the command line options by running `teapot -h`.

## Advanced usage

TEAPOT comes with predefined scores to compute the source and target side similarity. However, in some cases you might want to define you own score. Fortunately this can be done in a few steps if you are familiar with python:

1. Write your own `teapot.Scorer` subclass in a python source file (there are examples in [example/custom_scorers.py]). This is the hard part.
2. Call teapot with the arguments `--custom-scores-source path/to/your/scorer.py` and `--score [score_key]` where `[score_key]` is the shorthand you have defined for your scorer with `teapot.register_scorer` (again, see the examples in [example/custom_scorers.py] for a walkthrough)
3. If your scorer works fine, and it doesn't rely on heavy dependencies, consider contributing it to TEAPOT by
  1. Adding the class to [teapot/scorers.py]
  2. Adding a simple unit test to [tests/test_scorers.py]

Here is an example with the `f1` score defined in [example/custom_scorers.py]:

```bash
teapot \
    --src example/src.fr \
    --adv-src example/adv.charswap.fr \
    --out example/base.en \
    --adv-out example/adv.charswap.en \
    --ref example/ref.en \
    --custom-scores-source example/custom_scorers.py \
    --score f1
```

Or with the `constant` score (with auxiliary argument `--value`)

```bash
teapot \
    --src example/src.fr \
    --adv-src example/adv.charswap.fr \
    --out example/base.en \
    --adv-out example/adv.charswap.en \
    --ref example/ref.en \
    --custom-scores-source example/custom_scorers.py \
    --score constant \
    --value 0.3
```

