<div align="center"><h1>TEAPOT</h1></div>

TEAPOT (**T**ool for **E**valuating **A**dversarial **P**erturbations **O**n **T**ext) is a toolkit to evaluate the effectiveness of adversarial perturbations on NLP systems by taking into account the preservation of meaning in the source.

Adversarial perturbations (perturbations to the input of a model that elicit large changes in the output), have been shown to be an effective way of assessing the robustness of machine models.
However, these perturbations only indicate weaknesses in the model if they do not change the input so significantly that it legitimately result in changes in the expected output. While this is easy to control when the input is real-valued (images for example), the situation is more problematic on discrete data such as natural language.

TEAPOT is an implementation of the evaluation framework described in the NAACL 2019 paper [On Evaluation of Adversarial Perturbations for Sequence-to-Sequence Models](link_to_paper) (**FIXME**: add link), wherein an adversarial attack is evaluated using two quantities:

- `s_src(x, x')`: a measure of the semantic similarity between the original input `x` and its adversarial perturbation `x'`.
- `d_tgt(y(x),y(x'),y*)`: a measure of how much the output similarity `s_tgt` decreases when the model is ran on the adversarial input (giving output `y(x')`) instead of the original input (giving `y(x)`). Specifically `d_tgt(y(x),y(x'),y*)` is defined as `0` if `s_tgt(y,y*)<s_tgt(y',y*)` and `(s_tgt(y,y*)-s_tgt(y',y*))/s_tgt(y,y*)` otherwise.

An attack is declared **successful** on `x,y` when `s_src(x, x')+d_tgt(y(x),y(x'),y*)>1`, in other words, *an adversarial attack changing `x` to `x'` is successful if it destroys the target more than it destroys the source*.

With TEAPOT, you can compute `s_src`, `d_tgt` and the success rate of an attack easily using proxy metrics for the source and target similarity (`chrF` by default).

## Usage

Given the original input `example/src.fr`, adversarial input `example/adv.charswap.fr`, reference output `example/ref.en`, original output (output of the model on the original input) `example/base.en` and adversarial output (output of the model on the adversarial input) `example/adv.charswap.en`, running:

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

You can learn more about the command line options by running `teapot -h`. Notably you can specify which score to use with `--score {bleu,meteor,chrf}` (refer to the command line help for the list of scores implemented in your version).

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

