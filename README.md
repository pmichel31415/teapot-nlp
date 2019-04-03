<img align="left" height="64" src="teapot.gif" alt="teapot"/>

# TEAPOT

TEAPOT (**T**ool for **E**valuating **A**dversarial **P**erturbations **O**n **T**ext) is a toolkit to evaluate the effectiveness of adversarial perturbations on NLP systems by taking into account the preservation of meaning in the source.

Adversarial perturbations (perturbations to the input of a model that elicit large changes in the output), have been shown to be an effective way of assessing the robustness of machine models.
However, these perturbations only indicate weaknesses in the model if they do not change the input so significantly that it legitimately result in changes in the expected output. While this is easy to control when the input is real-valued (images for example), the situation is more problematic on discrete data such as natural language.

TEAPOT is an implementation of the evaluation framework described in the NAACL 2019 paper [On Evaluation of Adversarial Perturbations for Sequence-to-Sequence Models](https://arxiv.org/abs/1903.06620), wherein an adversarial attack is evaluated using two quantities:

- `s_src(x, x')`: a measure of the semantic similarity between the original input `x` and its adversarial perturbation `x'`.
- `d_tgt(y(x),y(x'),y*)`: a measure of how much the output similarity `s_tgt` (w.r.t. the reference `y*`) decreases when the model is ran on the adversarial input (giving output `y(x')`) instead of the original input (giving `y(x)`). Specifically `d_tgt(y(x),y(x'),y*)` is defined as `0` if `s_tgt(y,y*) < s_tgt(y',y*)` and `(s_tgt(y,y*) - s_tgt(y',y*)) / s_tgt(y,y*)` otherwise.

An attack is declared **successful** on `x,y` when `s_src(x,x') + d_tgt(y(x),y(x'),y*) > 1`, in other words, *an adversarial attack changing `x` to `x'` is successful if it destroys the target more than it destroys the source*.

With TEAPOT, you can compute `s_src`, `d_tgt` and the success rate of an attack easily using proxy metrics for the source and target similarity (`chrF` by default).



## Getting started

### Installation and Requirements

TEAPOT works with python>=3.6. The only required non-standard dependency for teapot is [sacrebleu](https://github.com/mjpost/sacreBLEU) (a neat tool for computing BLEU and chrF on detokenized text). You can install with `python setup.py install` from the root of the repo, or simply `pip install teapot-nlp` from anywhere you want.

### Basic Usage (sequence-to-sequence)

Given the original input `examples/MT/src.fr`, adversarial input `examples/MT/adv.charswap.fr`, reference output `examples/MT/ref.en`, original output (output of the model on the original input) `examples/MT/base.en` and adversarial output (output of the model on the adversarial input) `examples/MT/adv.charswap.en`, running:

```bash
teapot \
  --src examples/MT/src.fr \
  --adv-src examples/MT/adv.charswap.fr \
  --out examples/MT/base.en \
  --adv-out examples/MT/adv.charswap.en \
  --ref examples/MT/ref.en
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

You can learn more about the command line options by running `teapot -h`. Notably you can specify which score to use with `--s-src` and `--s-tgt` (refer to the command line help for the list of scores implemented in your version).

### Basic Usage (other tasks)

While the default settings are geared towards evaluating attacks on sequence-to-sequence models, teapot can be used to evaluate attacks on otjer types of NLP models. For example, for a text classification model we might use the zero-one loss as `s_tgt` (`1` if the output label is the same as the reference, `0` otherwise). Here is an example with an attack on a sentiment classification model trained on [SST](https://nlp.stanford.edu/sentiment/):

```bash
teapot \
  --src examples/sentiment/src.txt \
  --adv-src examples/sentiment/adv_src.txt \
  --out examples/sentiment/out.txt \
  --adv-out examples/sentiment/adv_out.txt \
  --ref examples/sentiment/ref.txt \
  --s-tgt zero_one \
  --success-threshold 1.8
```

Notice that we specified `--success-threshold 1.8`, which means that an attack will be considered successful only when `s_src(x,x') + d_tgt(y(x),y(x'),y*) > 1.8`. While using `1` as a threshold makes sense when `s_src` and `s_tgt` are the same, when `s_tgt` is the zero-one loss this is a poor choice, as any attack that flips the label will be successful regardless of `s_src`. By upping the threshold to `1.8` we enforce that `s_src` (here chrF) should be at least `0.8` for an attack to be successful.

## Advanced usage

### Custom scorers

TEAPOT comes with predefined scores to compute the source and target side similarity. However, in some cases you might want to define you own score. Fortunately this can be done in a few steps if you are familiar with python:

1. Write your own `teapot.Scorer` subclass in a python source file (there are examples in [examples/custom_scorers.py]()). This is the hard part.
2. Call teapot with the arguments `--custom-scores-source path/to/your/scorer.py` and `--score [score_key]` where `[score_key]` is the shorthand you have defined for your scorer with `teapot.register_scorer` (again, see the examples in [examples/custom_scorers.py]() for a walkthrough)
3. If your scorer works fine, and it doesn't rely on heavy dependencies, consider contributing it to TEAPOT by
    1. Adding the class to [teapot/scorers.py]()
    2. Adding a simple unit test to [tests/test_scorers.py]()

Here is an example where `s_tgt` is the `f1` score defined in [examples/custom_scorers.py]():

```bash
teapot \
  --src examples/MT/src.fr \
  --adv-src examples/MT/adv.charswap.fr \
  --out examples/MT/base.en \
  --adv-out examples/MT/adv.charswap.en \
  --ref examples/MT/ref.en \
  --custom-scores-source examples/custom_scorers.py \
  --s-tgt f1
```

Or when `s_src` is the `constant` score (with auxiliary argument `--value`)

```bash
teapot \
  --src examples/MT/src.fr \
  --adv-src examples/MT/adv.charswap.fr \
  --out examples/MT/base.en \
  --adv-out examples/MT/adv.charswap.en \
  --ref examples/MT/ref.en \
  --custom-scores-source examples/custom_scorers.py \
  --s-src constant \
  --value 0.3
```


### METEOR

You can use the [METEOR](http://www.cs.cmu.edu/~alavie/METEOR/) metric by specifying `--{s-src,s-tgt} meteor`, however this will require you to have java installed and METEOR somewhere on your machine and specify the path to the `.jar` with `--meteor-jar`. This is only tested for METEOR-1.5 on linux.

You can get METEOR by downloading it from the [website](http://www.cs.cmu.edu/~alavie/METEOR/) or in the command line:

```bash
wget http://www.cs.cmu.edu/\\\~alavie/METEOR/download/meteor-1.5.tar.gz
# This will put the jar at ./meteor-1.5/meteor-1.5.jar
tar xvzf meteor-1.5.tar.gz
```

TEAPOT will run the jar from python. You can specify the java command to run the jar with `--java-command` (default `java -Xmx2G -jar`)

### Programmatic Usage

Here is an example of how to use TEAPOT in your own code:

```python
import teapot
# Instantiate the scorer of your choice
chrf_scorer = teapot.ChrF()
bleu_scorer = teapot.BLEU()
meteor_scorer = teapot.METEOR("path/to/meteor.jar", java_command="java -Xmx2G -jar")
# Compute s_src for example
# This will return a list of chrf scores
s_src = chrf_scorer.score(adv_inputs, original_inputs)
# Compute d_tgt
# This will return a list of relative difference in scores (clamped to positive values)
d_tgt = chrf_scorer.rd_score(adv_outputs, original_outputs, reference_outputs)
```

## Reference-less evaluation

In the case where no target reference `y*` is available, one can treat as one the output `y(x)=y*`.
We can then use:

- `s_src(x, x')`: a measure of the semantic similarity between the original input `x` and its adversarial perturbation `x'`.
- `s_tgt(y(x), y(x'))`: a measure of the semantic similarity between the respective outputs.

Setting `y* := y(x)` (we are interested in how well the model deviates from its own predictions), the original criterion for success:
```s_src(x,x') + d_tgt(y(x),y(x'),y*) > 1```
becomes
```s_src(x,x') + 1 - s_tgt(y(x),y(x')) > 1```
which is equivalent to
```s_src(x,x') / s_tgt(y(x),y(x')) > 1.```
The intuition here is slightly different: now, a *successful* attack has caused the system to magnify the source-side adversarial noise.

Simply run the same commands without providing a reference file, for reference-less evaluation.
For example:

```
teapot --src examples/MT/src.fr \
      --adv-src examples/MT/adv.charswap.fr \
      --out examples/MT/base.en \
      --adv-out examples/MT/adv.charswap.en
```
will print
```
No reference file provided. We will use the reference-less criterion.
Source side preservation (ChrF):
Mean: 86.908
Std:  11.622
5%-95%: 64.109-97.683
--------------------------------------------------------------------------------
Target side preservation (ChrF):
Mean: 62.733
Std:  21.529
5%-95%: 16.650-90.555
--------------------------------------------------------------------------------
Success percentage: 97.60 %
```

## License

The code is released under the [MIT License](LICENSE). Credits to [@eseniko](https://giphy.com/eseniko) for the image.


## Citing

If you use this software in your own research, consider citing the following paper:

```
@InProceedings{michel2019onevaluation,
  author    = {Michel, Paul  and  Neubig, Graham and Li, Xian and Pino, Juan Miguel},
  title     = {On Evaluation of Adversarial Perturbations for Sequence-to-Sequence Models},
  year      = {2019},
  booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)}
}
```
