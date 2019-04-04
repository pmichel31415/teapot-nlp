import os.path
import argparse
from teapot import scorers
from teapot import utils


def get_args():
    parser = argparse.ArgumentParser("TEAPOT", conflict_handler="resolve")
    parser.add_argument(
        "--s-src",
        default="chrf",
        type=str,
        help="Score to evaluate similarity in the source."
        f"(choose from: {', '.join(list(scorers.scorers.keys()))})",
    )
    parser.add_argument(
        "--s-tgt",
        default="chrf",
        type=str,
        help="Score to evaluate similarity in the target."
        f"(choose from: {', '.join(list(scorers.scorers.keys()))})",
    )
    parser.add_argument(
        "--src",
        default=None,
        type=str,
        help="Original source",
    )
    parser.add_argument(
        "--adv-src",
        default=None,
        type=str,
        help="Adversarial perturbation of the source",
    )
    parser.add_argument(
        "--ref",
        default=None,
        type=str,
        help="Reference output",
    )
    parser.add_argument(
        "--out",
        default=None,
        type=str,
        help="Model output on the original source",
    )
    parser.add_argument(
        "--adv-out",
        default=None,
        type=str,
        help="Model output on the adversarial source",
    )
    parser.add_argument(
        "--src-lang",
        default=None,
        type=str,
        help="Source language. This is mostly used for Meteor, in which case"
        " choose one from: en, cz, de, es, fr, ar, da, fi, hu, it, nl, no,"
        " pt, ro, ru, se, tr",
    )
    parser.add_argument(
        "--tgt-lang",
        default=None,
        type=str,
        help="Target language. This is mostly used for Meteor, in which case"
        " choose one from: en, cz, de, es, fr, ar, da, fi, hu, it, nl, no,"
        " pt, ro, ru, se, tr",
    )
    parser.add_argument(
        "--scale",
        default=100,
        type=float,
        help="Scale for the scores.",
    )
    parser.add_argument(
        "--success-threshold",
        default=1.0,
        type=float,
        help="Threshold that the value of s_src + d_tgt must surpass to "
        "consider the attack successful. When s_src and s_tgt are the same "
        "this should be 1, however in other cases a higher value might be "
        "preferable (eg. when s_tgt is more 'optimistic' than s_src)."
        "For evaluation without references, this is the value that s_src/d_tgt"
        "must surpass to consider the attack successful.",
    )
    parser.add_argument(
        "--terse",
        action="store_true",
        help="Only output average scores, one on each line "
        "(for use in bash scripts)"
    )
    parser.add_argument(
        "--custom-scores-source",
        nargs="*",
        type=str,
        default=[],
        help="Path to python files containing custom scorers implementation"
    )

    args, _ = parser.parse_known_args()
    # Check arguments
    source_side = args.src is not None and args.adv_src is not None
    target_side = (
        args.out is not None and
        args.adv_out is not None
    )
    if not (source_side or target_side):
        raise ValueError(
            "You need to specify at least `--src` and `--adv-src` "
            "(for source side evaluation) OR `--out` and `--adv-out` "
            " (for target side evaluation)."
        )
    with_references = False
    if args.ref is not None:
        with_references = True
    # Check file existence
    if with_references:
        file_check_list = ["src", "adv_src", "out", "adv_out"]
    else:
        file_check_list = ["src", "adv_src", "ref", "out", "adv_out"]
    for name in file_check_list:
        filename = getattr(args, name, None)
        if filename is not None and not os.path.isfile(filename):
            raise ValueError(
                f"Specified file for \"{name}\" (\"{filename}\")"
                " does not exist"
            )
    # Load custom scorers source
    for source_file in args.custom_scores_source:
        path = os.path.abspath(source_file)
        if not os.path.isfile(path):
            raise ValueError(
                f"Can't find custom scorer source file \"{path}\""
            )
        scorers.read_custom_scorers_source(path)
    # Add scorer specific args
    scorer_src_class = scorers.get_scorer_class(args.s_src)
    scorer_src_class.add_args(parser)
    scorer_tgt_class = scorers.get_scorer_class(args.s_tgt)
    scorer_tgt_class.add_args(parser)
    # Parse again with scorer specific args
    args = parser.parse_args()
    return args, source_side, target_side, with_references


def main():
    # Command line args
    args, source_side, target_side, with_references = get_args()
    scale = args.scale
    if not with_references:
        print(
            "Note: No reference file provided. Will use the "
            "reference-less criterion."
        )
    # Scorer
    scorer_src, scorer_tgt = scorers.scorers_from_args(args)
    # Source side eval
    N = None
    if source_side:
        # Source score (s_src in the paper)
        s_src = scorer_src.score(
            utils.loadtxt(args.adv_src),
            utils.loadtxt(args.src),
            lang=args.src_lang,
        )
        # statistics
        N = len(s_src)
        s_src_avg, s_src_std, s_src_5, s_src_95 = utils.stats(s_src)
        # Print stats
        if args.terse:
            print(f"{s_src_avg*scale:.3f}")
        else:
            print(f"Source side preservation ({scorer_src.name}):")
            print(f"Mean:\t{s_src_avg*scale:.3f}")
            print(f"Std:\t{s_src_std*scale:.3f}")
            print(f"5%-95%:\t{s_src_5*scale:.3f}-{s_src_95*scale:.3f}")

    # Target side eval with references
    if target_side and with_references:
        # target relative decrease in score (d_tgt in the paper)
        d_tgt = scorer_tgt.rd_score(
            utils.loadtxt(args.adv_out),
            utils.loadtxt(args.out),
            utils.loadtxt(args.ref),
            lang=args.tgt_lang,
        )
        # Check size
        if N is None:
            N = len(d_tgt)
        elif len(d_tgt) != N:
            raise ValueError(
                f"The number of samples in the source ({N}) doesn't match "
                f"the number of samples in the target ({len(d_tgt)})"
            )
        # Statistics
        d_tgt_avg, d_tgt_std, d_tgt_5, d_tgt_95 = utils.stats(d_tgt)
        # Print stats
        if args.terse:
            print(f"{d_tgt_avg*scale:.3f}")
        else:
            if source_side:
                print("-" * 80)
            print(
                "Target side degradation "
                f"(relative decrease in {scorer_tgt.name}):"
            )
            print(f"Mean:\t{d_tgt_avg*scale:.3f}")
            print(f"Std:\t{d_tgt_std*scale:.3f}")
            print(f"5%-95%:\t{d_tgt_5*scale:.3f}-{d_tgt_95*scale:.3f}")
    # Both sided (success) with references
    if target_side and source_side and with_references:
        success = [
            float(s + d > args.success_threshold)
            for s, d in zip(s_src, d_tgt)
        ]
        success_fraction = sum(success) / N
        # Print success
        if args.terse:
            print(f"{success_fraction*100:.3f}")
        else:
            print("-" * 80)
            print(f"Success percentage: {success_fraction*100:.2f} %")

    # Target side eval without references
    if target_side and not with_references:
        # target relative decrease in score (d_tgt in the paper)
        d_tgt = scorer_tgt.score(
            utils.loadtxt(args.adv_out),
            utils.loadtxt(args.out),
            lang=args.tgt_lang,
        )
        # Check size
        if N is None:
            N = len(d_tgt)
        elif len(d_tgt) != N:
            raise ValueError(
                f"The number of samples in the source ({N}) doesn't match "
                f"the number of samples in the target ({len(d_tgt)})"
            )
        # Statistics
        d_tgt_avg, d_tgt_std, d_tgt_5, d_tgt_95 = utils.stats(d_tgt)
        # Print stats
        if args.terse:
            print(f"{d_tgt_avg*scale:.3f}")
        else:
            if source_side:
                print("-" * 80)
            print(
                f"Target side preservation ({scorer_tgt.name}):"
            )
            print(f"Mean:\t{d_tgt_avg*scale:.3f}")
            print(f"Std:\t{d_tgt_std*scale:.3f}")
            print(f"5%-95%:\t{d_tgt_5*scale:.3f}-{d_tgt_95*scale:.3f}")

    # Both sided (success) and without references
    if target_side and source_side and not with_references:
        success = [
            float(s/d > args.success_threshold)
            for s, d in zip(s_src, d_tgt)
        ]
        success_fraction = sum(success) / N
        # Print success
        if args.terse:
            print(f"{success_fraction*100:.3f}")
        else:
            print("-" * 80)
            print(f"Success percentage: {success_fraction*100:.2f} %")


if __name__ == "__main__":
    main()
