"""Command line: 'deephedge list | run | study | report | figure | change | check'.

    deephedge list                              the catalog of experiments
    deephedge run base/2 [--smoke] [--device cuda] [--pretrained]
    deephedge study random-states cost-1pct/1 [--control]
    deephedge report base/2 [--latex]
    deephedge figure bands base/1               bands, training, histogram or quantiles
    deephedge change base/10 vol-x2/10 --rule corrected-box
    deephedge check base/2                      compare with the published numbers (thesis_tables.json)

Results go to '--workdir' ('runs' by default), one folder per experiment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from .experiments import catalog
from .experiments.pipeline import ExperimentRun
from .report import latex_table, markdown_table

REFERENCE = Path(__file__).resolve().parent / "thesis_tables.json"   # the tables of the thesis, inside the package
STUDIES = ("random-states", "skip-one-date", "levy-check", "grid-convergence", "cross-criteria")
FIGURES = ("bands", "training", "histogram", "quantiles")


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _run(args) -> ExperimentRun:
    if args.threads:
        torch.set_num_threads(args.threads)
    spec = catalog.get(args.experiment, smoke=args.smoke)
    return ExperimentRun(spec, args.workdir, device=args.device, log=_log, pretrained=args.pretrained)


def cmd_list(args) -> int:
    for name, spec in catalog.CATALOG.items():
        dp = " + dynamic programming" if spec.dp else ""
        print(f"{name:20s} {spec.section:4s} {spec.description}{dp}")
    return 0


def cmd_run(args) -> int:
    result = _run(args).run()
    print(markdown_table(result))
    return 0


def cmd_study(args) -> int:
    from .experiments import studies
    run = _run(args)
    if args.study == "random-states":   # fine-tuned and measured on the CPU, as in the thesis
        result = studies.random_states(run, control=args.control, steps=args.steps, device="cpu")
    elif args.study == "skip-one-date":
        result = studies.skip_one_date(run)
    elif args.study == "levy-check":
        result = studies.levy_check(run, device=args.device)
    elif args.study == "grid-convergence":
        result = studies.grid_convergence(run)
    else:
        result = studies.cross_criteria(run)
    print(json.dumps(result, indent=1, default=float)[:4000])
    return 0


def cmd_report(args) -> int:
    result = _run(args).comparison()
    print(latex_table(result, caption=args.experiment, label="tab:" + args.experiment.replace("/", "-"))
          if args.latex else markdown_table(result))
    return 0


def cmd_figure(args) -> int:
    from . import report
    run = _run(args)
    path = Path(args.output or run.dir / f"{args.figure}.png")
    if args.figure == "training":
        run.network()
        if not (run.dir / "training.json").exists():
            print("no training history: the network was taken from the networks of the thesis (--pretrained)")
            return 1
        rules = [r.name for r in run.spec.rules if r.name in run.spec.compare]
        calibration = run.calibration() if rules else {}
        reference = calibration[rules[0]]["value"] if rules else None
        report.training_figure(report.load(run.dir / "training.json"), path, reference, rules[0] if rules else "")
    elif args.figure == "bands":
        if run.spec.n_assets != 1:
            print("the band figure is drawn for single-asset experiments")
            return 1
        strategies = run.strategies()
        keep = {k: strategies[k] for k in ("network", "dp", *(r.name for r in run.spec.rules)) if k in strategies}
        report.band_figure(run.problem, keep, path=path)
    else:
        outcomes = run.outcomes()
        losses = {k: outcomes[k].loss for k in run.spec.compare if k in outcomes}
        (report.histogram_figure if args.figure == "histogram" else report.quantile_figure)(losses, path=path)
    print(path)
    return 0


def cmd_change(args) -> int:
    """How the gap of a rule over the network changes from one experiment to another (thesis, Section 5.4)."""
    from .evaluation import paired_change, series
    pairs = []
    for name in (args.experiment, args.other):
        run = ExperimentRun(catalog.get(name, smoke=args.smoke), args.workdir, device=args.device, log=_log,
                            pretrained=args.pretrained)
        outcomes = run.outcomes()
        pairs.append((series(outcomes["network"], args.metric), series(outcomes[args.rule], args.metric)))
    metric = "mean" if args.metric == "cost" else args.metric
    point, low, high = paired_change(*pairs, metric, reps=50 if args.smoke else 1000, device=args.device)
    print(f"{args.rule} over the network, {args.metric}: {point:+.2f} points [{low:+.2f}; {high:+.2f}] "
          f"from {args.experiment} to {args.other}")
    return 0


def printed_decimals(x: float) -> int:
    """The decimals a published number was printed with (0.288 -> 3, 1.8 -> 1)."""
    text = repr(float(x))
    return 0 if "." not in text or "e" in text else len(text.split(".")[1].rstrip("0"))


def check(result: dict, reference: dict, tolerance: dict) -> list[tuple[str, bool, str]]:
    """Each published number of an experiment against a rerun: the network value of each metric within
    tolerance['value'] (relative), and each difference with the network within tolerance['points']
    percentage points of the published one. Every row and metric of the rerun's table must be published and every
    published one must be in the rerun, so a missing number fails instead of passing unseen. A number that rounds
    to the printed one is marked 'printed digits'."""
    rows = {r["name"]: r for r in result["table"]["rows"]}
    metrics = result["table"]["metrics"]
    versus = reference.get("versus", {})
    out = []
    for name in sorted(set(rows) - {"network"} - set(versus)):
        out.append((name, False, "in the rerun but not in the published table"))
    for name in [n for n in versus if n not in rows]:
        out.append((name, False, "in the published table but not in the rerun"))
    for name in ["network", *[n for n in versus if n in rows]]:
        published_row = reference.get("network", {}) if name == "network" else versus[name]
        for metric in metrics:
            what = f"{name} {metric}"
            if metric not in published_row:
                out.append((what, False, "no published value"))
                continue
            published = published_row[metric]
            if name == "network":
                got = rows[name]["values"][metric]
                d = printed_decimals(published)
                ok = abs(got / published - 1) <= tolerance["value"]
                exact = round(got, d) == round(published, d)
                detail = f"{got:.{d + 2}f} vs {published}"
            else:
                got = rows[name]["versus_reference"][metric]
                d = max(printed_decimals(v) for v in published)
                ok = abs(got[0] - published[0]) <= tolerance["points"]
                exact = round(got[0], d) == round(published[0], d)
                detail = (f"{got[0]:+.{d + 1}f}% [{got[1]:+.{d + 1}f}; {got[2]:+.{d + 1}f}] vs "
                          f"{published[0]:+}% [{published[1]:+}; {published[2]:+}]")
            out.append((what, ok, detail + ("  (printed digits)" if exact else "")))
    return out


def cmd_check(args) -> int:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if args.experiment not in reference["experiments"]:
        print(f"no published numbers for {args.experiment}")
        return 1
    result = _run(args).comparison()
    entry = reference["experiments"][args.experiment]
    tolerance = {**reference["tolerance"], **entry.get("tolerance", {})}
    lines = check(result, entry, tolerance)
    for what, ok, detail in lines:
        print(f"{'ok ' if ok else 'FAIL'} {what:28s} {detail}")
    return 0 if all(ok for _, ok, _ in lines) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deephedge", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the catalog of experiments").set_defaults(func=cmd_list)

    def experiment_parser(name: str, help_: str):
        p = sub.add_parser(name, help=help_)
        if name == "study":
            p.add_argument("study", choices=STUDIES)
        if name == "figure":
            p.add_argument("figure", choices=FIGURES)
        p.add_argument("experiment", choices=sorted(catalog.CATALOG), metavar="EXPERIMENT")
        p.add_argument("--workdir", default="runs")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--smoke", action="store_true", help="tiny version that runs in seconds")
        p.add_argument("--pretrained", action="store_true", help="use the networks of the thesis instead of training")
        p.add_argument("--threads", type=int, default=0)
        return p

    experiment_parser("run", "run every stage of an experiment").set_defaults(func=cmd_run)
    p = experiment_parser("study", "a study on top of an experiment")
    p.add_argument("--control", action="store_true")
    p.add_argument("--steps", type=int, default=3000)
    p.set_defaults(func=cmd_study)
    p = experiment_parser("report", "the table of an experiment")
    p.add_argument("--latex", action="store_true")
    p.set_defaults(func=cmd_report)
    p = experiment_parser("figure", "a figure of an experiment")
    p.add_argument("--output", default="")
    p.set_defaults(func=cmd_figure)
    p = experiment_parser("change", "change of the gap of a rule over the network between two experiments")
    p.add_argument("other", choices=sorted(catalog.CATALOG), metavar="OTHER")
    p.add_argument("--rule", default="corrected-box")
    p.add_argument("--metric", default="entropic")
    p.set_defaults(func=cmd_change)
    experiment_parser("check", "compare a run with the published numbers").set_defaults(func=cmd_check)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
