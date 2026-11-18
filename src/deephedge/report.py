"""Tables and basic figures from the results of the pipeline.

The tables follow the format of the thesis: the network row shows the value of each metric, every other
row its difference with the network in % of the network's value, with the 95% paired interval below, and
the bold rule of :class:`deephedge.evaluation.Comparison`. Differences get two decimals when every number
is below 1 in absolute value and one otherwise, plus one more while the interval is narrower than two
units of the last decimal (the mean cost has very narrow paired intervals).
"""
from __future__ import annotations

import json
from pathlib import Path

LABELS = {"network": "Network", "dp": "Dynamic programming", "delta": "Delta", "whalley-wilmott": "Whalley-Wilmott",
          "corrected-box": "Corrected box", "atkinson-box": "Atkinson box", "shifted-centre": "Shifted centre",
          "corrected-box/calibrated-centre": "Corrected box, calibrated centre"}
METRIC_LABELS = {"entropic": "Entropic risk", "cost": "Mean cost", "std": "Std. dev.", "mse": "Mean square",
                 "cvar95": "CVaR 95%", "cvar99": "CVaR 99%", "cvar999": "CVaR 99.9%", "mean": "Mean loss"}
def value_decimals(metric: str, metrics: list[str]) -> int:
    """Decimals of the network row: four for the small values of the tables without costs (mean square and
    entropic risk, as in the thesis), three otherwise."""
    return 4 if metric == "mse" or (metric == "entropic" and "mse" in metrics) else 3


def decimals(t: list[float] | tuple[float, ...]) -> int:
    d = 2 if max(abs(x) for x in t[:3]) < 1 else 1
    while d < 3 and t[2] - t[1] < 2 * 10 ** -d:
        d += 1
    return d


def _num(v: float, d: int, comma: bool) -> str:
    s = f"{abs(v):.{d}f}"
    sign = "-" if v < 0 and s.strip("0.") else ""
    s = sign + s
    return s.replace(".", ",") if comma else s


def _diff(t, comma: bool) -> tuple[str, str]:
    d = decimals(t)
    s = f"{abs(t[0]):.{d}f}"
    if s.strip("0."):
        s = ("-" if t[0] < 0 else "+") + s
    interval = f"[{_num(t[1], d, comma)}; {_num(t[2], d, comma)}]" if comma else \
        f"[{_num(t[1], d, comma)}, {_num(t[2], d, comma)}]"
    return (s.replace(".", ",") if comma else s) + "%", interval


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def markdown_table(comparison: dict) -> str:
    table = comparison["table"]
    metrics = table["metrics"]
    lines = ["| Strategy | " + " | ".join(METRIC_LABELS.get(m, m) for m in metrics) + " |",
             "|---|" + "---|" * len(metrics)]
    for row in table["rows"]:
        cells = []
        for m in metrics:
            if row["versus_reference"] is None:
                text = _num(row["values"][m], value_decimals(m, metrics), False)
            else:
                a, b = _diff(row["versus_reference"][m], False)
                text = f"{a} {b}"
            cells.append(f"**{text}**" if m in row["bold"] else text)
        lines.append(f"| {LABELS.get(row['name'], row['name'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latex_table(comparison: dict, caption: str = "", label: str = "", comma: bool = True) -> str:
    table = comparison["table"]
    metrics = table["metrics"]
    rows = []
    for row in table["rows"]:
        cells = [LABELS.get(row["name"], row["name"])]
        for m in metrics:
            bold = m in row["bold"]
            if row["versus_reference"] is None:
                s = _num(row["values"][m], value_decimals(m, metrics), comma).replace("-", "$-$")
                cells.append(f"\\textbf{{{s}}}" if bold else s)
            else:
                a, b = _diff(row["versus_reference"][m], comma)
                a, b = a.replace("-", "$-$").replace("%", "\\%"), b.replace("-", "$-$")
                if bold:
                    a = f"\\textbf{{\\boldmath {a}}}"
                cells.append(f"\\makecell[t]{{{a}\\\\{{\\scriptsize {b}}}}}")
        rows.append(" & ".join(cells) + " \\\\")
    # the labels of the CVaR columns carry a % sign, which starts a comment in LaTeX
    head = " & ".join(["Strategy"] + [METRIC_LABELS.get(m, m).replace("%", "\\%") for m in metrics]) + " \\\\"
    return "\n".join(["\\begin{table}[!htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                      "\\small", f"\\begin{{tabular}}{{l{'c' * len(metrics)}}}", "\\toprule", head, "\\midrule",
                      *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def band_figure(problem, strategies: dict, dates: tuple[int, ...] = (25, 45), path: str | Path = "bands.png",
                prices: tuple[float, float] = (80.0, 120.0)) -> Path:
    """No-trade bands of single-asset strategies against the price, at the given dates, for a holding at the
    middle of the band (the band of a single-asset strategy does not depend on the holding)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    grid = torch.linspace(*prices, 401, dtype=torch.float64)[:, None]
    fig, axes = plt.subplots(1, len(dates), figsize=(4.0 * len(dates), 3.6), sharey=True)
    axes = [axes] if len(dates) == 1 else axes
    for ax, t in zip(axes, dates, strict=True):
        for name, s in strategies.items():
            lower, upper = s.band(t, grid, torch.full_like(grid, 0.5))
            line = ax.plot(grid[:, 0], lower[:, 0].detach().clamp(0, 1), lw=1.4, label=LABELS.get(name, name))[0]
            ax.plot(grid[:, 0], upper[:, 0].detach().clamp(0, 1), lw=1.4, color=line.get_color())
        ax.set_title(f"{problem.market.dates - t} days to maturity")
        ax.set_xlabel("Price")
    axes[0].set_ylabel("Holding")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return Path(path)


def histogram_figure(losses: dict, path: str | Path = "histogram.png", bins: int = 120) -> Path:
    """Distribution of the loss of each strategy on the test sample (thesis, Figure 17)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    every = torch.cat([v for v in losses.values()])
    low, high = (float(q) for q in torch.quantile(every[::10], torch.tensor([0.0005, 0.9995], dtype=every.dtype)))
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for name, loss in losses.items():
        ax.hist(loss.clamp(low, high).numpy(), bins=bins, range=(low, high), histtype="step", lw=1.3, density=True,
                label=LABELS.get(name, name))
    ax.set_xlabel("Loss")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return Path(path)


def quantile_figure(losses: dict, reference: str = "network", path: str | Path = "quantiles.png") -> Path:
    """Difference between each percentile of the loss of every strategy and the same percentile of the loss of the
    reference, from the 1st to the 99.9th (thesis, Figure 18)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    levels = torch.cat([torch.arange(1, 100, dtype=torch.float64), torch.tensor([99.5, 99.9], dtype=torch.float64)])
    ref = torch.quantile(losses[reference], levels / 100)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for name, loss in losses.items():
        if name != reference:
            ax.plot(levels, torch.quantile(loss, levels / 100) - ref, lw=1.4, label=LABELS.get(name, name))
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("Percentile")
    ax.set_ylabel(f"Difference with the {LABELS.get(reference, reference).lower()}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return Path(path)


def training_figure(training: dict, path: str | Path = "training.png", reference: float | None = None,
                    reference_label: str = "") -> Path:
    """Validation risk along the training (thesis, Figure 14), raw or relative to 'reference'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [h["step"] for h in training["history"]]
    values = [sum(h["validation"]) for h in training["history"]]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    if reference is None:
        ax.plot(steps, values, lw=1.4)
        ax.set_ylabel("Validation risk")
    else:   # as in the thesis, in % above or below the calibrated rule on the same validation paths
        ax.plot(steps, [100 * (v / reference - 1) for v in values], lw=1.4)
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_ylabel(f"% over {LABELS.get(reference_label, reference_label)}")
    ax.axvline(training["best_step"], color="0.6", ls=":", lw=1)
    ax.set_xlabel("Training step")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return Path(path)
