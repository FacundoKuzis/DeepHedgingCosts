"""One pipeline for every experiment (thesis, Sections 4.1-4.4).

'ExperimentRun(spec, workdir)' runs the stages of an experiment and keeps each result in 'workdir':

    problem ──► network.pt, training.json      Trainer.fit on fresh paths, best validation step
            ──► dp/                            dynamic programme (one and two assets)
            ──► calibration.json               every rule calibrated on the validation sample
            ──► outcomes.pt                    loss and cost of every strategy on the 200,000 test paths
            ──► comparison.json                paired bootstrap against the network, bold rule

A stage whose file exists is not run again, so an interrupted run resumes where it stopped (the
training itself resumes from its checkpoint). The test sample is only used by the last two stages,
after every choice (network checkpoint, calibrated scales) has been frozen.

Results are only reused by the run that made them. 'run.json' records the fingerprint of the spec and where
the network comes from (trained here, or one of the networks of the thesis with 'pretrained', with its
SHA-256), and a run whose spec or origin differs from the record refuses the folder instead of mixing results.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import torch

from .. import calibration as cal
from ..dp import OneAssetDP, OneAssetGrid, OptimalBand, OptimalGridStrategy, TwoAssetDP
from ..evaluation import WITH_COSTS, WITHOUT_COSTS, Comparison, summarize
from ..hedging import Batch, HedgeOutcome, HedgingProblem
from ..pricing import frictionless_greeks
from ..risk import risk_measure
from ..strategies import DeltaHedge, HedgingStrategy, make_rule
from ..strategies.neural import neural_strategy
from ..training import StandardStarts, TrainConfig, Trainer
from . import catalog
from .spec import ExperimentSpec

SEARCHES = {"single-asset": cal.SINGLE_ASSET, "box": cal.BOX, "corrected-box": cal.CORRECTED_BOX, "cvar": cal.CVAR,
            "smoke": cal.SMOKE}
CENTRE_SEARCHES = ("centre", "local-centre")   # the shifted centre of Section 5.5 and the control of Section 5.2
PRETRAINED = Path(__file__).resolve().parents[1] / "pretrained"   # the networks of the thesis, inside the package


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(spec: ExperimentSpec) -> str:
    """What the results of an experiment depend on: every field of its spec but the descriptive ones."""
    return hashlib.sha256(repr(replace(spec, section="", description="")).encode()).hexdigest()


def solve_dp(problem: HedgingProblem, grid, risk_aversion: float, workdir: str | Path, *, device: str = "cpu",
             log: Callable[[str], None] = print) -> HedgingStrategy:
    """The dynamic programme of 'problem' on 'grid', solved once and kept in 'workdir': the band with one asset
    (a small file), the tables of every date with two ('50 * grid.table_bytes' on disk)."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if isinstance(grid, OneAssetGrid):
        path = workdir / f"one-asset_{grid.n_prices}x{grid.n_positions}.pt"
        if path.exists():
            return OptimalBand.from_state_dict(torch.load(path, weights_only=True))
        band = OneAssetDP(problem, risk_aversion, grid).solve(device)
        torch.save(band.state_dict(), path)
        log(f"dynamic programme {grid.n_prices} x {grid.n_positions} in {band.seconds:.0f} s, "
            f"predicted risk {band.predicted_risk():.6f}")
        return band
    table_dir = workdir / f"two-assets_{grid.n_prices}x{grid.n_positions}"
    if (table_dir / "meta.json").exists():
        return OptimalGridStrategy.load(table_dir)
    log(f"dynamic programme {grid.n_prices}^2 x {grid.n_positions}^2 on {device}: "
        f"{problem.market.dates * grid.table_bytes / 1e9:.1f} GB of tables in {table_dir}")
    strategy = TwoAssetDP(problem, risk_aversion, grid, table_dir).solve(device)
    log(f"dynamic programme solved in {strategy.seconds / 60:.1f} min, predicted risk {strategy.predicted_risk():.6f}")
    return strategy


class ExperimentRun:
    def __init__(self, spec: ExperimentSpec, workdir: str | Path = "runs", *, device: str = "cpu",
                 log: Callable[[str], None] | None = None, pretrained: bool = False, reps: int = 1000):
        self.spec = spec
        self.dir = Path(workdir) / spec.name.replace("/", "_")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.log = log or print
        self.pretrained = pretrained
        self.reps = 50 if "smoke" in spec.tags else reps
        self._problem: HedgingProblem | None = None
        self._paths: dict[str, torch.Tensor] = {}
        self._check_record()

    # --- what the folder holds ---------------------------------------------------------------------------
    @property
    def origin(self) -> str:
        return "pretrained" if self.pretrained else "trained"

    def _check_record(self) -> None:
        """Write 'run.json' in a new folder, or refuse a folder made with another spec or another network."""
        path = self.dir / "run.json"
        if not path.exists():
            found = sorted(p.name for p in self.dir.iterdir())
            if found:
                raise RuntimeError(f"{self.dir} holds results without a record of how they were made "
                                   f"({', '.join(found[:4])}); delete them or choose another --workdir")
            _write(path, {"experiment": self.spec.name, "spec": fingerprint(self.spec), "network": self.origin})
            return
        record = _read(path)
        if record["spec"] != fingerprint(self.spec):
            raise RuntimeError(f"{self.dir} holds the results of another specification of {self.spec.name} (the "
                               "catalog changed since they were made); delete them or choose another --workdir")
        if record["network"] != self.origin:
            hint = "drop --pretrained" if self.pretrained else "add --pretrained"
            raise RuntimeError(f"{self.dir} holds results with a {record['network']} network and this run asks for a "
                               f"{self.origin} one; {hint} to reuse them, or choose another --workdir")

    # --- data --------------------------------------------------------------------------------------------
    @property
    def problem(self) -> HedgingProblem:
        if self._problem is None:
            self._problem = self.spec.problem()
        return self._problem

    def paths(self, split: str) -> torch.Tensor:
        """The validation or test sample, float64 on the CPU (regenerated from its seed, never stored)."""
        if split not in self._paths:
            sample = self.spec.validation if split == "validation" else self.spec.test
            self._paths[split] = sample.draw(self.problem.market)
        return self._paths[split]

    # --- network -----------------------------------------------------------------------------------------
    def new_network(self) -> HedgingStrategy:
        p = self.problem
        return neural_strategy(p.market, p.option, with_costs=self.spec.cost > 0, seed=self.spec.network.weight_seed)

    def network(self) -> HedgingStrategy:
        """The trained network of the experiment or, with 'pretrained', the network of the thesis."""
        path = self.dir / "network.pt"
        strategy = self.new_network()
        if self.pretrained:
            source = PRETRAINED / (self.spec.name.replace("/", "_") + ".pt")
            if not source.exists():
                raise FileNotFoundError(f"there is no network of the thesis for {self.spec.name} ({source} does not "
                                        "exist); run it without --pretrained to train one")
            record = _read(self.dir / "run.json")
            if "pretrained_sha256" not in record:
                record["pretrained_sha256"] = _sha256(source)
                _write(self.dir / "run.json", record)
            elif record["pretrained_sha256"] != _sha256(source):
                raise RuntimeError(f"{self.dir} was evaluated with another version of {source.name}; delete it or "
                                   "choose another --workdir")
            if not path.exists():
                strategy.network.load_state_dict(torch.load(source, map_location="cpu", weights_only=True))
                torch.save(strategy.network.state_dict(), path)
                self.log(f"{self.spec.name}: network taken from {source.name}")
        elif not path.exists():
            self.train(strategy)
        strategy.network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        strategy.network.to(self.device).eval()
        return strategy

    def train(self, strategy: HedgingStrategy) -> None:
        n = self.spec.network
        config = TrainConfig(steps=n.steps, batch_size=n.batch_size, lr=n.lr, lr_final=n.lr_final, warmup=n.warmup,
                             clip_norm=n.clip_norm, validate_every=n.validate_every, batch_seed=n.batch_seed)
        trainer = Trainer(self.problem, risk_measure(n.criterion, self.spec.risk_aversion), config,
                          device=self.device, log=lambda m: self.log(f"{self.spec.name}: {m}"))
        validation = Batch(self.paths("validation").to(self.device, torch.float32))
        source = StandardStarts(self.problem, device=self.device, dtype=torch.float32)
        result = trainer.fit(strategy, source, validation, checkpoint=self.dir / "checkpoint.pt")
        torch.save(strategy.network.cpu().state_dict(), self.dir / "network.pt")
        result.save(self.dir / "training.json")
        (self.dir / "checkpoint.pt").unlink(missing_ok=True)

    # --- dynamic programme -------------------------------------------------------------------------------
    def dp(self) -> HedgingStrategy | None:
        if not self.spec.dp:
            return None
        return solve_dp(self.problem, self.spec.dp[-1], self.spec.risk_aversion, self.dir / "dp", device=self.device,
                        log=lambda m: self.log(f"{self.spec.name}: {m}"))

    # --- rules -------------------------------------------------------------------------------------------
    def rule(self, kind: str, scale: float, centre: float = 1.0) -> HedgingStrategy:
        p = self.problem
        return make_rule(kind, p.market, p.option, p.cost, scale, centre)

    def validation_risk(self, strategy: HedgingStrategy, criterion: str) -> float:
        outcome = self.problem.evaluate(strategy, self.paths("validation"), device=self.device)
        return float(risk_measure(criterion, self.spec.risk_aversion)(outcome.loss))

    def calibration(self, rules: tuple[str, ...] | None = None) -> dict:
        """The calibrated scale (and centre) of every rule of the spec, or of the 'rules' named, chosen on the
        validation sample."""
        path = self.dir / "calibration.json"
        state = _read(path) if path.exists() else {}
        smoke = "smoke" in self.spec.tags
        for r in self.spec.rules:
            if (rules is not None and r.name not in rules) or state.get(r.name, {}).get("done"):
                continue
            memo = state.setdefault(r.name, {"evaluations": []})     # [g, c, risk], kept across interruptions
            done = {(g, c): v for g, c, v in memo["evaluations"]}

            def evaluate(g: float, c: float = 1.0, r=r, memo=memo, done=done) -> float:
                if (g, c) not in done:
                    t0 = time.time()
                    done[(g, c)] = self.validation_risk(self.rule(r.kind, g, c), r.criterion)
                    memo["evaluations"].append([g, c, done[(g, c)]])
                    _write(path, state)
                    self.log(f"{self.spec.name}: {r.name} g={g:g} c={c:g} -> {done[(g, c)]:.6f} "
                             f"({time.time() - t0:.0f} s)")
                return done[(g, c)]

            if r.search in CENTRE_SEARCHES:   # the centre and the scale together, from the rule with c = 1
                base = next(b for b in self.spec.rules if b.kind == r.kind and b.search not in CENTRE_SEARCHES)
                b = state[base.name]
                start = cal.Calibration(b["scale"], b["value"], b["at_edge"])
                if r.search == "centre":
                    result = cal.calibrate_centre_and_scale(evaluate, start, cal.CENTRE_SMOKE if smoke else cal.CENTRE)
                else:
                    result = cal.refine_centre_and_scale(evaluate, start,
                                                         cal.LOCAL_CENTRE_SMOKE if smoke else cal.LOCAL_CENTRE)
            else:
                result = cal.calibrate_scale(evaluate, cal.SMOKE if smoke else SEARCHES[r.search])
            state[r.name] = {**result.as_dict(), "kind": r.kind, "criterion": r.criterion, "done": True,
                             "evaluations": memo["evaluations"]}
            _write(path, state)
        return state if rules is None else {k: v for k, v in state.items() if k in rules}

    # --- every strategy on the test sample ---------------------------------------------------------------
    def strategies(self) -> dict[str, HedgingStrategy]:
        p, spec = self.problem, self.spec
        out: dict[str, HedgingStrategy] = {"network": self.network()}
        if "uses:base" in spec.tags:      # the CVaR experiments also show the entropic network and rules
            base_name = spec.name.removesuffix("/smoke").replace("cvar/", "base/")
            base = ExperimentRun(catalog.get(base_name, smoke="smoke" in spec.tags),
                                 self.dir.parent, device=self.device, log=self.log, pretrained=self.pretrained)
            out["network/entropic"] = base.network()
            centre_at_delta = tuple(r.name for r in base.spec.rules if r.search not in CENTRE_SEARCHES)
            for r, c in base.calibration(centre_at_delta).items():
                out[f"{r}/entropic"] = self.rule(c["kind"], c["scale"])
        dp = self.dp()
        if dp is not None:
            out["dp"] = dp
        for name, c in self.calibration().items():
            out[name] = self.rule(c["kind"], c["scale"], c.get("centre", 1.0))
            if spec.theoretical_rule and c["criterion"] == "entropic" and "centre" not in name:
                out[f"{name}/theoretical"] = self.rule(c["kind"], spec.risk_aversion)
        out["delta"] = DeltaHedge(frictionless_greeks(p.market, p.option))
        return out

    def outcomes(self) -> dict[str, HedgeOutcome]:
        path = self.dir / "outcomes.pt"
        stored = torch.load(path, weights_only=True) if path.exists() else {}
        strategies = self.strategies()
        missing = [k for k in strategies if k not in stored]
        if missing:
            test = self.paths("test")
            for k in missing:
                t0 = time.time()
                o = self.problem.evaluate(strategies[k], test, device=self.device)
                stored[k] = {"loss": o.loss, "cost": o.cost, "turnover": o.turnover}
                torch.save(stored, path)
                self.log(f"{self.spec.name}: {k} on the test sample ({time.time() - t0:.0f} s)")
        return {k: HedgeOutcome(v["loss"], v["cost"], v["turnover"]) for k, v in stored.items()}

    def comparison(self) -> dict:
        path = self.dir / "comparison.json"
        if path.exists():
            return _read(path)
        outcomes = self.outcomes()
        rows = [k for k in self.spec.compare if k in outcomes]
        metrics = WITHOUT_COSTS if self.spec.cost == 0 else WITH_COSTS
        table = Comparison({k: outcomes[k] for k in rows}, "network", metrics,
                           risk_aversion=self.spec.risk_aversion, reps=self.reps, device=self.device)
        result = {"experiment": self.spec.name, "table": table.as_dict(), "ties": table.ties,
                  "summary": {k: summarize(o, self.spec.risk_aversion) for k, o in outcomes.items()}}
        _write(path, result)
        return _read(path)      # the same object whether the stage ran now or before

    def run(self) -> dict:
        t0 = time.time()
        result = self.comparison()
        self.log(f"{self.spec.name}: done in {(time.time() - t0) / 60:.1f} min")
        return result
