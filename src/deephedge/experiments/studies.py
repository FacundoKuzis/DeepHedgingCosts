"""The studies that sit on top of an experiment.

* :func:`random_states` (Sections 5.2.1 and 5.3): fine-tune the single-asset network on paths that start
  at a random date, price and holding, and compare it with the original network and the dynamic programme,
  on the test sample and from states the original network never visits. 'control=True' runs the same
  fine-tuning on the usual paths only, which separates the effect of the new states from that of training
  longer.
* :func:`risk_from_state`: entropic risk of what is left of the hedge from one fixed state.
* :func:`skip_one_date` (Section 5.1): a scale for the mean-square gap between the network and the delta
  without costs, the increase from skipping one rebalancing date or rebalancing every 2 or 5 days.
* :func:`levy_check` (Section 4.2.1): the Levy delta against the local-risk-minimising hedge computed by
  nested Monte Carlo at random test states.
* :func:`grid_convergence` (Sections 2.9 and 5.2): the two-asset dynamic programme on each grid of the experiment.
* :func:`cross_criteria` (Section 5.5): the networks and rules of the entropic and CVaR experiments, each
  measured with both criteria.
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import torch

from ..evaluation import paired_bootstrap
from ..hedging import Batch, HedgeOutcome
from ..pricing import LevyGreeks, frictionless_greeks
from ..risk import Entropic, MeanSquare, risk_measure
from ..training import RandomStarts, StandardStarts, TrainConfig, Trainer
from .pipeline import ExperimentRun, _write, solve_dp

# states (date, price, holding) the network of each cost never visits (Section 5.2.1)
RANDOM_STATE_PROBES = {0.01: ((45, 80.0, 0.5), (45, 80.0, 0.3), (45, 115.0, 0.3), (40, 80.0, 0.5)),
                       0.001: ((45, 115.0, 0.7), (45, 120.0, 0.5), (45, 80.0, 0.5), (40, 115.0, 0.7))}


@torch.no_grad()
def risk_from_state(problem, strategy, date: int, price: float, holding: float, *, n_paths: int = 200_000,
                    seed: int = 7, risk_aversion: float = 0.5) -> tuple[float, float]:
    """Entropic risk of payoff - gains + costs from (date, price, holding) to maturity, and the mean final
    holding. One asset; the shocks come from a CPU generator seeded with 'seed'."""
    m = problem.market
    z = m.sample_shocks((n_paths, m.dates - date), torch.Generator().manual_seed(seed))
    log_paths = torch.cat([torch.zeros(n_paths, 1, dtype=torch.float64),
                           torch.cumsum(m.drifts.item() + m.step_vols.item() * z, 1)], 1)
    paths = (price * torch.exp(log_paths))[..., None]
    position = torch.full((n_paths, 1), holding, dtype=torch.float64)
    gains = torch.zeros(n_paths, dtype=torch.float64)
    costs = torch.zeros(n_paths, dtype=torch.float64)
    for j, t in enumerate(range(date, m.dates)):
        new = strategy.act(t, paths[:, j], position).double()
        costs += problem.trading_cost(paths[:, j], new - position)
        gains += (new * (paths[:, j + 1] - paths[:, j])).sum(-1)
        position = new
    costs += problem.trading_cost(paths[:, -1], position)
    losses = problem.option.payoff(paths[:, -1]) - gains + costs
    return float(Entropic(risk_aversion)(losses)), float(position.mean())


def random_states(run: ExperimentRun, *, control: bool = False, steps: int = 3000, device: str = "cpu") -> dict:
    """Fine-tune the network of a single-asset experiment from random states (thesis protocol: 3,000 steps,
    learning rate 3e-4 to 1e-5, warm-up 50, 4,096 usual paths plus 4,096 random states per step, validation
    every 100 steps on the usual validation sample plus 65,536 random states, the starting network included).
    The whole study runs on 'device', the fine-tuning and the measurements; the thesis ran it on the CPU."""
    spec, problem = run.spec, run.problem
    out_dir = run.dir / ("random-states-control" if control else "random-states")
    out_dir.mkdir(exist_ok=True)
    original = run.network()
    original.network.to(device)
    tuned = run.new_network()
    tuned.network.load_state_dict(original.network.state_dict())
    smoke = "smoke" in spec.tags
    config = TrainConfig(steps=20 if smoke else steps, batch_size=256 if smoke else 4096, lr=3e-4, lr_final=1e-5,
                         warmup=5 if smoke else 50, clip_norm=1.0, validate_every=5 if smoke else 100,
                         batch_seed=20261101, validate_at_start=True)
    starts = RandomStarts(problem, device=device, dtype=torch.float32)
    n_val = 1024 if smoke else 65_536
    validation = [Batch(run.paths("validation").to(torch.float32))]
    if not control:   # drawn on the CPU, so the validation states are the same on every device
        validation.append(starts.sample(n_val, torch.Generator().manual_seed(20261102)))
    sources = [StandardStarts(problem, device=device, dtype=torch.float32)] + ([] if control else [starts])
    trainer = Trainer(problem, Entropic(spec.risk_aversion), config, device=device,
                      log=lambda msg: run.log(f"{spec.name} random states: {msg}"))
    result = trainer.fit(tuned, sources, validation, checkpoint=out_dir / "checkpoint.pt")
    torch.save({k: v.cpu() for k, v in tuned.network.state_dict().items()}, out_dir / "network.pt")
    result.save(out_dir / "training.json")
    (out_dir / "checkpoint.pt").unlink(missing_ok=True)

    test = run.paths("test")
    strategies = {"original": original, "tuned": tuned, "dp": run.dp()}
    losses = {k: problem.evaluate(s, test, device=device).loss for k, s in strategies.items() if s is not None}
    g = spec.risk_aversion
    comparisons = {"tuned_vs_original": paired_bootstrap(losses["original"], losses["tuned"], risk_aversion=g,
                                                         reps=run.reps, seed=0).percent}
    if "dp" in losses:
        comparisons["tuned_vs_dp"] = paired_bootstrap(losses["dp"], losses["tuned"], risk_aversion=g,
                                                      reps=run.reps, seed=1).percent
        comparisons["original_vs_dp"] = paired_bootstrap(losses["dp"], losses["original"], risk_aversion=g,
                                                         reps=run.reps, seed=1).percent
    probes = []
    for date, price, holding in RANDOM_STATE_PROBES.get(spec.cost, ()):
        row = {"date": date, "price": price, "holding": holding}
        for k, s in strategies.items():
            if s is not None:
                row[k], row[f"{k}_final_holding"] = risk_from_state(problem, s, date, price, holding,
                                                                    n_paths=2000 if smoke else 200_000,
                                                                    risk_aversion=g)
        probes.append(row)
    summary = {"control": control, "best_step": result.best_step, "test_entropic": {
        k: float(Entropic(g)(v)) for k, v in losses.items()}, "paired_percent": comparisons, "probes": probes}
    _write(out_dir / "results.json", summary)
    return summary


def skip_one_date(run: ExperimentRun, every: tuple[int, ...] = (2, 5), block: int = 25_000) -> dict:
    """Without costs: the mean-square error of the delta when one rebalancing date is skipped, or when it
    rebalances only every 'k' days, against the gap between the network and the daily delta."""
    problem, paths = run.problem, run.paths("test")
    greeks = frictionless_greeks(problem.market, problem.option)
    n_paths, n_dates = paths.shape[0], paths.shape[1] - 1
    losses, every_losses, skips = [], {k: [] for k in every}, []
    for i in range(0, n_paths, block):
        s = paths[i:i + block]
        gains = torch.zeros(s.shape[0], dtype=torch.float64)
        held, kept = None, {}
        every_gains = {k: torch.zeros_like(gains) for k in every}
        d = torch.zeros(s.shape[0], n_dates, dtype=torch.float64)
        for t in range(n_dates):
            phi = greeks.delta(t, s[:, t])
            ds = s[:, t + 1] - s[:, t]
            gains += (phi * ds).sum(-1)
            if t > 0:
                d[:, t] = ((phi - held) * ds).sum(-1)    # skipping date t keeps the holding of t - 1
            for k in every:
                if t % k == 0:
                    kept[k] = phi
                every_gains[k] += (kept[k] * ds).sum(-1)
            held = phi
        payoff = problem.option.payoff(s[:, -1])
        losses.append(payoff - problem.premium - gains)
        for k in every:
            every_losses[k].append(payoff - problem.premium - every_gains[k])
        skips.append(d)
    loss, d = torch.cat(losses), torch.cat(skips)
    mse = float((loss * loss).mean())
    skip = {t: 100 * (float(((loss + d[:, t]) ** 2).mean()) / mse - 1) for t in range(1, n_dates)}
    ratio = {k: float((torch.cat(v) ** 2).mean()) / mse for k, v in every_losses.items()}
    # on the device of the experiment, like its comparison table, so these are the losses of the network in that table
    network = problem.evaluate(run.network(), paths, device=run.device).loss
    excess = 100 * (float((network * network).mean()) / mse - 1)
    elasticity = math.log(ratio[2]) / math.log(2) if 2 in ratio else float("nan")
    result = {"mse_delta": mse, "network_over_delta_percent": excess, "skip_percent": skip,
              "skip_min": min(skip.values()), "skip_max": max(skip.values()), "every_ratio": ratio,
              "elasticity": elasticity, "equivalent_dates": n_dates / (1 + excess / 100) ** (1 / elasticity)}
    _write(run.dir / "skip-one-date.json", result)
    return result


def nested_local_risk_minimising(market, greeks: LevyGreeks, option, t: int, state: torch.Tensor, *,
                                 n_paths: int = 200_000, generator: torch.Generator | None = None,
                                 batches: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """phi_t = Cov_t(dS_t, H - sum_{u>t} delta_u dS_u) Cov_t(dS_t)^-1 by nested Monte Carlo from 'state' [n], with
    antithetic shocks and the Levy delta as control variate (each control term has zero conditional mean).
    Returns the hedge [n] and its standard error [n] from 'batches' batches."""
    n, half = market.n_assets, n_paths // 2
    dev = generator.device if generator is not None else torch.device("cpu")
    chol = market.cholesky.to(dev)
    drift, vol = market.drifts.to(dev), market.step_vols.to(dev)

    def step(prices, z):
        return prices * torch.exp(drift + vol * (z @ chol.T))

    start = state.to(dev, torch.float64)[None, :].expand(half, n)
    z1 = market.sample_shocks((half, n), generator, device=dev)
    rest = (market.sample_shocks((half, market.dates - t - 1, n), generator, device=dev)
            if market.dates - t > 1 else None)
    pairs = []
    for sign in (1.0, -1.0):
        first = step(start, sign * z1)
        prices, control = first, torch.zeros(half, dtype=torch.float64, device=dev)
        for u in range(t + 1, market.dates):
            nxt = step(prices, sign * rest[:, u - t - 1])
            control = control + (greeks.delta(u, prices) * (nxt - prices)).sum(-1)
            prices = nxt
        pairs.append((first - start, option.payoff(prices) - control))
    ds = torch.cat([pairs[0][0], pairs[1][0]])
    y = torch.cat([pairs[0][1], pairs[1][1]])
    cov = market.step_covariance(state.to(dev, torch.float64)[None, :])[0]

    def solve(sel):
        a, b = ds[sel], y[sel]
        return torch.linalg.solve(cov, ((a - a.mean(0)) * (b - b.mean())[:, None]).mean(0))

    idx = torch.arange(half, device=dev)
    per_batch = torch.stack([solve(torch.cat([b, b + half])) for b in torch.chunk(idx, batches)])
    return solve(torch.arange(2 * half, device=dev)), per_batch.std(0) / math.sqrt(batches)


def levy_check(run: ExperimentRun, dates: tuple[int, ...] = (0, 10, 20, 30, 40, 48), per_date: int = 40,
               inner_paths: int = 200_000, device: str = "cpu") -> dict:
    """Root-mean-square distance between the Levy delta and the nested Monte Carlo hedge at random test states,
    with the Monte Carlo noise subtracted (seeds of the thesis: 990001 + 10 n + 1 and 991001 + 10 n + 1)."""
    problem, spec = run.problem, run.spec
    n = spec.n_assets
    paths = run.paths("test")
    greeks = LevyGreeks(problem.market, problem.option)
    pick = torch.Generator().manual_seed(990_001 + 10 * n + 1)
    inner = torch.Generator(device=device).manual_seed(991_001 + 10 * n + 1)
    if "smoke" in spec.tags:
        dates, per_date, inner_paths = dates[:2], 2, 4_000
    states = []
    for t in dates:
        idx = torch.randperm(paths.shape[0], generator=pick)[:per_date]
        states += [(t, paths[i, t].clone()) for i in idx.tolist()]
    phis, errors, deltas = [], [], []
    for t, state in states:
        phi, se = nested_local_risk_minimising(problem.market, greeks, problem.option, t, state, n_paths=inner_paths,
                                               generator=inner)
        phis.append(phi.cpu())
        errors.append(se.cpu())
        deltas.append(greeks.delta(t, state[None].double())[0])
    phi, se, delta = torch.stack(phis), torch.stack(errors), torch.stack(deltas)
    noise = float((se ** 2).mean())
    d2 = float(((delta - phi) ** 2).mean())
    result = {"states": len(states), "inner_paths": inner_paths, "mc_standard_error_rms": math.sqrt(noise),
              "rmse": math.sqrt(d2), "rmse_without_noise": math.sqrt(max(d2 - noise, 0.0)),
              "mean_abs_hedge": float(phi.abs().mean())}
    _write(run.dir / "levy-check.json", result)
    return result


def grid_convergence(run: ExperimentRun, families: dict | None = None) -> dict:
    """The dynamic programme of a two-asset experiment on several grids, each measured on the test paths and compared
    with the finest grid of its family by the paired bootstrap (thesis, Sections 2.9 and 5.2). 'families'
    maps a name to grids ordered from coarse to fine; by default the families of 'catalog.GRID_STUDY' for the
    experiment, or its own grids."""
    from .catalog import GRID_STUDY
    from .spec import smoke_grid
    problem, spec = run.problem, run.spec
    base_name = spec.name.removesuffix("/smoke")
    families = families or GRID_STUDY.get(base_name) or {"grids": spec.dp}
    if "smoke" in spec.tags:   # two tiny grids per family (every smoke grid has the same size), so the comparison runs
        families = {k: (replace(smoke_grid(v[-1]), n_prices=11), smoke_grid(v[-1])) for k, v in families.items()}
    test = run.paths("test")
    losses: dict[str, torch.Tensor] = {}
    result: dict[str, dict] = {}
    for family, grids in families.items():
        rows = []
        for grid in grids:
            key = f"{grid.n_prices}x{grid.n_positions}"
            if key not in losses:
                strategy = solve_dp(problem, grid, spec.risk_aversion, run.dir / "dp-grids", device=run.device,
                                    log=run.log)
                losses[key] = problem.evaluate(strategy, test, device=run.device).loss
            rows.append(key)
        reference = rows[-1]
        result[family] = {"reference": reference, "grids": {
            k: {"entropic": float(Entropic(spec.risk_aversion)(losses[k])),
                **({} if k == reference else {"versus_reference_percent": paired_bootstrap(
                    losses[reference], losses[k], risk_aversion=spec.risk_aversion, reps=run.reps,
                    seed=20261107).percent})} for k in rows}}
    _write(run.dir / "grid-convergence.json", result)
    return result


def cross_criteria(cvar_run: ExperimentRun) -> dict:
    """Every strategy of a CVaR experiment (networks and rules trained or calibrated on either criterion) measured
    with the entropic risk and with the CVaR at 99% on the test sample."""
    outcomes: dict[str, HedgeOutcome] = cvar_run.outcomes()
    g = cvar_run.spec.risk_aversion
    entropic, cvar99 = risk_measure("entropic", g), risk_measure("cvar99")
    table = {k: {"entropic": float(entropic(o.loss)), "cvar99": float(cvar99(o.loss)),
                 "mse": float(MeanSquare()(o.loss)), "cost": float(o.cost.mean())} for k, o in outcomes.items()}
    _write(Path(cvar_run.dir) / "cross-criteria.json", table)
    return table
