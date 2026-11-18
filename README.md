# deephedge

Deep hedging of a European call, on one asset or on an equal-weight basket of 2, 3 or 10 correlated
assets, under proportional transaction costs. The network is measured against the strongest
benchmarks available for each problem: the dynamic-programming optimum with one and two assets, and
the asymptotic no-trade rules of Whalley and Wilmott (one asset) and of Atkinson and Ingpochai
(baskets), calibrated on the same criterion the network is trained on.

This is the code of the thesis *Deep Hedging con costos de transacción: comparación de estrategias de
cobertura para opciones sobre canastas* (Facundo Kuzis). [`docs/thesis_map.md`](docs/thesis_map.md) maps
every section, table and figure of the thesis to the experiment, the command and the module that produce it.

## The problem

A dealer sells a call on the basket level `B_t = (S_1,t + ... + S_n,t) / n`, strike 100, maturity 50
trading days, and receives its premium. Every day the dealer chooses how many units `phi_t,i` of each
asset to hold until the next day, and pays a proportional cost `k S_t,i |phi_t,i - phi_t-1,i|` on
every trade. At maturity the hedge is liquidated, also at a cost. The loss of a path is

    L = payoff(B_N) - premium - sum_t phi_t . (S_t+1 - S_t) + costs,

and the hedger minimises its entropic risk `(1/g) log E[exp(g L)]` with `g = 0.5` (the mean square
error without costs). Prices follow a correlated geometric Brownian motion with zero rates: volatilities
of 15% to 30%, correlation 0.5, costs of 0.1% (the base case) and 1%.

Without costs the delta hedge is the benchmark. With costs the optimal strategy keeps the holding
inside a *no-trade band* around the delta and trades only to its nearest edge. Dynamic programming
computes that band exactly, but its grid grows exponentially with the number of assets, so it is only
feasible with one and two. The network is the deep-hedging method of Buehler et al. (2019): a strategy
represented by a neural network and trained on simulated paths to minimise the same risk.

## What is compared

| Strategy | Class | What it is |
|---|---|---|
| Network | `strategies.neural.NeuralBand` (with costs), `NeuralPosition` (without) | three hidden layers of 64 units; with costs it outputs the centre and half-width of a band |
| Dynamic programming | `dp.OneAssetDP`, `dp.TwoAssetDP` | the optimum on a grid of prices and holdings, with one and two assets |
| Whalley-Wilmott | `strategies.WhalleyWilmottBand` | asymptotic band for small costs, one asset |
| Atkinson box | `strategies.AtkinsonBox` | its extension to baskets, order zero in the correlation |
| Corrected box | `strategies.corrected_box.CorrectedBox` | the box with the first-order correction for correlated assets; with ten assets and 0.1%, also with its centre calibrated, as a control |
| Delta | `strategies.DeltaHedge` | Black-Scholes with one asset, the Levy approximation with baskets |

The rules have a free scale that is calibrated on the validation paths. Every strategy implements
`act(t, prices, position)` and is evaluated by the same accounting, `HedgingProblem.run`, on the same
200,000 test paths, and the differences are reported with a paired bootstrap.

## Install

Python 3.11 and PyTorch 2.2 or later (install the build for your GPU from pytorch.org if you have one):

    python -m venv .venv
    .venv\Scripts\activate            # Windows; source .venv/bin/activate elsewhere
    pip install -e ".[dev]"
    pytest                            # 52 fast tests, about three minutes on a laptop
    pytest -m slow                    # 6 slow ones, about ten minutes

`pyproject.toml` asks for minimum versions. `requirements.txt` pins the exact versions the tests were run
with (`pip install -r requirements.txt` and then `pip install -e . --no-deps`). A regular `pip install .`
works as well, since the networks of the thesis and its published tables are package data.

## Quick start

    deephedge list                          # the 24 experiments of the thesis
    deephedge run base/2 --smoke            # every stage on tiny samples, in seconds
    deephedge run no-cost/1                 # a real experiment (about 10 minutes on a CPU)
    deephedge check no-cost/1               # its numbers against the published ones
    deephedge run base/3 --pretrained       # with the networks of the thesis, without training
    deephedge report base/3 --latex         # the table of the thesis
    deephedge figure bands cost-1pct/1      # bands, training, histogram or quantiles
    deephedge change base/10 vol-x2/10      # how the gap of the corrected box changes with the volatility
    deephedge study random-states cost-1pct/1

Each experiment writes to `runs/<experiment>/`: the network and its training history, the dynamic
programme, the calibration of every rule, the outcome of every strategy on the test paths and the
comparison table. A stage whose file exists is not run again, so an interrupted run resumes where it
stopped (training resumes from its last checkpoint). `run.json` records the specification of the
experiment and where the network comes from (trained, or the network of the thesis with `--pretrained`),
and a run that asks for another one refuses the folder instead of mixing its results.

## The library in a few lines

    import torch
    from deephedge import GBMMarket, HedgingProblem, Entropic
    from deephedge.evaluation import Comparison
    from deephedge.experiments.spec import base_vols
    from deephedge.strategies import DeltaHedge, make_rule, neural_strategy
    from deephedge.pricing import frictionless_greeks
    from deephedge.training import StandardStarts, TrainConfig, Trainer
    from deephedge.hedging import Batch

    market = GBMMarket(base_vols(2), rho=0.5)                 # 15% and 20%, 50 daily dates
    problem = HedgingProblem.create(market, cost=0.001)       # Monte Carlo premium of the basket call
    network = neural_strategy(market, problem.option, with_costs=True)
    validation = Batch(market.simulate(65_536, torch.Generator().manual_seed(1), dtype=torch.float32))
    Trainer(problem, Entropic(0.5), TrainConfig(steps=5000, warmup=150)).fit(
        network, StandardStarts(problem), validation)

    test = market.simulate(200_000, torch.Generator().manual_seed(2))
    strategies = {"network": network, "corrected-box": make_rule("corrected-box", market, problem.option, 0.001, 1.5),
                  "delta": DeltaHedge(frictionless_greeks(market, problem.option))}
    table = Comparison({k: problem.evaluate(s, test) for k, s in strategies.items()})

## Reproducing the thesis

The catalog (`src/deephedge/experiments/catalog.py`) holds every experiment with the seeds and the basket
premiums of the thesis, so the validation and test paths are exactly its paths. The premiums are stored
because a Monte Carlo price depends on the device that drew it.

`src/deephedge/pretrained/` has the networks of the thesis (about 1 MB), and `--pretrained` uses them
instead of training, which regenerates any table in minutes. An experiment without a network of the
thesis is an error with `--pretrained`, never a silent training run. The single-asset networks are the
ones with three inputs, the ones of the tables of the thesis.

`deephedge check <experiment>` compares every number of the table of an experiment with the published one
in `src/deephedge/thesis_tables.json`, the network value within 1% and each difference with the network
within half a percentage point. A missing row or metric fails. Retrained with the same seeds on the GPU of
the thesis, `no-cost/1` gives the network in `pretrained/` and its twelve numbers of Table 6 to the printed
digit. Retrained on a CPU, which draws other batches, `cost-1pct/1` is another network and passes in its
eighteen numbers of Table 8. Its entropic risk is 1.7714 (1.771 in the thesis), and the dynamic programme
is 0.11% below it (0.10% in the thesis).

The tests check the reductions the thesis relies on (Levy with one asset is Black-Scholes, the box with one
asset is Whalley-Wilmott, the corrected box without correlation is the box, the dynamic programme without
costs is the delta, two identical assets are one asset), the accounting of the loss, the gradient of the
CVaR step, that training lowers the risk and resumes where it stopped, the parameter counts of every
network, the calibration and the paired bootstrap. The slow ones run the slowest stages of the pipeline on
small samples and evaluate the networks of the thesis on its 200,000 test paths, with the dynamic programme
and the rules computed here, against the published numbers of `no-cost/1`, `no-cost/2` and `cost-1pct/1`.

## Running times

On a laptop (Intel i7, 12 threads; GeForce with 4 GB). The CPU times are with four threads.

| Stage | CPU | GPU |
|---|---|---|
| a network, 4,096 paths per step | 0.15 s per step (3,000 to 12,000 steps) | 0.11 s per step |
| a CVaR network, 32,768 paths per step | 1.1 s per step | 0.3 s per step |
| one-asset dynamic programme, grid of the thesis | about 20 s | about 3 s |
| two-asset dynamic programme | about 3 hours with 101 x 40 nodes; about 11 hours (estimated) with the 101 x 79 of the thesis | 28 to 31 minutes, writing 13 to 15 GB of tables |
| the corrected box on 200,000 paths | under a minute with two assets; about half an hour damped, with ten | not measured |
| the corrected box with its centre calibrated (`base/10`) | 51 evaluations of the damped box on the 65,536 validation paths, several hours | not measured |

The two-asset programme writes one table per date under `runs/<experiment>/dp/` and reads them back to
hedge; delete them with `OptimalGridStrategy.delete_tables()` when the experiment is done.
`deephedge run --smoke` runs a small version of any experiment in seconds.

## Layout

    src/deephedge/
      market.py            GBMMarket: correlated GBM with truncated shocks and an exact martingale drift
      pricing.py           BasketCall, Black-Scholes, the Levy approximation, Monte Carlo premiums
      hedging.py           HedgingProblem: the loss of every strategy, with or without gradient
      risk.py              MeanSquare, Entropic, CVaR (with its tail-only training step)
      strategies/          the interface, the delta, the asymptotic bands, the corrected box, the networks
      dp/                  the dynamic programmes with one and two assets
      training.py          Trainer, StandardStarts, RandomStarts
      calibration.py       the grid searches of the rules
      evaluation.py        metrics, paired bootstrap, paired change between experiments, bold rule
      experiments/         ExperimentSpec, the catalog, the pipeline, the studies
      report.py, cli.py    tables, figures, the command line
      pretrained/          the networks of the thesis (package data)
      thesis_tables.json   the published numbers of the result tables (package data)
    tests/                 the tests, one file per part of the package
    docs/thesis_map.md     thesis -> experiment -> command -> module

## References

- Buehler, H., Gonon, L., Teichmann, J. and Wood, B. (2019). Deep hedging. *Quantitative Finance* 19(8),
  1271-1291.
- Whalley, A. E. and Wilmott, P. (1997). An asymptotic analysis of an optimal hedging model for option
  pricing with transaction costs. *Mathematical Finance* 7(3), 307-324.
- Atkinson, C. and Ingpochai, P. (2012). The effect of correlation and transaction costs on the pricing of
  basket options. *Applied Mathematical Finance* 19(2), 131-179.
- Levy, E. (1992). Pricing European average rate currency options. *Journal of International Money and
  Finance* 11(5), 474-491.
- Hodges, S. D. and Neuberger, A. (1989). Optimal replication of contingent claims under transaction costs.
  *Review of Futures Markets* 8(2), 222-239.
- Davis, M. H. A., Panas, V. G. and Zariphopoulou, T. (1993). European option pricing with transaction
  costs. *SIAM Journal on Control and Optimization* 31(2), 470-493.
