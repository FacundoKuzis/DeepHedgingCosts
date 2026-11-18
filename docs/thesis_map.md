# From the thesis to the code

Numbering of the thesis *Deep Hedging con costos de transacción: comparación de estrategias de cobertura
para opciones sobre canastas*. In the thesis the dynamic-programming optimum is called *PD* (programación
dinámica); here it is `dp`. Commands write to `runs/<experiment>/`; add `--pretrained` to use the networks
of the thesis instead of training, and `--smoke` for a version that runs in seconds.

## Problem and method (Sections 2 and 3)

| Thesis | Content | Code |
|---|---|---|
| 2.1-2.2, Figure 1 | call price, delta and gamma | `pricing.black_scholes_call`, `pricing.black_scholes_delta_gamma` |
| 2.3 | correlated geometric Brownian motion, truncated shocks, martingale drift | `market.GBMMarket` |
| 2.4 | the basket call and the Levy approximation | `pricing.BasketCall`, `pricing.LevyGreeks` |
| 2.5 | proportional costs | `hedging.HedgingProblem.trading_cost` |
| 2.6 | mean square error and entropic risk | `risk.MeanSquare`, `risk.Entropic` |
| 2.7, Figure 3 | the no-trade band and the dynamic programme | `strategies.base.NoTradeBand`, `dp.OneAssetDP`, `dp.TwoAssetDP` |
| 2.8.1, eq. (2.19), Figure 2 | the band of Whalley and Wilmott | `strategies.bands.WhalleyWilmottBand` |
| 2.8.2, eq. (2.20), Figures 4-5 | the box of Atkinson and Ingpochai and its correction | `strategies.bands.AtkinsonBox`, `strategies.corrected_box` |
| 2.9 | the grids of the dynamic programme, the control of two identical assets | `dp.OneAssetGrid`, `dp.TwoAssetGrid`, `tests/test_dp.py` |
| 3.2, Table 1 | inputs of the network | `strategies.neural.SingleAssetFeatures`, `BasketFeatures` |
| 3.3, eq. (3.5)-(3.6), Figures 8-9 | position network and band network | `strategies.neural.PositionNetwork`, `BandNetwork` |
| 3.4, eq. (3.7)-(3.8), Table 2, Figure 11 | the result of a path | `hedging.HedgingProblem.run` |
| 3.5, eq. (3.10)-(3.12), Figure 10 | what the network minimises | `risk.Entropic`, `risk.RiskMeasure.training_loss` |
| 3.6 | backpropagation, Adam, training, validation and test | `training.Trainer` |
| 3.7, Table 3 | how the network and the grid grow with the assets | `strategies.neural.count_parameters`, `dp.TwoAssetGrid.table_bytes` |

Figures 1 to 11 explain the method and are drawn from these functions; they are not results. Figure 3
also shows the band with 32 rebalances a day, from a study of the rebalancing frequency that is not
part of the experiments and is not included.

## Experimental design (Section 4)

| Thesis | Content | Code |
|---|---|---|
| 4.1 | market, premiums, samples and seeds | `experiments.spec.ExperimentSpec`, `experiments.catalog` |
| 4.2.1 | the delta without costs, and the Levy delta against the local-risk-minimising hedge | `strategies.DeltaHedge`, `deephedge study levy-check no-cost/2` |
| 4.2.2, Table 4, Figures 12-13 | the rules with costs and their calibration | `calibration.ScaleSearch`, `calibration.calibrate_scale`, `calibration.refine_centre_and_scale`, `ExperimentRun.calibration` |
| 4.3, Figure 14 | training of the networks, hyperparameters | `training.TrainConfig`, `experiments.spec.NetworkTraining`, `deephedge figure training base/2` |
| 4.4, eq. (4.1)-(4.2), Figure 15 | metrics and the paired bootstrap | `risk.CVaR`, `evaluation.paired_bootstrap`, `evaluation.Comparison` |
| 4.5, Table 5 | the experiments | `experiments.catalog.CATALOG`, `deephedge list` |

Every run writes the calibrated scales (and centres) of its rules to `runs/<experiment>/calibration.json`.

## Results (Section 5)

| Thesis | Experiments | Command |
|---|---|---|
| Table 6 | `no-cost/{1,2,3,10}` | `deephedge run no-cost/1` (one per experiment) |
| Figure 16 | the same | `deephedge study skip-one-date no-cost/1` |
| Table 7 | `base/{1,2,3,10}` | `deephedge run base/2` |
| Table 7, the control of ten assets (corrected box with its centre calibrated) | `base/10` | `deephedge run base/10`, row `corrected-box/calibrated-centre` |
| 5.2, the grids of the two-asset programme | `base/2` | `deephedge study grid-convergence base/2` |
| Figure 17 | the same | `deephedge figure histogram base/2` |
| Figure 18 | the same | `deephedge figure quantiles base/2` |
| Figure 19 | `base/1` | `deephedge figure bands base/1` |
| 5.2.1, Figure 20 | `base/1` | `deephedge study random-states base/1`, and `--control` |
| Table 8 | `cost-1pct/{1,2,3,10}` | `deephedge run cost-1pct/2` |
| Figure 21 | `base/*` and `cost-1pct/*` | `deephedge report cost-1pct/2` for the CVaR at 99%, and the loss and cost of every path in `runs/<experiment>/outcomes.pt` for its split into costs and hedging error |
| Figure 22 | `cost-1pct/1` | `deephedge figure bands cost-1pct/1` |
| Figure 23 | `cost-1pct/1` | `deephedge study random-states cost-1pct/1`, and `--control` |
| Table 9 | `vol-x2/{2,3,10}`, `vol-x1.5/10`, `vol-x0.5/10`, `no-correlation/{2,3,10}` | `deephedge run vol-x2/10` |
| Figure 24 | `no-correlation/10`, `vol-x0.5/10`, `base/10`, `vol-x1.5/10`, `vol-x2/10` | `deephedge run vol-x2/10` (one per experiment), row `corrected-box` |
| 5.4, the changes between experiments | pairs of experiments | `deephedge change base/10 vol-x0.5/10 --rule corrected-box` |
| Table 10 | `cvar/{1,2,3,10}` (they also use `base/*`) | `deephedge run cvar/2` |
| Figure 25 | `cvar/1` | `deephedge figure bands cvar/1` |
| Figures 26-27 | `cvar/*` and `base/*` | `deephedge study cross-criteria cvar/2` |

`deephedge check <experiment>` compares the table of a run with the published numbers of Tables 6 to 10.

## The rows of each table

| Row of the thesis | Name in the code | Class |
|---|---|---|
| Red, Red con banda | `network` | `NeuralBand` (with costs), `NeuralPosition` (without) |
| PD | `dp` | `dp.OptimalBand` (one asset), `dp.OptimalGridStrategy` (two) |
| Banda de Whalley y Wilmott (centro fijo with the CVaR) | `whalley-wilmott` | `WhalleyWilmottBand` |
| Caja corregida (centro fijo with the CVaR) | `corrected-box` | `CorrectedBox` (damped with ten assets) |
| Caja corregida without correlation, where it is the box of order zero | `atkinson-box` | `AtkinsonBox` |
| Caja corregida con el centro calibrado | `corrected-box/calibrated-centre` | `CorrectedBox`, damped, centred at the delta of the volatilities multiplied by c |
| Banda de Whalley y Wilmott, centro corrido; Caja básica, centro corrido | `shifted-centre` | `AsymptoticBand` with another centre |
| Delta | `delta` | `DeltaHedge` |
| Escala teórica (g = 0,5) | `<rule>/theoretical` | the same rule with the risk aversion as its scale |
