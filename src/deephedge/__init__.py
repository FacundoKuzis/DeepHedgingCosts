"""deephedge: deep hedging of a basket call under proportional transaction costs.

The code behind the thesis *Deep Hedging con costos de transacción: comparación de estrategias de cobertura para
opciones sobre canastas* (Facundo Kuzis).
Start with :mod:`deephedge.experiments.catalog` for the experiments and with
'docs/thesis_map.md' for the map from the thesis to the code.
"""
from .hedging import HedgeOutcome, HedgingProblem
from .market import GBMMarket
from .pricing import BasketCall, option_premium
from .risk import CVaR, Entropic, MeanSquare, RiskMeasure

__version__ = "1.0.0"
__all__ = ["BasketCall", "CVaR", "Entropic", "GBMMarket", "HedgeOutcome", "HedgingProblem", "MeanSquare",
           "RiskMeasure", "option_premium"]
