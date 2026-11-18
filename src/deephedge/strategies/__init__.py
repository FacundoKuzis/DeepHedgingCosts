"""Hedging strategies. Every one implements 'act(t, prices, position) -> new position'."""
from .bands import AsymptoticBand, AtkinsonBox, WhalleyWilmottBand, asymptotic_rule
from .base import HedgingStrategy, NoHedge, NoTradeBand
from .delta import DeltaHedge
from .neural import BandNetwork, NeuralBand, NeuralPosition, PositionNetwork, neural_strategy

RULE_KINDS = ("whalley-wilmott", "atkinson-box", "corrected-box", "damped-corrected-box")


def make_rule(kind: str, market, option, cost: float, scale: float, centre: float = 1.0) -> HedgingStrategy:
    """A no-trade rule by name, with the scale 'g' of its half-width and the factor 'centre' that moves its
    centre to the delta computed with every volatility multiplied by it (1 is the frictionless delta). The shifted
    centre of the CVaR experiments moves the band of Whalley and Wilmott or the box of order zero (Section 5.5), and
    the control of the base case with ten assets moves the damped corrected box (Section 5.2)."""
    if kind in ("whalley-wilmott", "atkinson-box"):
        if (kind == "whalley-wilmott") != (market.n_assets == 1):
            raise ValueError(f"{kind} does not apply to {market.n_assets} asset(s)")
        return asymptotic_rule(market, option, cost, scale, centre_vol_scale=centre)
    if kind in ("corrected-box", "damped-corrected-box"):
        from .corrected_box import corrected_box
        return corrected_box(market, option, cost, scale, damped=kind == "damped-corrected-box",
                             centre_vol_scale=centre)
    raise ValueError(f"unknown rule {kind!r}; choose from {RULE_KINDS}")


__all__ = ["AsymptoticBand", "AtkinsonBox", "BandNetwork", "DeltaHedge", "HedgingStrategy", "NeuralBand",
           "NeuralPosition", "NoHedge", "NoTradeBand", "PositionNetwork", "RULE_KINDS", "WhalleyWilmottBand",
           "asymptotic_rule", "make_rule", "neural_strategy"]
