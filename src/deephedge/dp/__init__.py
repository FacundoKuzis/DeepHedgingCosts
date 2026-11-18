"""The dynamic-programming optimum, the benchmark of the networks with one and two assets (thesis, Section 2.7).

:class:`OneAssetDP` solves the one-asset problem and returns an :class:`OptimalBand`; :class:`TwoAssetDP` solves the
two-asset problem and returns an :class:`OptimalGridStrategy`, which reads the tables of the programme from disk.
Both strategies implement :meth:`~deephedge.strategies.HedgingStrategy.act`, so they are evaluated by the same
accounting as every other strategy, and both report the entropic risk the programme predicts for them.
"""
from .one_asset import OneAssetDP, OneAssetGrid, OptimalBand
from .two_assets import OptimalGridStrategy, TwoAssetDP, TwoAssetGrid

__all__ = ["OneAssetDP", "OneAssetGrid", "OptimalBand", "OptimalGridStrategy", "TwoAssetDP", "TwoAssetGrid"]
