import math

import torch

from deephedge import GBMMarket
from deephedge.pricing import BasketCall
from deephedge.strategies.neural import BandNetwork, SingleAssetFeatures

OPTION = BasketCall(100.0)


def test_single_asset_features():
    # the three inputs of the thesis (Table 1), in their order
    f = SingleAssetFeatures(GBMMarket((0.2,)), OPTION)
    x = math.log(1.1)
    tau = (50 - 10) / 252
    expected = [x / (0.2 * math.sqrt(tau + 1 / 252)), math.sqrt(tau / (50 / 252)), 5 * x]
    assert torch.allclose(f(10, torch.tensor([[110.0]], dtype=torch.float64))[0],
                          torch.tensor(expected, dtype=torch.float64))


def test_band_network_projects_the_holding_onto_its_band():
    # thesis, eq. (3.6): the projection written with ReLUs is the clip to the band
    net = BandNetwork(3, 1, sees_position=False)
    features = torch.randn(100, 3)
    position = torch.rand(100, 1) * 1.4 - 0.2
    lo, hi = net.band(features, position)
    assert torch.allclose(net(features, position), torch.minimum(torch.maximum(position, lo), hi), atol=1e-7)
