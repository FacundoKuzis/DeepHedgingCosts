import json

from deephedge.cli import REFERENCE
from deephedge.evaluation import WITH_COSTS, WITHOUT_COSTS
from deephedge.experiments import catalog
from deephedge.strategies.neural import count_parameters, neural_strategy


def test_the_24_experiments():
    names = set(catalog.CATALOG)
    assert len(names) == 24
    for prefix, ns in (("no-cost", (1, 2, 3, 10)), ("base", (1, 2, 3, 10)), ("cost-1pct", (1, 2, 3, 10)),
                       ("vol-x2", (2, 3, 10)), ("no-correlation", (2, 3, 10)), ("cvar", (1, 2, 3, 10))):
        assert {f"{prefix}/{n}" for n in ns} <= names
    assert {"vol-x1.5/10", "vol-x0.5/10"} <= names


def test_every_experiment_builds_its_problem_and_its_network():
    # with the parameter counts of the networks of the thesis (Section 3.7)
    counts = {(1, True): 8706, (2, True): 9220, (3, True): 9542, (10, True): 11796,
              (1, False): 8641, (2, False): 8962, (3, False): 9155, (10, False): 10506}
    for name, spec in catalog.CATALOG.items():
        problem = spec.problem()
        assert problem.n_assets == spec.n_assets, name
        network = neural_strategy(problem.market, problem.option, with_costs=spec.cost > 0).network
        assert count_parameters(network) == counts[(spec.n_assets, spec.cost > 0)], name


def test_the_published_tables_are_complete():
    # every experiment of the catalog, with the rows it compares and the six metrics of its table in each row
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert set(reference["experiments"]) == set(catalog.CATALOG)
    for name, entry in reference["experiments"].items():
        spec = catalog.CATALOG[name]
        metrics = set(WITHOUT_COSTS if spec.cost == 0 else WITH_COSTS)
        assert sorted(["network", *entry["versus"]]) == sorted(spec.compare), name
        assert set(entry["network"]) == metrics, name
        assert all(set(values) == metrics for values in entry["versus"].values()), name
