"""End-to-end runs of the pipeline: the smoke versions on tiny samples, and the networks of the thesis on its test
paths (slow)."""
import json
from dataclasses import replace

import pytest
import torch

from deephedge.cli import REFERENCE, check
from deephedge.experiments import catalog
from deephedge.experiments.pipeline import PRETRAINED, ExperimentRun


def quiet(message: str) -> None:
    pass


def test_a_folder_is_reused_only_by_the_run_that_made_it(tmp_path):
    spec = catalog.get("no-cost/1", smoke=True)
    ExperimentRun(spec, tmp_path, log=quiet).network()                  # trained here
    ExperimentRun(spec, tmp_path, log=quiet)                            # the same run, reused
    with pytest.raises(RuntimeError, match="trained network"):
        ExperimentRun(spec, tmp_path, log=quiet, pretrained=True)
    with pytest.raises(RuntimeError, match="another specification"):
        ExperimentRun(spec.replace(network=replace(spec.network, steps=30)), tmp_path, log=quiet)
    (tmp_path / "no-cost_1_smoke" / "run.json").unlink()
    with pytest.raises(RuntimeError, match="without a record"):
        ExperimentRun(spec, tmp_path, log=quiet)


def test_pretrained_is_the_network_of_the_thesis(tmp_path):
    run = ExperimentRun(catalog.get("base/1"), tmp_path, log=quiet, pretrained=True)
    weights = run.network().network.state_dict()
    published = torch.load(PRETRAINED / "base_1.pt", weights_only=True)
    assert all(torch.equal(weights[k].cpu(), v) for k, v in published.items())
    # an experiment without a network of the thesis is an error, never a training run
    smoke = ExperimentRun(catalog.get("base/1", smoke=True), tmp_path, log=quiet, pretrained=True)
    with pytest.raises(FileNotFoundError, match="no network of the thesis"):
        smoke.network()
    assert not (smoke.dir / "network.pt").exists()


def test_smoke_run_without_costs(tmp_path):
    run = ExperimentRun(catalog.get("no-cost/1", smoke=True), tmp_path, log=quiet)
    rows = {r["name"]: r for r in run.run()["table"]["rows"]}
    assert list(rows) == ["network", "delta"]
    assert set(rows["delta"]["versus_reference"]) == {"mse", "entropic", "std", "cvar95", "cvar99", "cvar999"}
    for name in ("network.pt", "training.json", "outcomes.pt", "comparison.json"):
        assert (run.dir / name).exists()


def test_smoke_run_with_the_dynamic_programme(tmp_path):
    result = ExperimentRun(catalog.get("base/1", smoke=True), tmp_path, log=quiet).run()
    assert [r["name"] for r in result["table"]["rows"]] == ["network", "dp", "whalley-wilmott"]


def test_smoke_run_with_rules(tmp_path):
    run = ExperimentRun(catalog.get("no-correlation/3", smoke=True), tmp_path, log=quiet)
    result = run.run()
    assert [r["name"] for r in result["table"]["rows"]] == ["network", "atkinson-box"]
    calibration = json.loads((run.dir / "calibration.json").read_text(encoding="utf-8"))
    assert calibration["atkinson-box"]["done"] and calibration["atkinson-box"]["scale"] > 0


@pytest.mark.slow
def test_smoke_run_with_the_corrected_box(tmp_path):
    result = ExperimentRun(catalog.get("base/3", smoke=True), tmp_path, log=quiet).run()
    assert [r["name"] for r in result["table"]["rows"]] == ["network", "corrected-box"]


@pytest.mark.slow
def test_smoke_run_with_the_two_asset_programme(tmp_path):
    result = ExperimentRun(catalog.get("base/2", smoke=True), tmp_path, log=quiet).run()
    assert [r["name"] for r in result["table"]["rows"]] == ["network", "dp", "corrected-box"]


@pytest.mark.slow
def test_smoke_run_with_the_calibrated_centre(tmp_path):
    run = ExperimentRun(catalog.get("base/10", smoke=True), tmp_path, log=quiet)
    rows = [r["name"] for r in run.run()["table"]["rows"]]
    assert rows == ["network", "corrected-box", "corrected-box/calibrated-centre"]
    calibration = json.loads((run.dir / "calibration.json").read_text(encoding="utf-8"))
    assert calibration["corrected-box/calibrated-centre"]["kind"] == "damped-corrected-box"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["no-cost/1", "no-cost/2", "cost-1pct/1"])
def test_the_networks_of_the_thesis_give_its_published_numbers(tmp_path, name):
    # the networks of the thesis on its 200,000 test paths, with the dynamic programme and the rules computed here
    comparison = ExperimentRun(catalog.get(name), tmp_path, log=quiet, pretrained=True).comparison()
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    entry = reference["experiments"][name]
    failed = [(what, detail) for what, ok, detail in check(comparison, entry, reference["tolerance"]) if not ok]
    assert not failed
