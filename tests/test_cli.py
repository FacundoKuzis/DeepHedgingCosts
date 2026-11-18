import copy
import json

from deephedge.cli import REFERENCE, check, main
from deephedge.evaluation import WITH_COSTS

TOLERANCE = {"value": 0.01, "points": 0.5}


def test_run_report_figures_and_change(tmp_path, capsys):
    common = ["--smoke", "--workdir", str(tmp_path), "--device", "cpu"]
    assert main(["run", "no-cost/2", *common]) == 0
    assert "| Network |" in capsys.readouterr().out
    assert main(["report", "no-cost/2", "--latex", *common]) == 0
    assert r"\begin{tabular}" in capsys.readouterr().out
    for figure in ("training", "histogram", "quantiles"):
        assert main(["figure", figure, "no-cost/2", *common]) == 0
    assert main(["figure", "bands", "cost-1pct/1", *common]) == 0
    assert (tmp_path / "cost-1pct_1_smoke" / "bands.png").exists()
    assert main(["change", "no-cost/2", "no-cost/3", "--rule", "delta", "--metric", "mse", *common]) == 0


def test_check_fails_on_every_wrong_or_missing_number():
    # the published numbers of base/1 pass against themselves, and fail with a wrong number, a missing row or a
    # missing metric
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))["experiments"]["base/1"]
    table = {"metrics": list(WITH_COSTS), "rows": [
        {"name": "network", "values": dict(reference["network"]), "versus_reference": None},
        *({"name": k, "values": {}, "versus_reference": dict(v)} for k, v in reference["versus"].items())]}
    assert all(ok for _, ok, _ in check({"table": table}, reference, TOLERANCE))
    wrong = copy.deepcopy(table)
    wrong["rows"][0]["values"]["cost"] *= 3
    wrong["rows"][1]["versus_reference"]["cvar99"] = [40.0, 39.0, 41.0]
    assert [w for w, ok, _ in check({"table": wrong}, reference, TOLERANCE) if not ok] == ["network cost", "dp cvar99"]
    missing_row = copy.deepcopy(table)
    del missing_row["rows"][2]
    assert [w for w, ok, _ in check({"table": missing_row}, reference, TOLERANCE) if not ok] == ["whalley-wilmott"]
    incomplete = {**reference, "network": {k: v for k, v in reference["network"].items() if k != "cvar999"}}
    assert [w for w, ok, _ in check({"table": table}, incomplete, TOLERANCE) if not ok] == ["network cvar999"]
