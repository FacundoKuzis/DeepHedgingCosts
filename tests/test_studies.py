"""The studies on top of an experiment, in their smoke versions: each one runs end to end from the command line."""
import pytest

from deephedge.cli import main

STUDIES = [("random-states", "base/1"), ("skip-one-date", "no-cost/1"), ("levy-check", "base/2"),
           ("grid-convergence", "base/2"), ("cross-criteria", "cvar/1")]


@pytest.mark.parametrize(("study", "name"), STUDIES, ids=[study for study, _ in STUDIES])
def test_every_study_runs(tmp_path, study, name):
    assert main(["study", study, name, "--smoke", "--workdir", str(tmp_path), "--device", "cpu"]) == 0
