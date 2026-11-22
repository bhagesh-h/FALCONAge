"""`falconage demo` and `falconage run`: the one-command path a new user takes."""

from __future__ import annotations

import pandas as pd
import pytest

from falconage.cli.app import main
from falconage.demo import demo_data, write_demo


def test_demo_data_covers_the_bundled_array_clocks_and_is_labelled():
    d = demo_data(n=10)
    assert d.n_samples == 10
    assert {"age", "sex", "tissue", "group"} <= set(d.obs.columns)
    assert d.X.min().min() > 0 and d.X.max().max() < 1


@pytest.mark.slow
def test_run_writes_the_report_scores_and_warnings(tmp_path):
    betas, samples = write_demo(tmp_path / "in", n=12)
    out = tmp_path / "out"
    rc = main(["run", str(betas), "--samples", str(samples), "--outdir", str(out),
               "--clocks", "horvath2013,hannum", "--no-figures"])
    assert rc == 0
    for f in ("report.html", "scores.csv", "warnings.csv", "run_manifest.json",
              "technical_se.csv"):
        assert (out / f).exists(), f
    acc = pd.read_csv(out / "acceleration.csv")
    assert len(acc) == 12            # age came from the sample sheet


def test_run_refuses_a_sample_sheet_that_matches_no_sample(tmp_path, capsys):
    betas, _ = write_demo(tmp_path / "in", n=6)
    sheet = tmp_path / "wrong.csv"
    pd.DataFrame({"sample_id": ["X1", "X2"], "age": [30, 40]}).to_csv(sheet, index=False)
    rc = main(["run", str(betas), "--samples", str(sheet), "--outdir", str(tmp_path / "o")])
    assert rc == 2
    assert "none of the sample ids" in capsys.readouterr().err


def test_errors_are_one_line_with_an_exit_status_by_kind(tmp_path, capsys):
    assert main(["run", str(tmp_path / "absent.csv")]) == 3
    err = capsys.readouterr().err
    assert err.startswith("falconage: error: input not found") and "Traceback" not in err
    assert main(["run", "x.csv", "--log-file", "/proc/nope/x.log"]) == 3


def test_versions_prints_yaml_for_a_pipeline(capsys):
    assert main(["versions", "--process", "FALCONAGE_RUN"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == '"FALCONAGE_RUN":' and out[1].startswith("    falconage: ")
