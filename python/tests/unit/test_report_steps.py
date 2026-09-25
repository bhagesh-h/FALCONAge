"""The step-ordered report: every file of a run appears once, under its step."""

from __future__ import annotations

import re
import shutil

import numpy as np
import pandas as pd
import pytest

import falconage as fa
from falconage.report import STEPS, run_report
from falconage.report.outputs import OWN, classify, collect


def _data():
    reg = fa.registry.load()
    feats = sorted(set().union(*(reg.feature_ids(c) for c in
                                 ("hannum", "horvath2013", "dnamphenoage"))))
    rng = np.random.default_rng(5)
    n = 16
    age = np.linspace(22, 78, n)
    base = rng.uniform(0.15, 0.85, len(feats))
    X = np.clip(base + rng.normal(0, 0.0015, len(feats)) * (age[:, None] - 50)
                + rng.normal(0, 0.01, (n, len(feats))), 0.001, 0.999)
    ids = [f"s{i:02d}" for i in range(n)]
    obs = pd.DataFrame({"age": age, "tissue": "whole blood",
                        "group": ["case", "control"] * (n // 2)}, index=ids)
    return fa.FalconData(X=pd.DataFrame(X, index=ids, columns=feats), obs=obs,
                         modality="dna_methylation", platform="450K")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    written = run_report(_data(), out, clocks=["hannum", "horvath2013", "dnamphenoage"],
                         group_col="group", quarto=True, render=False, log=lambda s: None)
    return out, written


def test_every_file_of_the_run_appears_once_under_its_step(run_dir):
    out, written = run_dir
    qmd = written["quarto"].read_text(encoding="utf-8")
    files = [p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()]
    listed = [f for f in files if not any(re.fullmatch(pat.replace("*", ".*"), f)
                                          for pat in OWN)]
    assert "run_manifest.json" in listed and "scores_wide.csv" in listed
    assert any(f.startswith("figures/") for f in listed)
    for f in listed:
        o = classify(f)
        n = qmd.count(f)
        if f == "run_manifest.json":
            assert "Coefficients used" in qmd
        elif o is not None and o.kind == "text" and f == "consensus_verdict.txt":
            assert "<strong>Verdict.</strong>" in qmd
        else:
            assert n >= 1, f"{f} is in the output directory and not in the report"


def test_the_steps_appear_in_order(run_dir):
    _, written = run_dir
    qmd = written["quarto"].read_text(encoding="utf-8")
    at = [qmd.index(f"## {s.number}. {s.title} {{#step-{s.number}}}") for s in STEPS]
    assert at == sorted(at)


def test_each_file_is_placed_under_the_step_the_table_gives_it(run_dir):
    out, _ = run_dir
    placed = collect(out)
    step_of = {p.relative_to(out).as_posix(): n for n, items in placed.items()
               for p, _ in items}
    assert step_of["run_manifest.json"] == 0
    assert step_of["qc_per_sample.csv"] == 1
    assert step_of["qc.csv"] == 2
    assert step_of["scores_wide.csv"] == 3
    assert step_of["acceleration.csv"] == 6
    assert step_of["consensus.csv"] == 7
    assert step_of["evidence.csv"] == 8
    assert step_of["report.html"] == 9
    assert step_of["figures/ba_vs_ca_hannum.png"] == 4


def test_the_report_loads_nothing_from_outside(run_dir):
    _, written = run_dir
    qmd = written["quarto"].read_text(encoding="utf-8")
    assert not re.search(r'(src|href)="https?://', qmd)
    assert "embed-resources: true" in qmd


@pytest.mark.skipif(shutil.which("quarto") is None, reason="Quarto not on the path")
def test_the_rendered_report_is_one_self_contained_file(run_dir):
    from falconage.report import write_quarto_report

    out, _ = run_dir
    html = write_quarto_report(out, render=True)
    text = html.read_text(encoding="utf-8")
    assert not re.search(r'<(script|link|img)[^>]+(src|href)="(?!data:)[^"#]', text)
