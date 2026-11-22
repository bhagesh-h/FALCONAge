"""The ``falconage`` command line.

argparse, so the CLI works with the base install. The verbs mirror the Python
API, so a pipeline written against either translates directly.

Built to run unattended inside a workflow manager (Nextflow, Snakemake):

* every verb takes ``--threads``, ``--log-level`` and ``--log-file``;
* progress goes to stdout, log records to stderr and, with ``--log-file``, to
  that file as well;
* the exit status says what went wrong: 0 success, 1 internal error, 2 usage
  error, 3 a problem with the input data, 4 a scoring or analysis error. An
  error prints one line naming the cause; ``--log-level debug`` adds the
  traceback;
* ``falconage versions`` prints the YAML a ``versions.yml`` needs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .._version import __version__


#: Exit status by kind of failure; see the module docstring.
EXIT_OK, EXIT_INTERNAL, EXIT_USAGE, EXIT_INPUT, EXIT_ANALYSIS = 0, 1, 2, 3, 4

_LOG_FILE = None


def _p(*a, **k):
    """Progress to stdout, flushed for a pipe, and to the log file if one is open."""
    print(*a, **k, flush=True)
    if _LOG_FILE is not None:
        print(*a, file=_LOG_FILE, flush=True)


# ---------------------------------------------------------------------------
def cmd_config(args) -> int:
    import falconage as fa

    cfg = fa.config()
    if args.json:
        _p(json.dumps(cfg, indent=2, sort_keys=True))
        return 0
    _p(f"FALCONAge {cfg['falconage']}")
    _p(f"  registry     {cfg['registry_version']}  ({cfg['n_clocks']} clocks)")
    t = cfg["clocks_by_availability"]
    _p(f"  availability {t['bundled']} bundled - {t['untraced']} untraced - "
       f"{t['licensed']} licensed")
    _p(f"  numpy        {cfg['numpy']}")
    _p(f"  cpu cores    {cfg['cpu_count']}")
    return 0


def _availability_line(clocks) -> str:
    """How many of these clocks run, stated before how many are catalogued."""
    n = {a: sum(c.availability == a for c in clocks)
         for a in ("bundled", "licensed", "untraced")}
    return (f"{n['bundled']} score offline (bundled), "
            f"{n['licensed']} need a licensed coefficient file, "
            f"{n['untraced']} have no traced coefficients")


def cmd_clocks(args) -> int:
    import falconage as fa

    reg = fa.registry.load()

    if args.action == "list":
        from ..registry.registry import normalise_availability

        clocks = list(reg)
        if args.tier:
            # The parser accepts the retired letters; compared as typed, "A"
            # matched no clock and the listing came back empty.
            tier = normalise_availability(args.tier)
            clocks = [c for c in clocks if c.availability == tier]
        if args.data_type:
            clocks = [c for c in clocks if c.data_type == args.data_type]
        if args.generation:
            clocks = [c for c in clocks if c.generation == args.generation]
        if args.untraced:
            clocks = [c for c in clocks if not c.coefficient_source.primary_source_traced]
        if args.search:
            ids = {c.id for c in reg.search(args.search)}
            clocks = [c for c in clocks if c.id in ids]
        clocks = sorted(clocks, key=lambda c: c.id)[: args.limit or None]

        _p(f"{'id':<26}{'availability':<14}{'gen':<12}{'scale':<22}{'n':>7}  name")
        for c in clocks:
            _p(f"{c.id:<26}{c.availability:<14}{c.generation:<12}{c.scale_type:<22}"
               f"{c.n_features or '?':>7}  {c.name[:40]}")
        _p(f"\n{len(clocks)} clock(s): {_availability_line(clocks)}")
        return 0

    if args.action == "info":
        from ..models.ops import describe_chain

        c = reg.get(args.clock)
        _p(f"{c.id}  --  {c.name}")
        _p(f"  year          {c.year}")
        _p(f"  species       {c.species}")
        _p(f"  data type     {c.data_type}")
        _p(f"  generation    {c.generation}")
        _p(f"  predicts      {', '.join(c.predicts)}  ({', '.join(c.unit)})")
        _p(f"  scale type    {c.scale_type}")
        _p(f"  legal ops     {', '.join(sorted(c.legal_operations))}")
        _p(f"  platform      {', '.join(c.platform) or 'unspecified'}")
        _p(f"  tissue        {', '.join(c.tissue) or 'unspecified'}")
        _p(f"  features      {c.n_features or 'unknown'}")
        _p(f"  postprocess   {describe_chain(c.postprocess)}")
        _p(f"  availability  {c.availability}")
        cs = c.coefficient_source
        _p(f"  provenance    {cs.provenance or 'unrecorded'}")
        _p(f"  traced        {cs.primary_source_traced}")
        if c.availability == "licensed":
            _p("")
            _p(reg.unavailable_message(c.id))
        if c.notes:
            _p(f"\n  {c.notes}")
        _p(f"\n  {c.cite()}")
        return 0

    if args.action == "cite":
        _p(reg.get(args.clock).cite(args.style))
        return 0

    return 2


def cmd_download(args) -> int:
    import falconage as fa

    res = fa.download(args.accession, dry_run=args.dry_run,
                      **({"want": args.want} if args.want else {}))
    _p(repr(res))
    for f in res.files:
        _p(f"  {f}")
    for n in res.notes:
        _p(f"  note: {n}")
    if res.samples is not None and len(res.samples):
        _p(f"  sample table: {res.samples.shape[0]} x {res.samples.shape[1]}")
        if args.outdir:
            p = Path(args.outdir) / f"{args.accession}_samples.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            res.samples.to_csv(p)
            _p(f"  wrote {p}")
    return 0


def cmd_preprocess(args) -> int:
    import falconage as fa

    data = fa.read(args.input)
    data = fa.prepare(data)
    report = fa.qc(data)
    _p(report.summary().to_string())
    for w in report.warnings:
        _p(f"  warning: {w}")
    out = Path(args.output)
    data.write_h5ad(out)
    _p(f"wrote {out}")
    return 0


def cmd_score(args) -> int:
    import falconage as fa

    data = fa.read(args.input)
    if data.modality == "dna_methylation":
        data = fa.prepare(data)

    clocks = args.clocks
    if clocks not in ("compatible", "all"):
        clocks = [c.strip() for c in clocks.split(",") if c.strip()]

    res = fa.score(data, clocks=clocks, dtype=args.dtype,
                   min_coverage=args.min_coverage, imputation=args.imputation)
    written = res.write(args.outdir)
    _p(repr(res))
    _p(res.summary().to_string())
    for k, v in written.items():
        _p(f"  {k:<12} {v}")
    if res.skipped and args.show_skipped:
        _p("\nskipped:")
        for k, v in res.skipped.items():
            _p(f"  {k:<24} {v}")
    return 0


def cmd_bench(args) -> int:
    import falconage as fa

    data = fa.read(args.input)
    res = fa.score(data, clocks=args.clocks)
    bench = fa.run_benchmark(res, condition_col=args.condition_col,
                             control=args.control, dataset_col=args.dataset_col)
    _p(bench.summary().to_string())
    if args.outdir:
        d = Path(args.outdir)
        d.mkdir(parents=True, exist_ok=True)
        bench.summary().to_csv(d / "benchmark_summary.csv")
        bench.per_dataset.to_csv(d / "benchmark_per_dataset.csv", index=False)
        _p(f"wrote {d}/benchmark_summary.csv")
    return 0


def cmd_cache(args) -> int:
    from ..download import cache_info, clear_cache

    if args.action == "ls":
        df = cache_info()
        _p(df.to_string(index=False) if len(df) else "cache is empty")
        if len(df):
            _p(f"\n{len(df)} file(s), {df['bytes'].sum() / 1e6:.1f} MB")
        return 0
    if args.action == "rm":
        freed = clear_cache(confirm=args.yes)
        _p(f"freed {freed / 1e6:.1f} MB")
        return 0
    return 2




def cmd_power(args) -> int:
    """How many samples, before any array is run."""
    import falconage as fa

    result = None
    if args.pilot:
        data = fa.read(args.pilot)
        if data.modality == "dna_methylation":
            data = fa.prepare(data)
        result = fa.score(data, clocks=[args.clock], min_coverage=args.min_coverage)
        fa.technical_se(result, data)

    p = fa.power(args.clock, effect=args.effect, sd=args.sd, result=result,
                 icc=args.icc, alpha=args.alpha, power=args.target_power,
                 replicates=args.replicates)
    _p(f"{p.clock}: to see {p.effect:g} at {p.power:.0%} power (alpha {p.alpha})")
    _p(f"  n per group   {p.n_per_group}")
    _p(f"  n total       {p.n_total}")
    _p(f"  sd used       {p.sd:.4g}   ({p.assumptions})")
    if p.icc is None:
        _p("  reliability   not established for this clock; the n above is not "
           "adjusted for measurement error")
    else:
        _p(f"  technical ICC {p.icc:.3f}   ({p.icc_source})")
        if p.n_if_perfectly_measured is not None:
            waste = p.n_total - 2 * p.n_if_perfectly_measured
            _p(f"  of which      {waste} sample(s) exist only to average out the assay")
    if p.replicates > 1:
        _p(f"  replicates    {p.replicates} per sample, assumed averaged")
    return 0


def cmd_consensus(args) -> int:
    """Does a group difference survive the multi-clock rule?"""
    import falconage as fa

    data = fa.read(args.input)
    if data.modality == "dna_methylation":
        data = fa.prepare(data)
    res = fa.score(data, clocks=args.clocks, min_coverage=args.min_coverage)
    rep = fa.consensus(res, args.group_col, reference=args.reference,
                       alpha=args.alpha)
    _p(f"verdict: {rep.verdict}")
    _p(f"  {rep.why}")
    _p("")
    _p(rep.table[["generation", "basis", "delta", "cohens_d", "p", "q_bh",
                  "p_bonferroni", "sig_bonferroni"]].round(5).to_string())
    if args.outdir:
        d = Path(args.outdir)
        d.mkdir(parents=True, exist_ok=True)
        rep.table.to_csv(d / "consensus.csv")
        (d / "consensus_verdict.txt").write_text(f"{rep.verdict}\n{rep.why}\n",
                                                 encoding="utf-8")
        _p(f"\nwrote {d}/consensus.csv")
    # A verdict is a result, not a failure: exit 0 unless asked otherwise, so a
    # workflow manager does not treat "unsupported" as a crashed task.
    return 5 if (args.fail_unsupported and rep.verdict == "unsupported") else 0


def cmd_report(args) -> int:
    """Read, check, score, quantify, interpret, and write the run's deliverables.

    The command a laboratory runs. The sequence itself is
    :func:`falconage.report.run_report`, so R and scripts get the same
    directory; this reads the input and prints what happened.
    """
    import falconage as fa
    from falconage.report import run_report

    data = fa.read(args.input)
    if data.modality == "dna_methylation":
        data = fa.prepare(data)
    _p(f"read {data.n_samples} sample(s) x {data.n_features} feature(s)"
       + (f" on {data.platform}" if data.platform else ""))

    clocks = args.clocks
    if clocks not in ("compatible", "all"):
        clocks = [c.strip() for c in clocks.split(",") if c.strip()]
    written = run_report(data, args.outdir, clocks=clocks, group_col=args.group_col,
                         reference=args.reference, level=args.level,
                         min_coverage=args.min_coverage, figures=not args.no_figures,
                         quarto=args.quarto, render=not args.no_render, log=_p)
    _p(f"\nwrote {written['report']}")
    if "quarto" in written:
        _p(f"wrote {written['quarto']}")
    return 0

def cmd_versions(args) -> int:
    """The software versions, as the YAML a pipeline's versions.yml takes."""
    import platform

    import numpy
    import pandas
    import scipy

    from .._version import REGISTRY_VERSION

    rows = {"falconage": __version__, "falconage_registry": REGISTRY_VERSION,
            "python": platform.python_version(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scipy": scipy.__version__}
    if args.process:
        _p(f'"{args.process}":')
        for k, v in rows.items():
            _p(f"    {k}: {v}")
    else:
        for k, v in rows.items():
            _p(f"{k}: {v}")
    return 0


def _load_input(path: str, kind: str, samples: str | None):
    """Read one input the way `run` and `demo` do, and attach the sample sheet."""
    import pandas as pd

    import falconage as fa

    for what, f in (("input", path), ("sample sheet", samples)):
        if f and not Path(f).exists():
            raise FileNotFoundError(f"{what} not found: {f}")
    if kind == "clinical":
        data = fa.read_clinical(path)
    elif kind == "nightingale":
        data = fa.read_nightingale(path)
    else:
        data = fa.read(path)
    if samples:
        sheet = pd.read_csv(samples, sep=None, engine="python", index_col=0)
        sheet.index = sheet.index.astype(str)
        missing = [s for s in data.obs.index if s not in sheet.index]
        if len(missing) == len(data.obs.index):
            raise SystemExit(
                f"none of the sample ids in {path} appear in the first column of "
                f"{samples}.\n  First ids in the data: "
                f"{', '.join(list(data.obs.index[:3]))}")
        if missing:
            _p(f"note: {len(missing)} sample(s) are not in {samples}; their age, "
               "sex and tissue stay unknown")
        new = [c for c in sheet.columns if c not in data.obs.columns]
        data.obs = data.obs.join(sheet[new], how="left")
    if data.modality == "dna_methylation":
        data = fa.prepare(data)
    return data


def cmd_run(args) -> int:
    """Everything, in one command: read, check, score, quantify, report."""
    import falconage as fa
    from falconage.report import run_report

    # Per-clock notes go to warnings.csv and the report; the console gets a
    # summary unless --verbose or --log-level asks for the full stream.
    if not (args.verbose or args.log_level in ("info", "debug")):
        fa.configure("none")
    if getattr(args, "clocks_dir", None):
        added = [fa.add_clock(y) for y in sorted(Path(args.clocks_dir).glob("*.y*ml"))]
        if not added:
            raise SystemExit(f"no .yaml clock entries in {args.clocks_dir}")
        _p(f"added {len(added)} clock(s) of your own: {', '.join(added)}")
    data = _load_input(args.input, args.kind, args.samples)
    outdir = args.outdir or f"{Path(args.input.rstrip('/')).name.split('.')[0]}_falconage"
    _p(f"read {data.n_samples} sample(s) x {data.n_features} feature(s)"
       + (f" on {data.platform}" if data.platform else ""))
    for col in ("age", "tissue"):
        if col not in data.obs.columns:
            _p(f"note: no '{col}' column"
               + (" -- age acceleration is skipped" if col == "age"
                  else " -- each clock's tissue is not checked")
               + "; add it with --samples sample_sheet.csv")
    clocks = args.clocks
    if clocks not in ("compatible", "all"):
        clocks = [c.strip() for c in clocks.split(",") if c.strip()]
    group = args.group_col
    if group is None and "group" in data.obs.columns:
        group = "group"
    written = run_report(data, outdir, clocks=clocks, group_col=group,
                         min_coverage=args.min_coverage, figures=not args.no_figures,
                         log=_p)
    import pandas as pd
    n_warn = len(pd.read_csv(written["warnings"])) if "warnings" in written else 0
    if getattr(args, "format", "csv") == "tsv":
        for f in sorted(Path(outdir).glob("*.csv")):
            pd.read_csv(f, low_memory=False).to_csv(f.with_suffix(".tsv"), sep="\t", index=False)
            f.unlink()
    warn_file = Path(outdir) / ("warnings.tsv" if getattr(args, "format", "csv") == "tsv"
                                else "warnings.csv")
    if n_warn:
        _p(f"{n_warn} note(s) about individual clocks: {warn_file}")
    _p(f"\ndone. Open {written['report']}")
    return 0


def cmd_demo(args) -> int:
    """Run the whole pipeline on a small synthetic dataset."""
    from falconage.demo import NOTE, write_demo

    outdir = Path(args.outdir)
    betas, samples = write_demo(outdir / "input")
    _p(NOTE)
    _p(f"wrote {betas} and {samples}\n")
    args.input, args.samples, args.outdir = str(betas), str(samples), str(outdir / "results")
    args.kind, args.clocks, args.group_col = "auto", "compatible", "group"
    args.clocks_dir = None
    return cmd_run(args)


# ---------------------------------------------------------------------------
EPILOG = """\
examples:
  falconage demo                                    everything, on synthetic data
  falconage run betas.csv --samples samples.csv     everything, on your data
  falconage run idats/ --samples samples.csv --outdir results --format tsv
  falconage clocks list --tier bundled              which clocks score offline
  falconage versions --process FALCONAGE_RUN        YAML for a pipeline's versions.yml

exit status: 0 success, 1 internal error, 2 usage error, 3 input data error,
             4 scoring or analysis error, 5 consensus verdict 'unsupported'
             (only with --fail-unsupported)

docs: https://bhagesh-h.github.io/FALCONAge/
"""


class _Fmt(argparse.RawDescriptionHelpFormatter):
    """Keep the hand-wrapped epilog; show a default only when it says something."""

    def _get_help_string(self, action):
        h = action.help or ""
        d = action.default
        if ("default" in h or d in (None, False, argparse.SUPPRESS)
                or action.option_strings == [] or isinstance(d, bool)):
            return h
        return f"{h} (default: {d})"


def _common() -> argparse.ArgumentParser:
    """Options every verb takes."""
    c = argparse.ArgumentParser(add_help=False)
    g = c.add_argument_group("run control")
    g.add_argument("--threads", type=int, default=None, metavar="N",
                   help="CPU threads for numerical work (default: the container's "
                        "OMP_NUM_THREADS, else all cores)")
    g.add_argument("--log-level", dest="log_level", default="info",
                   choices=["quiet", "info", "debug"],
                   help="quiet: errors only; info: progress and warnings; "
                        "debug: everything, and the traceback of an error")
    g.add_argument("--log-file", dest="log_file", default=None, metavar="PATH",
                   help="also write the log to this file")
    return c


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="falconage", formatter_class=_Fmt,
        description="FALCONAge: biological age and aging-clock scoring from DNA "
                    "methylation, clinical chemistry and NMR metabolomics data.",
        epilog=EPILOG)
    ap.add_argument("--version", action="version", version=f"falconage {__version__}")
    sub = ap.add_subparsers(dest="cmd", title="commands", metavar="<command>")
    common = _common()

    def verb(name, help, description=None):
        return sub.add_parser(name, help=help, description=description or help,
                              parents=[common], formatter_class=_Fmt)

    p = verb("run", "everything in one command: score, uncertainty, figures, report",
             "Read the input, check it, score every clock it supports, estimate "
             "uncertainty and age acceleration, test group differences across "
             "clocks, draw the figures and write report.html, all into --outdir.")
    p.add_argument("input", help="a beta matrix (.csv, .tsv, .parquet, .h5ad), a GEO "
                   "series matrix, a folder of IDAT files, or a clinical or NMR table")
    p.add_argument("--samples", default=None, metavar="CSV",
                   help="sample sheet: first column the sample id, then any of "
                        "age, sex, tissue, group")
    p.add_argument("--outdir", default=None,
                   help="output folder (default: <input name>_falconage)")
    p.add_argument("--format", default="csv", choices=["csv", "tsv"],
                   help="format of the output tables")
    p.add_argument("--kind", default="auto", choices=["auto", "clinical", "nightingale"],
                   help="set for a clinical chemistry or NMR table")
    p.add_argument("--clocks", default="compatible",
                   help="'compatible' or a comma-separated list of clock ids")
    p.add_argument("--clocks-dir", dest="clocks_dir", default=None, metavar="DIR",
                   help="folder of clocks of your own (.yaml + .csv), scored "
                        "alongside the catalogue")
    p.add_argument("--group-col", dest="group_col", default=None,
                   help="sample-sheet column to compare groups on "
                        "(default: 'group' if present)")
    p.add_argument("--min-coverage", dest="min_coverage", type=float, default=0.8,
                   help="smallest share of a clock's features that must be present")
    p.add_argument("--no-figures", dest="no_figures", action="store_true",
                   help="skip the figures (faster)")
    p.add_argument("--verbose", action="store_true",
                   help="print every per-clock note as it happens")
    p.set_defaults(fn=cmd_run)

    p = verb("demo", "run everything on a small synthetic dataset")
    p.add_argument("--outdir", default="falconage_demo", help="output folder")
    p.add_argument("--format", default="csv", choices=["csv", "tsv"],
                   help="format of the output tables")
    p.add_argument("--min-coverage", dest="min_coverage", type=float, default=0.8,
                   help="smallest share of a clock's features that must be present")
    p.add_argument("--no-figures", dest="no_figures", action="store_true",
                   help="skip the figures (faster)")
    p.add_argument("--verbose", action="store_true",
                   help="print every per-clock note as it happens")
    p.set_defaults(fn=cmd_demo)

    p = verb("versions", "software versions, as YAML for a pipeline's versions.yml")
    p.add_argument("--process", default=None,
                   help="nest the versions under this process name, as nf-core does")
    p.set_defaults(fn=cmd_versions)

    p = verb("config", "what this installation resolved to")
    p.add_argument("--json", action="store_true", help="print as JSON")
    p.set_defaults(fn=cmd_config)

    p = verb("clocks", "browse the clock catalogue")
    p.add_argument("action", choices=["list", "info", "cite"],
                   help="list clocks, show one, or print its citation")
    p.add_argument("clock", nargs="?", help="clock id, for info and cite")
    p.add_argument("--tier", metavar="AVAILABILITY",
                   choices=["bundled", "untraced", "licensed", "A", "B", "C"],
                   help="bundled, untraced or licensed")
    p.add_argument("--data-type", dest="data_type",
                   help="e.g. dna_methylation, clinical_chemistry")
    p.add_argument("--generation", help="first, second, pace, ...")
    p.add_argument("--search", help="substring of the id, name, target or citation")
    p.add_argument("--untraced", action="store_true",
                   help="only clocks with no established primary source")
    p.add_argument("--limit", type=int, default=0, help="show at most this many (0: all)")
    p.add_argument("--style", default="plain", choices=["plain", "bibtex"],
                   help="citation style, for cite")
    p.set_defaults(fn=cmd_clocks)

    p = verb("download", "fetch public data by accession")
    p.add_argument("accession", help="e.g. GSE40279")
    p.add_argument("--want", choices=["matrix", "suppl", "both"],
                   help="the series matrix, the supplementary files, or both")
    p.add_argument("--outdir", help="where to save the files")
    p.add_argument("--dry-run", action="store_true",
                   help="say what would be fetched and how many bytes, fetch nothing")
    p.set_defaults(fn=cmd_download)

    p = verb("preprocess", "raw or public data to a scoreable .h5ad")
    p.add_argument("--input", required=True, help="a beta matrix, series matrix or IDAT folder")
    p.add_argument("--output", required=True, help="the .h5ad to write")
    p.set_defaults(fn=cmd_preprocess)

    p = verb("score", "score a dataset and write the score tables only")
    p.add_argument("--input", required=True, help="a beta matrix, series matrix or .h5ad")
    p.add_argument("--outdir", required=True, help="output folder")
    p.add_argument("--clocks", default="compatible",
                   help="'compatible', 'all', or a comma-separated list")
    p.add_argument("--dtype", default=None, choices=["float64", "float32"],
                   help="floating-point precision (default float64)")
    p.add_argument("--min-coverage", dest="min_coverage", type=float, default=0.8,
                   help="smallest share of a clock's features that must be present")
    p.add_argument("--imputation", default="reference", choices=["reference", "mean", "none"],
                   help="how an absent feature is filled; 'none' refuses instead")
    p.add_argument("--show-skipped", dest="show_skipped", action="store_true",
                   help="list every clock not scored, with the reason")
    p.set_defaults(fn=cmd_score)

    p = verb("bench", "run the AA1/AA2 case/control benchmark")
    p.add_argument("--input", required=True, help="scored or scoreable data with conditions")
    p.add_argument("--outdir", help="output folder")
    p.add_argument("--clocks", default="compatible", help="'compatible' or a list")
    p.add_argument("--condition-col", dest="condition_col", default="condition",
                   help="column holding the condition")
    p.add_argument("--control", default="HC", help="the control level")
    p.add_argument("--dataset-col", dest="dataset_col", help="column naming the study")
    p.set_defaults(fn=cmd_bench)

    p = verb("cache", "inspect or clear the download cache")
    p.add_argument("action", choices=["ls", "rm"], help="list or remove")
    p.add_argument("--yes", action="store_true", help="remove without asking")
    p.set_defaults(fn=cmd_cache)

    p = verb("power", "how many samples are needed to see an effect")
    p.add_argument("--clock", required=True, help="clock id")
    p.add_argument("--effect", type=float, required=True,
                   help="the difference worth detecting, in the clock's own unit")
    p.add_argument("--sd", type=float, default=None,
                   help="population SD; measured from --pilot when given")
    p.add_argument("--pilot", default=None,
                   help="a scored pilot dataset, to measure sd and the ICC from")
    p.add_argument("--icc", type=float, default=None, help="technical ICC, if known")
    p.add_argument("--alpha", type=float, default=0.05, help="significance level")
    p.add_argument("--target-power", type=float, default=0.80, help="power to aim for")
    p.add_argument("--replicates", type=int, default=1, help="technical replicates per sample")
    p.add_argument("--min-coverage", type=float, default=0.8,
                   help="coverage floor for --pilot")
    p.set_defaults(fn=cmd_power)

    p = verb("consensus", "does a group difference hold up across clocks?")
    p.add_argument("input", help="a beta matrix or .h5ad with a group column")
    p.add_argument("--group-col", default="condition", help="column holding the groups")
    p.add_argument("--reference", default=None, help="the comparison group")
    p.add_argument("--clocks", default="compatible", help="'compatible' or a list")
    p.add_argument("--alpha", type=float, default=0.05, help="significance level")
    p.add_argument("--min-coverage", type=float, default=0.8,
                   help="smallest share of a clock's features that must be present")
    p.add_argument("--outdir", default=None, help="write consensus.csv and the verdict here")
    p.add_argument("--fail-unsupported", dest="fail_unsupported", action="store_true",
                   help="exit with status 5 when the verdict is 'unsupported'")
    p.set_defaults(fn=cmd_consensus)

    p = verb("report", "read, check, score and write report.html (run is the simpler form)")
    p.add_argument("input", help="a beta matrix, series matrix or .h5ad")
    p.add_argument("--outdir", default="falconage_report", help="output folder")
    p.add_argument("--clocks", default="compatible", help="'compatible' or a list")
    p.add_argument("--group-col", default=None, help="column to compare groups on")
    p.add_argument("--reference", default=None, help="the comparison group")
    p.add_argument("--level", type=float, default=0.90, help="prediction-interval level")
    p.add_argument("--min-coverage", type=float, default=0.8,
                   help="smallest share of a clock's features that must be present")
    p.add_argument("--no-figures", action="store_true", help="skip the figures")
    p.add_argument("--quarto", action="store_true",
                   help="also write the step-ordered Quarto report and render it")
    p.add_argument("--no-render", action="store_true",
                   help="with --quarto, write falconage_report.qmd without rendering it")
    p.set_defaults(fn=cmd_report)

    return ap


def _set_threads(n: int | None) -> None:
    """Pin the numerical libraries' thread pools before numpy is imported.

    A workflow manager allots a task a number of CPUs; using more than that
    oversubscribes the node. Set from ``--threads`` (in Nextflow,
    ``--threads ${task.cpus}``).
    """
    import os

    if n is None:
        return
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(max(1, n))


def _exit_code(exc: BaseException) -> int:
    from ..core.errors import (AnalysisError, DataError, DownloadError, FalconError,
                               RegistryError, ScoringError)

    if isinstance(exc, (DataError, DownloadError, FileNotFoundError, IsADirectoryError,
                        PermissionError, UnicodeDecodeError)):
        return EXIT_INPUT
    if isinstance(exc, (ScoringError, AnalysisError, RegistryError, FalconError)):
        return EXIT_ANALYSIS
    return EXIT_INTERNAL


def main(argv: list[str] | None = None) -> int:
    global _LOG_FILE
    import logging

    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "cmd", None):
        ap.print_help()
        return EXIT_OK
    _set_threads(getattr(args, "threads", None))

    from ..core.logging import configure, get_logger

    level = getattr(args, "log_level", "info")
    configure({"quiet": "none", "info": "inform", "debug": "debug"}[level])
    log = get_logger()
    if getattr(args, "log_file", None):
        try:
            Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
            _LOG_FILE = open(args.log_file, "a", encoding="utf-8")
        except OSError as exc:
            print(f"falconage: error: cannot write the log file {args.log_file}: "
                  f"{exc.strerror}", file=sys.stderr)
            return EXIT_INPUT
        fh = logging.StreamHandler(_LOG_FILE)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        log.addHandler(fh)
    log.debug("falconage %s: %s", __version__, " ".join(argv if argv is not None else sys.argv[1:]))
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        return 130
    except SystemExit as exc:          # a verb's own one-line refusal
        if isinstance(exc.code, str):
            print(f"falconage: error: {exc.code}", file=sys.stderr)
            if _LOG_FILE:
                print(f"error: {exc.code}", file=_LOG_FILE)
            return EXIT_USAGE
        raise
    except Exception as exc:  # noqa: BLE001 - the one place every failure is reported
        code = _exit_code(exc)
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        kind = "internal error" if code == EXIT_INTERNAL else "error"
        print(f"falconage: {kind}: {first}", file=sys.stderr)
        rest = str(exc).strip().splitlines()[1:]
        for ln in rest:
            print(f"  {ln.strip()}", file=sys.stderr)
        if level == "debug" or code == EXIT_INTERNAL:
            import traceback
            traceback.print_exc(file=sys.stderr if level == "debug" else (_LOG_FILE or sys.stderr))
        if code == EXIT_INTERNAL and level != "debug":
            print("  this is a bug: please report it with the output of "
                  "`falconage versions` and a rerun with --log-level debug",
                  file=sys.stderr)
        if _LOG_FILE:
            print(f"{kind}: {exc}", file=_LOG_FILE)
        return code
    finally:
        if _LOG_FILE:
            _LOG_FILE.close()
            _LOG_FILE = None


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
