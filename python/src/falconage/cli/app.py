"""The ``falconage`` command line.

argparse rather than typer, so the CLI works with the base install. The verbs
mirror the Python API one for one, and ``inst/scripts/falconage.R`` mirrors
these, so a pipeline written against any of the three translates directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .._version import __version__


def _p(*a, **k):  # print to stdout, unbuffered enough for a pipe
    print(*a, **k, flush=True)


# ---------------------------------------------------------------------------
def cmd_config(args) -> int:
    import falconage as fa

    cfg = fa.config()
    if args.json:
        _p(json.dumps(cfg, indent=2, sort_keys=True))
        return 0
    _p(f"FALCONAge {cfg['falconage']}")
    _p(f"  registry     {cfg['registry_version']}  ({cfg['n_clocks']} clocks)")
    t = cfg["clocks_by_tier"]
    _p(f"  tiers        A {t['A']} ship - B {t['B']} need a traced source - "
       f"C {t['C']} scaffold only")
    _p(f"  numpy        {cfg['numpy']}")
    _p(f"  torch        {cfg['torch'] or 'not installed (CPU path needs none)'}")
    _p(f"  devices      {', '.join(cfg['devices'])}")
    if cfg.get("cuda_devices"):
        _p(f"  cuda         {cfg['cuda_version']}: {', '.join(cfg['cuda_devices'])}")
    return 0


def cmd_clocks(args) -> int:
    import falconage as fa

    reg = fa.registry.load()

    if args.action == "list":
        clocks = list(reg)
        if args.tier:
            clocks = [c for c in clocks if c.availability == args.tier]
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

        _p(f"{'id':<26}{'tier':<6}{'gen':<12}{'scale':<22}{'n':>7}  name")
        for c in clocks:
            _p(f"{c.id:<26}{c.availability:<6}{c.generation:<12}{c.scale_type:<22}"
               f"{c.n_features or '?':>7}  {c.name[:40]}")
        _p(f"\n{len(clocks)} clock(s)")
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
        _p(f"  availability  tier {c.availability}")
        cs = c.coefficient_source
        _p(f"  provenance    {cs.provenance or 'unrecorded'}")
        _p(f"  traced        {cs.primary_source_traced}")
        if c.availability == "C":
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

    res = fa.score(data, clocks=clocks, device=args.device, dtype=args.dtype,
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
    return 0 if rep.verdict != "unsupported" else 3


def cmd_report(args) -> int:
    """Read, check, score, quantify, interpret, and write one HTML file.

    The command a laboratory runs. Everything else in this CLI is a piece of
    it, exposed separately for people who want the pieces.
    """
    import falconage as fa

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    data = fa.read(args.input)
    if data.modality == "dna_methylation":
        data = fa.prepare(data)
    _p(f"read {data.n_samples} sample(s) x {data.n_features} feature(s)"
       + (f" on {data.platform}" if data.platform else ""))

    if data.modality == "dna_methylation":
        report = fa.qc(data)
        report.per_sample.to_csv(out / "qc_per_sample.csv")
        for w in report.warnings:
            _p(f"  QC: {w}")

    clocks = args.clocks
    if clocks not in ("compatible", "all"):
        clocks = [c.strip() for c in clocks.split(",") if c.strip()]
    res = fa.score(data, clocks=clocks, min_coverage=args.min_coverage)
    _p(f"scored {res.scores.shape[1]} clock(s); {len(res.skipped)} skipped")

    se = conf = cons = None
    try:
        se = fa.technical_se(res, data)
        se.se.to_csv(out / "technical_se.csv")
        se.diagnostics.to_csv(out / "reliability_diagnostics.csv")
    except Exception as exc:                      # noqa: BLE001
        _p(f"  technical_se unavailable: {exc}")
    try:
        conf = fa.conformal_interval(res, level=args.level)
        conf.to_csv(out / "conformal_interval.csv", index=False)
    except Exception as exc:                      # noqa: BLE001
        _p(f"  conformal interval unavailable: {exc}")
    if args.group_col and args.group_col in res.obs.columns:
        try:
            cons = fa.consensus(res, args.group_col, reference=args.reference)
            _p(f"  consensus: {cons.verdict}")
        except Exception as exc:                  # noqa: BLE001
            _p(f"  consensus unavailable: {exc}")

    res.write(out)
    res.interpretation().to_csv(out / "interpretation.csv")
    res.evidence().to_csv(out / "evidence.csv", index=False)

    if not args.no_figures:
        from falconage import plot as fplot

        acc = None
        if "age" in res.obs.columns:
            try:
                acc = fa.acceleration(res, method="residual")
            except Exception:                     # noqa: BLE001
                acc = None
        w = fplot.save_all(res, out / "figures", data=data, acc=acc,
                           group=args.group_col, se=se, conformal=conf,
                           consensus=cons)
        _p(f"  {len(w)} figure(s)")

    from falconage.report import write_report

    html = write_report(res, out / "report.html", group=args.group_col)
    _p(f"\nwrote {html}")
    return 0

# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="falconage",
        description="Multiomic biological age and aging clock scoring.",
        epilog="Docs: https://bhagesh-h.github.io/FALCONAge/",
    )
    ap.add_argument("--version", action="version", version=f"falconage {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("config", help="what this installation resolved to")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_config)

    p = sub.add_parser("clocks", help="browse the registry")
    p.add_argument("action", choices=["list", "info", "cite"])
    p.add_argument("clock", nargs="?")
    p.add_argument("--tier", choices=["A", "B", "C"])
    p.add_argument("--data-type", dest="data_type")
    p.add_argument("--generation")
    p.add_argument("--search")
    p.add_argument("--untraced", action="store_true",
                   help="only clocks with no established primary source")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--style", default="plain", choices=["plain", "bibtex"])
    p.set_defaults(fn=cmd_clocks)

    p = sub.add_parser("download", help="fetch public data by accession")
    p.add_argument("accession")
    p.add_argument("--want", choices=["matrix", "suppl", "both"])
    p.add_argument("--outdir")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("preprocess", help="raw or public data to a scoreable .h5ad")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(fn=cmd_preprocess)

    p = sub.add_parser("score", help="score a dataset")
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--clocks", default="compatible")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default=None, choices=[None, "float64", "float32"])
    p.add_argument("--min-coverage", dest="min_coverage", type=float, default=0.8)
    p.add_argument("--imputation", default="reference",
                   choices=["reference", "mean", "none"])
    p.add_argument("--show-skipped", dest="show_skipped", action="store_true")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("bench", help="run the AA1/AA2 benchmark")
    p.add_argument("--input", required=True)
    p.add_argument("--outdir")
    p.add_argument("--clocks", default="compatible")
    p.add_argument("--condition-col", dest="condition_col", default="condition")
    p.add_argument("--control", default="HC")
    p.add_argument("--dataset-col", dest="dataset_col")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("cache", help="inspect or clear the download cache")
    p.add_argument("action", choices=["ls", "rm"])
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_cache)

    # The verb that runs before any array does. It needs no data file, which is
    # the point: it is the earliest moment this package can be useful.
    p = sub.add_parser("power", help="how many samples to see an effect")
    p.add_argument("--clock", required=True)
    p.add_argument("--effect", type=float, required=True,
                   help="the difference worth detecting, in the clock's own unit")
    p.add_argument("--sd", type=float, default=None,
                   help="population SD; measured from --pilot when given")
    p.add_argument("--pilot", default=None,
                   help="a scored pilot dataset, to measure sd and the ICC from")
    p.add_argument("--icc", type=float, default=None)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--target-power", type=float, default=0.80)
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--min-coverage", type=float, default=0.8)
    p.set_defaults(fn=cmd_power)

    p = sub.add_parser("consensus",
                       help="does a group difference hold up across clocks?")
    p.add_argument("input")
    p.add_argument("--group-col", default="condition")
    p.add_argument("--reference", default=None)
    p.add_argument("--clocks", default="compatible")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-coverage", type=float, default=0.8)
    p.add_argument("--outdir", default=None)
    p.set_defaults(fn=cmd_consensus)

    p = sub.add_parser("report",
                       help="one command: read, QC, score, quantify, write HTML")
    p.add_argument("input")
    p.add_argument("--outdir", default="falconage_report")
    p.add_argument("--clocks", default="compatible")
    p.add_argument("--group-col", default=None)
    p.add_argument("--reference", default=None)
    p.add_argument("--level", type=float, default=0.90)
    p.add_argument("--min-coverage", type=float, default=0.8)
    p.add_argument("--no-figures", action="store_true")
    p.set_defaults(fn=cmd_report)

    return ap


def main(argv: list[str] | None = None) -> int:
    from ..core.errors import FalconError
    from ..core.logging import configure

    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "cmd", None):
        ap.print_help()
        return 0
    configure()
    try:
        return int(args.fn(args) or 0)
    except FalconError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
