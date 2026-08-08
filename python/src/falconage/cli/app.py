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
