"""Raw IDAT to scoreable betas: the chain the clocks were actually fitted on.

At first FALCONAge asked users to normalise elsewhere -- sesame or minfi --
and read the resulting betas. That was honest and it was the single largest
reason somebody with their own arrays could not use this package. This module
closes it.

THE CHAIN, AND WHY THE ORDER IS LOAD-BEARING

    addresses -> in-band and out-of-band signal   (needs the manifest)
              -> pOOBAH detection p-values         (uses OOB as the null)
              -> noob background correction        (uses OOB as the background)
              -> dye-bias correction               (equalises the two channels)
              -> betas                             (M / (M + U + offset))

Detection is computed **before** background correction, because pOOBAH's null is
the out-of-band distribution as measured, and a background-corrected OOB is no
longer a null -- it has had the thing it is a null for subtracted from it.
Running them the other way round gives every probe an optimistic p-value.

Dye bias comes **after** noob and before betas, because the two channels'
scale difference is multiplicative on signal and additive background would
otherwise be scaled with it.

WHAT OUT-OF-BAND MEANS, SINCE EVERYTHING HERE RESTS ON IT. An Infinium type I
probe has two beads, and both are read in the same colour. The *other* colour at
those same addresses measures nothing real -- it is the array's background,
measured on the same chip, in the same run, at tens of thousands of addresses.
Type II probes have no out-of-band signal, which is why the null comes from the
type I probes and is applied to all of them.

IMPLEMENTED FROM THE PAPERS, NOT PORTED

* pOOBAH -- Zhou, Triche, Laird & Shen, *SeSAMe*, Nucleic Acids Res 2018;46:e123.
* noob -- Triche, Weisenberger, Van Den Berg, Laird & Siegmund, Nucleic Acids
  Res 2013;41:e90, with the normal-exponential deconvolution of Silver, Ritchie
  & Smyth, Biostatistics 2009;10:352.
* Huber's M-estimator for the background location and scale, as minfi uses, from
  Huber & Ronchetti, *Robust Statistics*, 2nd ed., §4.
* Nonlinear dye bias -- the quantile-matching form in SeSAMe.

None of these is a port. There is no sesame or minfi in this environment to port
from; each is written from its published definition and checked against a
property the definition implies, and then the whole chain is checked against
published betas for the same physical samples (see the corpus tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..core.container import FalconData
from ..core.errors import DataError

__all__ = ["RawSignal", "idat_to_betas", "read_idat_dir"]

#: Added to the denominator of every beta. 100 is Illumina's own convention and
#: what minfi and sesame both use; it stops a probe with almost no signal in
#: either channel from producing a beta of 0 or 1 out of pure noise.
BETA_OFFSET = 100.0

#: pOOBAH's default. Zhou et al. use 0.05; probes above it are not detected
#: above background and become NaN rather than a number.
DETECTION_P = 0.05


def _huber(x: np.ndarray, k: float = 1.5, tol: float = 1e-6,
           max_iter: int = 50) -> tuple[float, float]:
    """Huber's M-estimate of location and scale.

    The background is the bulk of the out-of-band distribution, and its upper
    tail is not background at all -- it is cross-hybridisation and the odd
    genuinely bright bead. A mean and an SD are pulled by that tail; Huber's
    estimator winsorises it at ``k`` scaled deviations and iterates. minfi uses
    ``MASS::huber`` here for the same reason.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise DataError("not enough out-of-band signal to estimate a background")
    mu = float(np.median(x))
    s = float(np.median(np.abs(x - mu))) * 1.4826
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(x)) or 1.0
    for _ in range(max_iter):
        y = np.clip(x, mu - k * s, mu + k * s)
        mu_new = float(np.mean(y))
        # The consistency factor for the winsorised variance at this k, so the
        # estimate is unbiased for a Gaussian rather than merely robust.
        from scipy.stats import norm

        beta = 2 * (k ** 2 * (1 - norm.cdf(k)) - k * norm.pdf(k)) + \
            (2 * norm.cdf(k) - 1)
        s_new = float(np.sqrt(np.mean((y - mu_new) ** 2) / beta))
        if abs(mu_new - mu) < tol * max(abs(mu), 1.0) and \
                abs(s_new - s) < tol * max(s, 1.0):
            mu, s = mu_new, s_new
            break
        mu, s = mu_new, max(s_new, 1e-8)
    return mu, s


def _normexp_signal(x: np.ndarray, mu: float, sigma: float,
                    alpha: float) -> np.ndarray:
    """Expected signal given observed = normal background + exponential signal.

    The posterior mean of the exponential component, from Silver, Ritchie &
    Smyth (2009). This is the correction limma applies as ``normexp.signal`` and
    that noob applies per channel:

        mu_sf = x - mu - sigma^2 / alpha
        E[s | x] = mu_sf + sigma^2 * phi(mu_sf/sigma) / Phi(mu_sf/sigma)

    Computed through ``log_ndtr`` rather than as a ratio of a density to a tail
    probability. For a probe well below background the ratio is a very small
    number over a very small number and evaluates to ``0/0``; in logs it is a
    difference of two well-conditioned quantities and stays finite. That case is
    not rare -- it is every undetected probe on the array.
    """
    from scipy.special import log_ndtr
    from scipy.stats import norm

    sigma = max(float(sigma), 1e-8)
    alpha = max(float(alpha), 1e-8)
    mu_sf = np.asarray(x, dtype=np.float64) - mu - (sigma ** 2) / alpha
    z = mu_sf / sigma
    log_ratio = norm.logpdf(z) - log_ndtr(z)
    return np.maximum(mu_sf + sigma * np.exp(log_ratio), 0.0)


@dataclass
class RawSignal:
    """One sample's in-band and out-of-band intensities, probe by probe.

    The intermediate every step in the chain reads and writes. Keeping it
    explicit rather than passing a beta vector around is what lets detection be
    computed on the uncorrected out-of-band signal and background correction on
    the same numbers afterwards.
    """

    meth: pd.Series                 # in-band methylated, by probe
    unmeth: pd.Series               # in-band unmethylated, by probe
    probe_type: pd.Series           # "I" or "II"
    channel: pd.Series              # "Grn"/"Red" for type I, "" for type II
    oob_grn: np.ndarray             # green at type-I-red addresses
    oob_red: np.ndarray             # red at type-I-green addresses
    platform: str
    sample_id: str
    detection_p: pd.Series | None = None
    steps: list[str] = field(default_factory=list)

    def betas(self, *, offset: float = BETA_OFFSET,
              detection_p: float | None = DETECTION_P) -> pd.Series:
        """``M / (M + U + offset)``, with undetected probes set to NaN.

        NaN rather than a number, and this is the point of computing detection
        at all: a probe that did not rise above background has no measurement,
        and a beta of 0.43 from two noise readings is indistinguishable from a
        real 0.43 once it is in the matrix.
        """
        m = self.meth.to_numpy(dtype=np.float64)
        u = self.unmeth.to_numpy(dtype=np.float64)
        b = m / (m + u + offset)
        if detection_p is not None and self.detection_p is not None:
            b = np.where(self.detection_p.to_numpy() > detection_p, np.nan, b)
        return pd.Series(b, index=self.meth.index, name=self.sample_id)

    def summary(self) -> dict:
        return {
            "sample_id": self.sample_id, "platform": self.platform,
            "n_probes": int(len(self.meth)),
            "n_type_i": int((self.probe_type == "I").sum()),
            "n_type_ii": int((self.probe_type == "II").sum()),
            "n_oob_grn": int(self.oob_grn.size), "n_oob_red": int(self.oob_red.size),
            "steps": ", ".join(self.steps) or "none",
            "median_detection_p": (None if self.detection_p is None
                                   else round(float(self.detection_p.median()), 6)),
            "frac_undetected": (None if self.detection_p is None else
                                round(float((self.detection_p > DETECTION_P).mean()), 6)),
        }

    def __repr__(self) -> str:  # pragma: no cover - display only
        s = self.summary()
        return (f"RawSignal({s['sample_id']}, {s['platform']}, "
                f"{s['n_probes']} probes, steps: {s['steps']})")


def _split_signal(addr: pd.Series, grn: np.ndarray, red: np.ndarray,
                  man: pd.DataFrame, sample_id: str, platform: str) -> RawSignal:
    """Resolve bead addresses into per-probe methylated and unmethylated signal."""
    # Both sides cast to int64 before the lookup. IDATs store addresses as
    # int32 and the manifests as strings that become float64 the moment one
    # column has a blank in it, and reindexing an int index with float keys
    # matches nothing at all -- silently, and with the same symptom as a wrong
    # platform.
    pos = pd.Series(np.arange(len(addr)), index=addr.to_numpy().astype(np.int64))
    pos = pos[~pos.index.duplicated(keep="first")]

    a = pos.reindex(man["address_a"].fillna(-1).to_numpy().astype(np.int64))
    b = pos.reindex(man["address_b"].fillna(-1).to_numpy().astype(np.int64))
    have = a.notna().to_numpy().copy()
    is_i = (man["type"] == "I").to_numpy()
    # A type I probe needs both of its addresses on the chip; a type II needs one.
    have &= np.where(is_i, b.notna().to_numpy(), True)
    if not have.any():
        raise DataError(
            f"{sample_id}: none of the {platform} manifest's addresses are in "
            "this IDAT. The platform is almost certainly wrong.")

    man = man[have]
    ai = a[have].to_numpy(dtype=np.int64)
    bi = np.where(b[have].notna().to_numpy(), b[have].fillna(0).to_numpy(), 0
                  ).astype(np.int64)
    is_i = (man["type"] == "I").to_numpy()
    grn_i = (man["channel"] == "Grn").to_numpy() & is_i
    red_i = (man["channel"] == "Red").to_numpy() & is_i
    is_ii = ~is_i

    meth = np.empty(len(man), dtype=np.float64)
    unmeth = np.empty(len(man), dtype=np.float64)

    # Type I: both beads read in the probe's own colour. B is methylated.
    meth[grn_i] = grn[bi[grn_i]]
    unmeth[grn_i] = grn[ai[grn_i]]
    meth[red_i] = red[bi[red_i]]
    unmeth[red_i] = red[ai[red_i]]
    # Type II: one bead, green is methylated and red is unmethylated.
    meth[is_ii] = grn[ai[is_ii]]
    unmeth[is_ii] = red[ai[is_ii]]

    # Out of band: the colour a type I probe does NOT read in, at its own two
    # addresses. Nothing real is measured there, which is exactly what makes it
    # a background sample taken on this chip in this run.
    oob_red = np.concatenate([red[ai[grn_i]], red[bi[grn_i]]])
    oob_grn = np.concatenate([grn[ai[red_i]], grn[bi[red_i]]])

    idx = pd.Index(man.index, name="feature_id")
    return RawSignal(
        meth=pd.Series(meth, index=idx, name=sample_id),
        unmeth=pd.Series(unmeth, index=idx, name=sample_id),
        probe_type=pd.Series(man["type"].to_numpy(), index=idx),
        channel=pd.Series(man["channel"].to_numpy(), index=idx),
        oob_grn=oob_grn, oob_red=oob_red,
        platform=platform, sample_id=sample_id, steps=["decoded"])


def poobah(sig: RawSignal) -> RawSignal:
    """Detection p-values against the out-of-band background (Zhou 2018).

    For each probe, the probability that a signal at least this bright would be
    seen from background alone, read off the empirical distribution of the
    out-of-band intensities in the relevant channel. A type I probe is compared
    against its own channel's null in both beads; a type II probe's methylated
    bead is green and its unmethylated bead is red, so it is compared against
    both nulls.

    The p-value is the *smaller* of the two tail probabilities, i.e. the more
    optimistic bead. A probe is detected if either of its beads rose above
    background, which is the right question: a fully methylated probe has almost
    no unmethylated signal by construction, and requiring both would fail every
    probe at an extreme.
    """
    def tail(null: np.ndarray, x: np.ndarray) -> np.ndarray:
        null = np.sort(np.asarray(null, dtype=np.float64))
        if null.size == 0:
            return np.zeros_like(x)
        # 1 - ECDF, computed by search rather than by building a step function.
        return 1.0 - np.searchsorted(null, x, side="right") / null.size

    m = sig.meth.to_numpy(dtype=np.float64)
    u = sig.unmeth.to_numpy(dtype=np.float64)
    t = sig.probe_type.to_numpy()
    ch = sig.channel.to_numpy()

    p = np.ones(len(m))
    grn_i = (t == "I") & (ch == "Grn")
    red_i = (t == "I") & (ch == "Red")
    ii = t == "II"

    if grn_i.any():
        p[grn_i] = np.minimum(tail(sig.oob_grn, m[grn_i]), tail(sig.oob_grn, u[grn_i]))
    if red_i.any():
        p[red_i] = np.minimum(tail(sig.oob_red, m[red_i]), tail(sig.oob_red, u[red_i]))
    if ii.any():
        p[ii] = np.minimum(tail(sig.oob_grn, m[ii]), tail(sig.oob_red, u[ii]))

    sig.detection_p = pd.Series(p, index=sig.meth.index, name="detection_p")
    sig.steps = [*sig.steps, "poobah"]
    return sig


def noob(sig: RawSignal, *, offset: float = 15.0) -> RawSignal:
    """Normal-exponential background correction using the out-of-band signal.

    Triche et al. (2013). Each channel's background is estimated from that
    channel's out-of-band intensities -- measured on this chip, in this run --
    and the observed signal is deconvolved into background plus an exponential
    signal component. ``offset`` is added afterwards, as in the paper, to keep
    corrected values away from zero.

    Requires detection to have been computed already, and says so rather than
    silently reordering: pOOBAH's null is the *uncorrected* out-of-band
    distribution. Correcting first subtracts the background from the very
    numbers that define what background looks like, and every probe then
    appears to clear a bar that has been lowered underneath it.
    """
    if sig.detection_p is None:
        raise DataError(
            "run poobah() before noob().\n"
            "  Detection is measured against the uncorrected out-of-band "
            "distribution. Background-correcting first removes the background "
            "from its own null, and every probe passes.")

    t = sig.probe_type.to_numpy()
    ch = sig.channel.to_numpy()
    m = sig.meth.to_numpy(dtype=np.float64).copy()
    u = sig.unmeth.to_numpy(dtype=np.float64).copy()

    grn_i = (t == "I") & (ch == "Grn")
    red_i = (t == "I") & (ch == "Red")
    ii = t == "II"

    for null, meth_mask, unmeth_mask in (
            (sig.oob_grn, grn_i | ii, grn_i),      # green: type I-Grn both, type II meth
            (sig.oob_red, red_i, red_i | ii),      # red:   type I-Red both, type II unmeth
    ):
        mu, sigma = _huber(null)
        obs = np.concatenate([m[meth_mask], u[unmeth_mask]])
        alpha = max(float(np.mean(obs)) - mu, 10.0)
        if meth_mask.any():
            m[meth_mask] = _normexp_signal(m[meth_mask], mu, sigma, alpha) + offset
        if unmeth_mask.any():
            u[unmeth_mask] = _normexp_signal(u[unmeth_mask], mu, sigma, alpha) + offset

    sig.meth = pd.Series(m, index=sig.meth.index, name=sig.sample_id)
    sig.unmeth = pd.Series(u, index=sig.unmeth.index, name=sig.sample_id)
    sig.steps = [*sig.steps, "noob"]
    return sig


def dye_bias(sig: RawSignal, *, method: str = "nonlinear") -> RawSignal:
    """Equalise the two colour channels. **Off by default -- see the warning.**

    The green and red dyes do not incorporate or fluoresce equally, so a type II
    probe -- whose methylated bead is green and unmethylated bead is red --
    carries a tilt that has nothing to do with methylation. Type I probes are
    the usual reference, because each reads both of its beads in a single
    channel.

    .. warning::

       This step is **not validated** and is not run by :func:`idat_to_betas`
       unless asked for. Measured on the corpus's EPIC v1 IDATs against the
       published betas for the same physical samples, it moves the median beta
       by **+0.10 to +0.12** -- far more than a dye correction should. The cause
       is visible in the data: on that chip the red channel runs about twice as
       hot as the green (median out-of-band 1,171 against 415), and mapping red
       onto green therefore halves every type II unmethylated signal, which
       raises every type II beta.

       That is a real limitation of the assumption, not an arithmetic slip.
       Matching the two channels' *distributions* presumes type I green probes
       and type I red probes measure the same underlying quantity; on this array
       they do not -- their median betas are 0.05 and 0.14. A correct correction
       needs the normalisation control probes, whose addresses are not in the
       core-columns manifest FALCONAge fetches.

       Left in, opt-in and documented, because the arithmetic is right for the
       assumption it makes and somebody with control-probe addresses can supply
       a better reference. Turned on by default it would silently make results
       worse, which is the failure this package exists to avoid.

    ``"nonlinear"`` matches quantiles; ``"linear"`` scales by the median ratio.
    """
    t = sig.probe_type.to_numpy()
    ch = sig.channel.to_numpy()
    grn_i, red_i, ii = (t == "I") & (ch == "Grn"), (t == "I") & (ch == "Red"), t == "II"

    m = sig.meth.to_numpy(dtype=np.float64).copy()
    u = sig.unmeth.to_numpy(dtype=np.float64).copy()

    ref = np.sort(np.concatenate([m[grn_i], u[grn_i]]))     # green, from type I
    src = np.sort(np.concatenate([m[red_i], u[red_i]]))     # red, from type I
    if ref.size < 100 or src.size < 100:
        raise DataError(
            "too few type I probes in one channel to estimate dye bias; this "
            "array does not support the correction")

    if method == "linear":
        scale = float(np.median(ref)) / max(float(np.median(src)), 1e-8)
        for arr, mask in ((m, red_i), (u, red_i), (u, ii)):
            arr[mask] = arr[mask] * scale
    elif method == "nonlinear":
        # Quantile map: where a red value sits in the red distribution, put it
        # at the same place in the green one. Interpolated, so it is monotone
        # and does not collapse ties.
        q_src = np.linspace(0, 1, src.size)
        q_ref = np.linspace(0, 1, ref.size)

        def remap(x):
            q = np.interp(x, src, q_src)
            return np.interp(q, q_ref, ref)

        m[red_i] = remap(m[red_i])
        u[red_i] = remap(u[red_i])
        u[ii] = remap(u[ii])            # type II unmethylated is read in red
    else:
        raise DataError(f"dye_bias method={method!r}; expected linear or nonlinear")

    sig.meth = pd.Series(m, index=sig.meth.index, name=sig.sample_id)
    sig.unmeth = pd.Series(u, index=sig.unmeth.index, name=sig.sample_id)
    sig.steps = [*sig.steps, f"dye_bias:{method}"]
    return sig


def idat_to_betas(grn: str | Path, red: str | Path, *, platform: str | None = None,
                  sample_id: str | None = None, correct: bool = True,
                  detection_p: float | None = DETECTION_P,
                  dye: str | None = None, raw: bool = False):
    """One Grn/Red IDAT pair to a beta vector, through the full chain.

    Parameters
    ----------
    platform
        ``None`` detects it from the bead-address count and refuses rather than
        guessing when nothing matches within 10%.
    correct
        Run pOOBAH detection and noob background correction. ``False`` still
        computes detection -- it costs nothing and an undetected probe is not a
        measurement whatever else is done -- and then returns uncorrected betas.
    detection_p
        Probes above this become NaN. ``None`` keeps them, which is a decision
        to score numbers that are not measurements.
    dye
        ``None`` (default), ``"linear"`` or ``"nonlinear"``. Off by default; see
        the warning on :func:`dye_bias` for the measurement that decided that.
    raw
        Return the :class:`RawSignal` instead of the betas, for anyone who wants
        the intermediate.
    """
    from ..io.methylation import read_idat_pair
    from .manifest import detect_manifest_platform, load_manifest

    pair = read_idat_pair(grn, red)
    addr = pd.Series(pair["illumina_ids"])
    plat = platform or detect_manifest_platform(len(addr))
    man = load_manifest(plat)
    sid = sample_id or Path(grn).name.split("_Grn")[0].split(".")[0]

    sig = _split_signal(addr, np.asarray(pair["grn"], dtype=np.float64),
                        np.asarray(pair["red"], dtype=np.float64), man, sid, plat)
    sig = poobah(sig)
    if correct:
        sig = noob(sig)
    if dye:
        sig = dye_bias(sig, method=dye)
    return sig if raw else sig.betas(detection_p=detection_p)


def read_idat_dir(paths: Sequence[str | Path] | str | Path, *,
                  platform: str | None = None, obs: pd.DataFrame | None = None,
                  correct: bool = True, detection_p: float | None = DETECTION_P,
                  dye: str | None = None) -> FalconData:
    """Every Grn/Red pair in a directory, as one scoreable matrix.

    Pairs are matched on the filename stem before ``_Grn``/``_Red``, which is
    Illumina's own convention and what GEO distributes. The run manifest records
    which array manifest resolved the addresses and which steps ran, because a
    beta matrix is a function of both.
    """
    if isinstance(paths, (str, Path)) and Path(paths).is_dir():
        found = sorted(Path(paths).glob("*_Grn.idat*"))
        pairs = [(g, Path(str(g).replace("_Grn", "_Red"))) for g in found]
    else:
        items = [Path(p) for p in paths]  # type: ignore[union-attr]
        grns = sorted(p for p in items if "_Grn" in p.name)
        pairs = [(g, Path(str(g).replace("_Grn", "_Red"))) for g in grns]

    pairs = [(g, r) for g, r in pairs if r.exists()]
    if not pairs:
        raise DataError(
            "no Grn/Red IDAT pairs found.\n"
            "  Pairs are matched on the stem before _Grn / _Red, which is what "
            "Illumina writes and what GEO distributes. A lone Grn file is not a "
            "sample.")

    cols, notes = {}, []
    plat = platform
    for g, r in pairs:
        sig = idat_to_betas(g, r, platform=plat, correct=correct,
                            detection_p=detection_p, dye=dye, raw=True)
        plat = plat or sig.platform
        cols[sig.sample_id] = sig.betas(detection_p=detection_p)
        notes.append(sig.summary())

    X = pd.DataFrame(cols).T
    from .manifest import manifest_record

    d = FalconData(X=X, obs=obs if obs is not None else pd.DataFrame(index=X.index),
                   modality="dna_methylation", platform=plat,
                   uns={"idat_manifest": manifest_record(plat),
                        "idat_samples": notes,
                        "pipeline": notes[0]["steps"] if notes else "",
                        "detection_p_threshold": detection_p})
    return d
