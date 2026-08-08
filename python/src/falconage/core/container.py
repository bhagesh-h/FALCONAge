"""FalconData: samples x features, plus the metadata a clock needs.

WHY NOT JUST A DataFrame. Three things have to travel together and stay in
step: the matrix, the per-sample annotation (age, sex, tissue, condition), and
the declaration of what the numbers *are* -- beta values, clinical chemistry in
named units, RRBS coverage-weighted methylation. A bare DataFrame carries the
first and loses the other two, and every clock needs all three to refuse
politely instead of computing something meaningless.

WHY NOT AnnData AS THE ONLY CONTAINER. It is the right on-disk format and
``to_anndata``/``from_anndata`` are here, but requiring it at import time drags
in h5py for a user with a 40-row clinical CSV. AnnData is an optional extra;
the in-memory object is plain pandas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .errors import DataError

Modality = Literal["dna_methylation", "clinical_chemistry", "rrbs", "transcriptomics"]


@dataclass
class FalconData:
    """A scoreable dataset.

    Attributes
    ----------
    X
        samples x features. Row index is the sample id, column index is the
        exact feature identifier the clocks match on (``cg`` probe ids for
        arrays, ``chr:pos`` for RRBS, canonical marker names for chemistry).
    obs
        Per-sample annotation, indexed identically to ``X``. Conventional
        columns are ``age``, ``sex``, ``tissue``, ``condition`` -- conventional,
        not required, and each is only needed by the analyses that use it.
    modality
        What the numbers are. Governs which clocks are compatible and which
        preprocessing has already been applied.
    units
        For clinical chemistry only: canonical marker name to unit string.
        Empty for methylation, where beta is dimensionless by construction.
    platform
        For methylation: ``27K``, ``450K``, ``EPICv1``, ``EPICv2``, ``MSA``,
        ``MammalMethylChip40``, or ``None`` when unknown.
    """

    X: pd.DataFrame
    obs: pd.DataFrame
    modality: Modality
    units: dict[str, str] = field(default_factory=dict)
    platform: str | None = None
    #: Which organism the samples came from. Defaults to human because most
    #: data is, but it is checked rather than assumed: the mammalian array
    #: carries 96% of Horvath2013's CpGs, so a zebra scores at high coverage
    #: and returns a confident number from a clock fitted on people. Coverage
    #: is not validity, and this is the field that lets score() say so.
    species: str = "Homo sapiens"
    #: Free-form provenance carried through to the run manifest.
    uns: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.X, pd.DataFrame):
            raise DataError("X must be a pandas DataFrame of samples x features")
        if self.obs is None or len(self.obs) == 0:
            self.obs = pd.DataFrame(index=self.X.index)
        if not self.X.index.equals(self.obs.index):
            # Reindexing silently would let a mislabelled obs table attach the
            # wrong age to every sample, which no downstream check would catch.
            common = self.X.index.intersection(self.obs.index)
            if len(common) == 0:
                raise DataError(
                    "X and obs share no sample ids.\n"
                    f"  X[:3]   = {list(self.X.index[:3])}\n"
                    f"  obs[:3] = {list(self.obs.index[:3])}"
                )
            self.X = self.X.loc[common]
            self.obs = self.obs.loc[common]
        if self.X.columns.has_duplicates:
            dup = self.X.columns[self.X.columns.duplicated()].unique()[:5]
            raise DataError(
                f"duplicate feature ids in X: {list(dup)}...\n"
                "  On EPIC v2 this usually means replicate probes were not "
                "aggregated; see falconage.preprocess.aggregate_replicate_probes."
            )

    # -- shape ------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def sample_ids(self) -> pd.Index:
        return self.X.index

    @property
    def features(self) -> pd.Index:
        return self.X.columns

    def __repr__(self) -> str:  # pragma: no cover - display only
        miss = float(self.X.isna().to_numpy().mean()) * 100
        plat = f", {self.platform}" if self.platform else ""
        return (f"FalconData({self.n_samples} samples x {self.n_features} features, "
                f"{self.modality}{plat}, {miss:.1f}% missing)")

    def summary(self) -> pd.Series:
        return pd.Series({
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "modality": self.modality,
            "platform": self.platform or "unknown",
            "species": self.species,
            "missing_fraction": float(self.X.isna().to_numpy().mean()),
            "obs_columns": ", ".join(map(str, self.obs.columns)),
        })

    # -- subsetting -------------------------------------------------------
    def subset(self, samples=None, features=None) -> FalconData:
        X = self.X
        obs = self.obs
        if samples is not None:
            X, obs = X.loc[samples], obs.loc[samples]
        if features is not None:
            X = X.loc[:, [f for f in features if f in X.columns]]
        return FalconData(X=X, obs=obs, modality=self.modality, units=dict(self.units),
                          platform=self.platform, species=self.species, uns=dict(self.uns))

    def coverage(self, features: list[str]) -> float:
        """Fraction of ``features`` present and not entirely missing.

        Presence alone is not enough: GEO series matrices routinely carry a
        probe column that is NaN for every sample, and counting that as covered
        is how a clock reports 80% coverage of features it cannot actually see.
        """
        if not features:
            return 0.0
        present = [f for f in features if f in self.X.columns]
        if not present:
            return 0.0
        usable = self.X[present].notna().any(axis=0).sum()
        return float(usable) / len(features)

    # -- interchange ------------------------------------------------------
    def to_anndata(self):
        """Convert to AnnData. Needs the ``anndata`` extra."""
        try:
            import anndata as ad
        except ImportError as exc:  # pragma: no cover
            raise DataError("AnnData support needs anndata: pip install anndata") from exc
        a = ad.AnnData(X=self.X.to_numpy(dtype=np.float64), obs=self.obs.copy())
        a.var_names = self.X.columns.astype(str)
        a.obs_names = self.X.index.astype(str)
        a.uns["falconage"] = {"modality": self.modality, "platform": self.platform,
                              "units": self.units, "species": self.species, **self.uns}
        return a

    @classmethod
    def from_anndata(cls, a) -> FalconData:
        meta = dict(a.uns.get("falconage", {}))
        X = pd.DataFrame(np.asarray(a.X, dtype=np.float64),
                         index=a.obs_names.astype(str), columns=a.var_names.astype(str))
        return cls(X=X, obs=a.obs.copy(),
                   modality=meta.pop("modality", "dna_methylation"),
                   units=meta.pop("units", {}) or {},
                   platform=meta.pop("platform", None),
                   species=meta.pop("species", "Homo sapiens"), uns=meta)

    def write_h5ad(self, path: str | Path) -> Path:
        """Write the interchange format both languages read."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.to_anndata().write_h5ad(p)
        return p

    @classmethod
    def read_h5ad(cls, path: str | Path) -> FalconData:
        try:
            import anndata as ad
        except ImportError as exc:  # pragma: no cover
            raise DataError("reading .h5ad needs anndata: pip install anndata") from exc
        return cls.from_anndata(ad.read_h5ad(Path(path)))
