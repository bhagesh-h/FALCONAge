"""Accession routing, asserted without touching the network.

Every test here is offline on purpose. What can go wrong in this module and
actually bite someone is not a flaky archive -- it is an accession sent to the
wrong API, which returns a 404 that reads like "the dataset does not exist"
rather than "FALCONAge asked the wrong service". That is pure string matching
and deserves to be pinned.
"""

from __future__ import annotations

import pytest

from falconage.core.errors import DownloadError
from falconage.download import resolve_source


@pytest.mark.parametrize("accession,source", [
    ("GSE182991", "geo_series"),
    ("gse182991", "geo_series"),
    ("GSM5514000", "geo_sample"),
    ("E-MTAB-9756", "arrayexpress"),
    ("SRP123456", "sra"),
    ("SRR9876543", "sra"),
    ("PRJNA123456", "sra"),
    ("ERR1234567", "sra"),
    ("PXD012345", "pride"),
    ("MTBLS1234", "metabolights"),
    ("TCGA-BRCA", "gdc"),
    ("TARGET-AML", "gdc"),
    ("7fc1a1b2-3c4d-5e6f-7a8b-9c0d1e2f3a4b", "gdc"),
    ("10.6084/m9.figshare.12345678", "figshare"),
    ("10.5281/zenodo.1234567", "zenodo"),
    ("owner/dataset", "huggingface"),
    ("https://example.org/betas.csv", "url"),
])
def test_accession_routing(accession, source):
    assert resolve_source(accession) == source


def test_figshare_and_zenodo_dois_do_not_collide():
    """Both are bare DOIs; only the prefix separates them.

    Sending a Figshare DOI to the Zenodo API returns a 404 that looks like a
    missing record, which is the least helpful possible failure.
    """
    assert resolve_source("10.6084/m9.figshare.999") == "figshare"
    assert resolve_source("10.5281/zenodo.999") == "zenodo"


def test_a_gdc_project_is_not_mistaken_for_a_hugging_face_repo():
    """`owner/name` is a permissive pattern and GDC ids must be tested first."""
    assert resolve_source("TCGA-LUAD") == "gdc"
    assert resolve_source("CPTAC-3") == "gdc"


def test_unknown_accession_lists_what_is_supported():
    with pytest.raises(DownloadError) as exc:
        resolve_source("not-an-accession-!!")
    msg = str(exc.value)
    for token in ("GSE", "PXD", "MTBLS", "GDC", "Hugging Face"):
        assert token in msg
    # And says why the credentialed archives are absent rather than omitting them.
    assert "dbGaP" in msg and "credentials" in msg


def test_credentialed_archives_refuse_with_a_route_forward():
    import falconage as fa

    with pytest.raises(DownloadError) as exc:
        fa.download("synapse")
    msg = str(exc.value)
    assert "synapseclient" in msg
    assert "falconage.io.read" in msg, "say what to do once the files are local"


def test_pride_refuses_an_unfiltered_project_rather_than_starting_it():
    """A PRIDE project is mostly raw instrument output no clock reads.

    The refusal is the feature: without an extension filter this would begin a
    multi-gigabyte transfer that nobody sized. Checked by signature so the test
    stays offline.
    """
    import inspect

    from falconage.download import pride

    sig = inspect.signature(pride)
    assert "extensions" in sig.parameters
    assert sig.parameters["extensions"].default == ()


def test_sra_returns_a_run_table_and_does_not_pretend_to_score():
    """SRA holds reads. Alignment and methylation calling are not in scope."""
    from falconage.download import DownloadResult, sra

    assert "run_table" in DownloadResult.__dataclass_fields__
    doc = sra.__doc__ or ""
    assert "alignment" in doc.lower()
