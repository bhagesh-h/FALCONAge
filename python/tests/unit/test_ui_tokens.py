"""The interface tokens: one set of colours for the site and both reports."""

from __future__ import annotations

import re

import pytest

from falconage.plot import spec

TEXT_ROLES = ("ink", "muted", "accent", "pass", "warn", "stop")


def _luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    """WCAG 2.1 contrast ratio."""
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_contrast_matches_the_wcag_reference_values():
    assert contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast("#777777", "#ffffff") == pytest.approx(4.48, abs=0.01)


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("role", TEXT_ROLES)
def test_every_text_colour_meets_aa_on_both_surfaces(mode, role):
    """WCAG 2.1 AA: 4.5:1 for body text. Warning text in vermillion was 3.9:1."""
    ui = spec.ui(mode)
    for surface in ("bg", "panel"):
        assert contrast(ui[role], ui[surface]) >= 4.5, (mode, role, surface)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_the_brand_colour_is_only_large_text_safe(mode):
    """3:1 is the AA floor for large text, which is where the wordmark sits."""
    ui = spec.ui(mode)
    assert contrast(ui["brand"], ui["bg"]) >= 3.0


def test_the_reports_spell_no_colour_of_their_own():
    """Their CSS names roles; the hex values live in colorscheme.yaml. White is
    allowed, as the ground figures are drawn on."""
    from falconage.report import html as report_html
    from falconage.report import quarto as report_quarto

    for rules in (report_html.RULES, report_quarto.TABLE_RULES):
        found = set(re.findall(r"#[0-9a-fA-F]{3,8}\b", rules)) - {"#ffffff"}
        assert not found, found
    css = report_html.css()
    for mode in ("light", "dark"):
        for value in spec.ui(mode).values():
            assert value in css
