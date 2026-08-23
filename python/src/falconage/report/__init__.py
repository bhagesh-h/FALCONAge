"""HTML report assembly.

Two writers, for two different documents. `write_report` is the compact
self-contained page that survives being emailed. `write_quarto_report` is the
full results document: every table searchable and collapsible, every figure
carrying its interpretation, and the clocks grouped by what they were trained to
predict rather than by name.
"""

from .html import write_report
from .quarto import CATEGORIES, categorise, write_quarto_report

__all__ = ["CATEGORIES", "categorise", "write_quarto_report", "write_report"]
