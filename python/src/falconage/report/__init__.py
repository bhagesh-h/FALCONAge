"""HTML report assembly.

Two documents. `write_report` is the compact self-contained page that survives
being emailed. `write_quarto_report` is the full record of a run: every file in
its output directory, placed under the step of the analysis that produced it,
in step order, with every table searchable and every figure carrying what to
look for. `run_report` runs the analysis and writes both.
"""

from .html import write_report
from .outputs import OUTPUTS, STEPS
from .quarto import CATEGORIES, categorise, write_quarto_report
from .run import run_report

__all__ = ["CATEGORIES", "OUTPUTS", "STEPS", "categorise", "run_report",
           "write_quarto_report", "write_report"]
