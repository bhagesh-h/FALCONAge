"""Which question does a clock answer?

A reader arrives with a question ("who is at risk of dying sooner?"), not a
clock name, and the registry answers it from each clock's declared `predicts`
field. Two places need the same answer: the routing table on
``docs/clocks/choosing.qmd``, and the message a user sees when a licensed clock
cannot be scored, which names the bundled clocks that answer the same question
instead. They share this module so they cannot disagree.

The patterns are over `predicts` rather than exact strings. The registry holds
more than eighty distinct `predicts` values, and matching literally would mean
editing this list every time a clock is catalogued. A value that matches nothing
fails the documentation build (``docs/build_catalogue.py``) rather than
vanishing from the page that exists to help people find clocks.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

#: The question, and a pattern over `predicts` that answers it. **Ordered: first
#: match wins**, so the specific rules come before the general ones --
#: "physical-fitness biological age" has to be caught by the organ-system rule
#: before the plain "biological age" one takes it.
QUESTIONS: list[tuple[str, str]] = [
    ("How old does this sample look?",
     r"^(chronological|relative) age$|retroelement methylation age"),
    ("How old is this newborn, gestationally?",
     r"gestational age"),
    ("How fast is this person aging?",
     r"pace of aging|intervention-responsive"),
    ("Which organ system is aging fastest?",
     r"[- ]system biological age|multisystem|physical-fitness biological age"),
    ("Who is at risk of dying sooner, or is frailer?",
     r"mortality|phenotypic age|^biological age$|frailty|healthspan|lifespan"
     r"|frailty index|electronic medical record age"
     r"|physiological dysregulation|intrinsic capacity|time to death"),
    ("Is damage separable from adaptation?",
     r"(damaging|adaptive|causal) epigenetic age"),
    ("How much has this tissue divided?",
     r"mitotic|replicative|divisions|proliferation|passage age|senescence"),
    ("What is the blood's cell composition?",
     r"proportion|cell composition"),
    ("How long are the telomeres?",
     r"telomere"),
    ("What is this person exposed to, or how do they live?",
     r"smoking|alcohol|body mass|BMI|body fat|cholesterol|waist|hip|VO2max"
     r"|grip strength|gait speed|educational attainment|stress|physical activity"
     r"|diet|exposure"),
    ("What is a specific protein or lab value likely to be?",
     r"C-reactive protein|GDF-15|PAI-1|TIMP-1|adrenomedullin|beta-2-microglobulin"
     r"|beta-2 microglobulin|cystatin C|interleukin|leptin|hemoglobin A1c"
     r"|growth differentiation factor|plasminogen activator|TIMP metallopeptidase"
     r"|smoking pack-years|methylation$|methylation score"),
    ("Is a named disease more likely?",
     r"disease|cancer|carcinoma|Alzheimer|depressive|syndrome"),
    ("What is the sample's chromosomal sex, or its species?",
     r"chromosome|^sex$|species"),
]


def question_for(predicts: Iterable[str]) -> str | None:
    """The first question any of these `predicts` values answers, or None."""
    for p in predicts:
        s = str(p).strip()
        for question, pattern in QUESTIONS:
            if re.search(pattern, s, flags=re.IGNORECASE):
                return question
    return None
