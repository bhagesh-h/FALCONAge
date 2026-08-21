# Image prompts

Explanatory diagrams for FALCONAge. Every prompt below is one image; **the
heading is the filename**. Generate `## three-quantities.png` and save it as
`images/three-quantities.png`.

These are *concept* diagrams. The 26 figures under
[`test/output_figures/gallery/`](../test/output_figures/gallery/) are already
generated from real runs and are not to be re-drawn here -- those are data, these
are explanation. If a diagram would show numbers that come from a run, put the
number in the caption, not in the picture, so the two cannot disagree.

---

## How to use this file

1. Pick a heading. The heading text, exactly, is the output filename.
2. Read the **house style** below before generating anything. It is not
   decoration: the same images are used on a site with a light and a dark theme,
   in a PDF, and in Notion, and an image that only works on one of those has to
   be made again.
3. Generate, save into `images/`, and say which ones are done. Nothing is wired
   into the documentation until the file exists.

---

## House style

**Palette.** Use these and nothing else. They are Okabe-Ito, chosen so the
figures stay readable for the ~8% of men with red-green colour vision
deficiency, and every generated figure in this project already uses them. A
diagram in different colours will look like it came from somewhere else.

| Role | Hex | Use for |
|---|---|---|
| Primary | `#0072B2` | the first thing, the default, "decelerated / younger" |
| Secondary | `#D55E00` | the second thing, "accelerated / older", refusals |
| Green | `#009E73` | pass, permitted, healthy |
| Amber | `#E69F00` | warning, marginal |
| Purple | `#CC79A7` | highlight, a third category |
| Sky | `#56B4E9` | control group, background category |
| Neutral | `#7F7F7F` | anything unemphasised |
| Rule | `#404040` | axes, reference lines, arrows, thresholds |

**Backgrounds and both themes.** Transparent background by default. Ink must
read on white *and* on `#1B1B1B`, so:

- Never use pure black or pure white for lines or text. Use `#404040` for rules
  and `#0072B2`/`#D55E00` for emphasis; both survive either theme.
- If a diagram needs a filled panel, produce **two files**: `name.png` on white
  and `name-dark.png` on `#1B1B1B`, same geometry, same palette.
- No drop shadows, no gradients, no glow, no 3-D. They read as noise at print
  size and they break at small widths.

**Text.** Image generators garble text, so every prompt lists the exact strings
to render and there are never many. Rules:

- Only the strings the prompt names. No invented labels, no lorem, no captions
  baked into the image.
- Sans-serif, sentence case, no smaller than about 2.5% of the image height.
- If a label would need more than four words, it belongs in the page caption
  instead. Send it back rather than shrinking the type.
- Numbers are allowed only where the prompt gives them explicitly.

**Geometry.** 1600 × 1000 px unless the prompt says otherwise (that is the
3:2-ish shape the site's content column and the PDF's text block both take
without cropping). Leave 5% padding on every side; Notion crops tight images.
PNG, 8-bit, transparent where possible.

**Accuracy.** These illustrate a package that refuses to guess, so a diagram
that implies something false is worse than no diagram. Two specific traps:

- Do not draw a methylation beta outside 0 to 1.
- Do not draw a clock output as "age" when the prompt says it is a rate, a
  division count or a hazard. That distinction is the point of several of these.

---

# The images

Ordered by where they appear in the argument, not by priority. The **Used in**
line for each says which of the three destinations it serves.

---

## three-quantities.png

**Concept.** The single most common confusion in the field: "how old" is three
different questions with three different answers and three different units.

**Draw.** Three panels side by side, equal size, each with a simple visual
metaphor and one label:

- Left, `#0072B2`: a calendar or milestone line. Label: `Chronological age`.
  Sub-label: `years since birth`.
- Middle, `#D55E00`: a worn versus new component of the same object. Label:
  `Biological age`. Sub-label: `wear for that age`.
- Right, `#009E73`: a speedometer or a slope. Label: `Pace of aging`.
  Sub-label: `years per year`.

Under all three, one rule in `#404040` with the text
`Different units. Not interchangeable.`

**Used in.** Site landing page, PDF chapter 1, Notion §1.

---

## clock-anatomy.png

**Concept.** Every clock in the catalogue is the same four-part pipeline, and
reproducing a clock means reproducing all four in order. Most published
disagreements are in the last part, not the weights.

**Draw.** A left-to-right chain of four labelled boxes with arrows, plus a
final output pill:

1. `Feature set` (`#0072B2`) -- a grid of small dots with a handful highlighted
2. `Preprocess` (`#56B4E9`) -- a small transform glyph
3. `Weights + intercept` (`#E69F00`) -- a bar-weighted sum glyph
4. `Postprocess` (`#CC79A7`) -- a curve mapping a line onto a different scale
5. Output pill (`#009E73`) -- label `Score + its scale`

Under box 4, a small `#D55E00` caret with the label
`most disagreements live here`.

**Used in.** Site architecture page, PDF chapter 5, Notion §5.1.

---

## clock-generations.png

**Concept.** Generations are defined by their *training target*, not their date,
and the target determines what the output means.

**Draw.** A vertical stack of five rows. Each row: a coloured tag on the left,
the training target in the middle, what the output means on the right.

- `First` (`#0072B2`) -- `Chronological age` -- `An age; the residual is the signal`
- `Second` (`#D55E00`) -- `Mortality-weighted composite` -- `A hazard in year-like units`
- `Third` (`#009E73`) -- `Longitudinal rate` -- `A ratio: years per year`
- `Causal` (`#CC79A7`) -- `Damaging vs adaptive` -- `Two scores, no shared origin`
- `Mitotic` (`#E69F00`) -- `Division count` -- `Divisions, not time`

**Used in.** Site guide, PDF chapter 5, Notion §5.2.

---

## methylation-chemistry.png

**Concept.** Four ways to read a methyl group, and what each one can and cannot
distinguish. Bisulfite cannot separate 5mC from 5hmC; the alternatives can.

**Draw.** Four columns, each with a simplified DNA strand carrying a marked
cytosine and an arrow to what the sequencer reads:

- `Bisulfite` (`#0072B2`) -- unmethylated C becomes T; note under it:
  `5mC + 5hmC together`
- `oxBS` (`#56B4E9`) -- note: `5mC alone`
- `EM-seq / TAPS` (`#009E73`) -- note: `enzymatic, DNA intact`
- `Nanopore` (`#CC79A7`) -- a current trace glyph; note: `reads the base directly`

**Used in.** Site science page, PDF chapter 3, Notion §3.

---

## array-probe-types.png

**Concept.** One chip carries two chemistries, they do not produce the same beta
distribution, and that is why a correction step exists.

**Draw.** Top half: Type I, two beads one colour channel; Type II, one bead two
channels. Use `#0072B2` and `#D55E00` for the two channels. Bottom half: two
density curves on one axis from 0 to 1 -- Type I bimodal with peaks near the
ends, Type II visibly compressed toward the middle. One arrow between them
labelled `BMIQ`.

Labels: `Type I`, `Type II`, `Beta`, `BMIQ`.

**Used in.** Site science page, PDF chapter 3, Notion §3.3 and §4.3.

---

## idat-to-beta-pipeline.png

**Concept.** The order of the preprocessing chain is not arbitrary, and every
step is a decision that has to be recorded.

**Draw.** A vertical flow of six stages, each a rounded box with a one-line
label, connected by arrows, with a thin `#7F7F7F` margin note beside each:

1. `Raw intensities` -- note `two channels per probe`
2. `Detection` -- note `pOOBAH: did we measure it?`
3. `Background + dye` -- note `noob`
4. `Type I/II` -- note `BMIQ`
5. `Masking` -- note `cross-reactive, SNP, sex`
6. `Beta matrix` -- note `samples x CpGs`

**Used in.** Site guide, PDF chapter 4, Notion §4.1.

---

## scale-types-and-legal-operations.png

**Concept.** The output's scale decides which operations are defined. This is
the idea the whole refusal system rests on.

**Draw.** A matrix: rows are scales, columns are operations, cells are a green
`#009E73` tick or a vermillion `#D55E00` cross. No other ink.

Rows: `Years`, `Years, no fixed zero`, `Pace ratio`, `Divisions`,
`Log hazard`, `Proportion`.
Columns: `Correlate`, `Difference`, `Residual`, `Acceleration`.

Ticks: everything under `Correlate`. `Difference` for all but log hazard.
`Residual` for the two year scales only. `Acceleration` for `Years` only.

**Used in.** Site landing page and architecture page, PDF chapter 5, Notion §5.3.

---

## coverage-vs-coefficient-mass.png

**Concept.** Counting probes is the wrong measure of whether a clock can be
scored. What matters is how much of the model's weight the present probes carry.

**Draw.** Two horizontal bars, same length, stacked.

- Top bar, labelled `Probes present`, `#009E73` filling 96% of it.
- Bottom bar, labelled `Model weight present`, `#E69F00` filling 61% of it.
- A dashed `#404040` vertical line at 80% across both, labelled `Floor`.

The top bar clears the line, the bottom does not. That contrast is the entire
image.

**Used in.** Site landing page, PDF chapter 4, Notion §4.5.

---

## tissue-mismatch.png

**Concept.** Correlation is not agreement. Two tissues from the same people can
correlate well and still disagree by years.

**Draw.** Left: a scatter of ~90 points, `#56B4E9`, clearly correlated along a
line that is visibly *not* the identity line; draw the identity line dashed in
`#404040` for contrast. Right: the same samples as paired points with connecting
lines, showing a consistent vertical offset.

Labels: `Buffy coat`, `Saliva`, `Same 91 people`.

**Used in.** Site landing page, PDF chapter 2, Notion §2.4.

---

## mitotic-vs-chronological.png

**Concept.** A division counter and an age estimator measure different things.
A high-turnover tissue is mitotically older at the same chronological age, and
that is the clock working.

**Draw.** Two tracks running left to right against a shared `Time` axis in
`#404040`. Top track `#0072B2`, few cell-division glyphs, label `Slow turnover`.
Bottom track `#D55E00`, many division glyphs, label `Fast turnover`. A vertical
line at the right labelled `Same chronological age`, with the two tracks having
accumulated visibly different counts.

**Used in.** Site guide, PDF chapter 2 and 6, Notion §2.3.

---

## epitoc2-inversion.png

**Concept.** epiTOC2 is not a weighted sum. Each site carries two parameters and
yields its own estimate of divisions; the reported number is their mean over the
sites the dataset actually has.

**Draw.** Three sites stacked, each showing: measured beta, minus a fetal ground
state, divided by a per-site rate, giving a per-site estimate. Then a brace
collecting the three into `mean`, then `x2`, then the output pill
`Divisions per stem cell` in `#009E73`.

Labels: `Measured`, `Ground state`, `Rate per division`, `mean`, `x2`,
`Divisions per stem cell`. One `#7F7F7F` note: `divisor = sites present`.

**Used in.** Site architecture page, PDF chapter 6, Notion §6.4.

---

## uncertainty-decomposition.png

**Concept.** A score is not a point. Part of its spread is the biology and part
is the assay, and the ratio is what decides whether a study can see anything.

**Draw.** A horizontal bar broken into two segments: a wide `#0072B2` segment
labelled `Between-sample spread` and a narrower `#D55E00` segment labelled
`Assay noise`. Below it, a second bar where the two are nearly equal, labelled
`Cohort too narrow to detect anything`. To the right of each, a small bracketed
interval glyph.

**Used in.** Site landing page, PDF chapter 7, Notion §7.3.

---

## tiers-and-provenance.png

**Concept.** Whether a clock can be run is a question about rights and evidence,
not about quality, and the catalogue says which.

**Draw.** Three columns as stacked cards:

- `Tier A` (`#009E73`) -- `Scores offline` -- a file glyph with a tick
- `Tier B` (`#E69F00`) -- `No traced source` -- a file glyph with a question mark
- `Tier C` (`#D55E00`) -- `Licence restricted` -- a file glyph with a lock

Beneath, one horizontal chain in `#404040`:
`Paper` → `Supplement` → `Checksum` → `Score`, with the note
`copied from another package` struck through in `#D55E00`.

**Used in.** Site clock catalogue, PDF chapter 9, Notion §9.2.

---

## refusal-gates.png

**Concept.** A score passes through gates on the way out, and each gate is a
measurement rather than an opinion.

**Draw.** A left-to-right pipeline with five gates. Each gate is a narrow
vertical bar; a passing path continues in `#009E73`, a refused path exits
downward in `#D55E00` with a short label.

Gates and their exit labels: `Specimen` → `wrong tissue`; `Coverage` →
`too little model`; `Scale` → `operation undefined`; `Units` → `not declared`;
`Weights` → `not distributable`. Final output pill: `Score + manifest`.

**Used in.** Site landing page and architecture page, PDF chapter 9, Notion §9.4.

---

## falconage-dataflow.png

**Concept.** How the package is put together, at the altitude someone needs
before reading any of it.

**Draw.** Four stacked bands, each with three or four boxes:

- `Read` (`#56B4E9`) -- `IDAT`, `Betas`, `Clinical`, `Proteomic / RNA`
- `Prepare` (`#0072B2`) -- `Normalise`, `Mask`, `Align to clock`
- `Score` (`#E69F00`) -- `Registry`, `Model class`, `Refusals`
- `Report` (`#009E73`) -- `Scores`, `Uncertainty`, `Manifest`, `Figures`

One vertical arrow down the left in `#404040` labelled `One numerical core`,
spanning all four bands, with a small note `Python and R, identical results`.

**Used in.** Site architecture page, PDF chapter 9, Notion §9.1.

---

## aa1-aa2-benchmark.png

**Concept.** Correlating with age proves almost nothing. The test is whether the
residual separates cases from controls, and whether a clock's own bias is
discounted before it gets credit.

**Draw.** Left: two overlapping distributions, `#56B4E9` controls and `#D55E00`
cases, on an axis labelled `Age acceleration`, with the case distribution
shifted right; label `AA2: has controls`. Right: one distribution shifted from a
dashed zero line in `#404040`; label `AA1: no controls`. Below both, a short bar
labelled `Total` with a `#7F7F7F` slice cut out of it labelled `bias discount`.

**Used in.** Site science page, PDF chapter 8, Notion §8.1.

---

## Wiring an image in once it exists

Nothing renders until it is referenced. When a file lands in `images/`:

- **Site and PDF** -- reference it from the relevant `.qmd` with
  `![Caption](../images/name.png)`, and add `images/**` to the `resources:` list
  in `docs/_quarto.yml` so it is copied into `_site`. `docs/build_docs.py`
  writes that config, so the change goes in the generator, not the output.
- **Notion** -- images are uploaded to the page rather than linked from the
  repository; a raw GitHub URL renders but breaks if the repo ever moves.
- **Check both themes** before committing. The site has a light and a dark mode
  and the PDF is neither.
