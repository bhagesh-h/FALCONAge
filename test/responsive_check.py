#!/usr/bin/env python3
"""Check the rendered site at phone, tablet and desktop widths.

WHY THIS EXISTS. The site was built and reviewed on a desktop, and a rule added
to put the sidebar logo above the blurb -- ``#quarto-sidebar { display: flex }``
-- silently broke every phone. An id selector is specificity (1,0,0); the rule
Quarto uses to hide the sidebar behind a toggle is ``.collapse:not(.show) {
display: none }`` at (0,2,0). The id won, so the sidebar stayed painted on top
of the article at every width. Nothing in the build failed, the desktop looked
right, and the first report of it was a person reading the site on a phone.

Reading CSS is how that bug was introduced. This measures instead:

1. **Horizontal overflow.** Any element painted wider than the viewport, with
   the ancestor chain, skipping anything inside a deliberate scroll container.
   A page that scrolls sideways on a phone fails.
2. **Overlap.** Text boxes painted on top of each other. Uses
   ``getClientRects()`` rather than ``getBoundingClientRect()``: a wrapped
   inline element's bounding box spans every line it touches, so two spans on
   consecutive lines of one code block register as colliding when nothing of
   the sort is happening. That false positive produced 79 phantom hits the
   first time this was run.
3. **The sidebar actually collapses**, and still opens, and is still in the
   right order when it does. That is the specific regression above.

Usage
-----
    quarto render                       # docs/_site must exist
    python test/responsive_check.py [--site docs/_site]

Needs playwright and a chromium build. Neither belongs on a developer's
machine, so run it in a container:

    docker run --rm -v "$PWD:/work" -w /work \\
      mcr.microsoft.com/playwright/python:v1.49.0-noble \\
      sh -c "pip install -q playwright==1.49.0 && python test/responsive_check.py"

Exits non-zero on any failure, so it can gate a release.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 320 is the narrowest phone still in use, 360 the commonest Android, 390 a
# current iPhone, 768 the tablet edge, 1280 a laptop. The interesting ones are
# 768 and below -- that is where Quarto swaps the docked column for a drawer,
# and where a desktop-only review never looks.
WIDTHS = [320, 360, 390, 768, 1280]

PAGES = [
    "index.html",
    "clocks.html",
    "gpu.html",
    "science.html",
    "architecture.html",
    "guide/FALCONAge.html",
    "guide/clocks.html",
    "reference/index.html",
]

OVERFLOW = """
() => {
  const docW = document.documentElement.clientWidth;
  const items = [];
  const seen = new Set();
  const inScroller = (el) => {
    for (let p = el.parentElement; p; p = p.parentElement) {
      const ov = getComputedStyle(p).overflowX;
      if (ov === 'auto' || ov === 'scroll') return true;
    }
    return false;
  };
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    const over = Math.round(r.right - docW);
    if (over <= 1 && r.left >= -1) continue;
    if (inScroller(el)) continue;
    const cls = (typeof el.className === 'string' && el.className)
      ? '.' + el.className.trim().split(/\\s+/).slice(0, 2).join('.') : '';
    const key = el.tagName.toLowerCase() + cls;
    if (seen.has(key)) continue;
    seen.add(key);
    items.push({what: key.slice(0, 60), over,
                text: (el.textContent || '').trim().slice(0, 40)});
  }
  items.sort((a, b) => b.over - a.over);
  return {scrollW: document.documentElement.scrollWidth, docW,
          items: items.slice(0, 5)};
}
"""

OVERLAP = """
() => {
  const boxes = [];
  for (const el of document.querySelectorAll('body *')) {
    if (el.children.length) continue;
    if (!(el.textContent || '').trim()) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.opacity === '0') continue;
    if (cs.position === 'fixed' || cs.position === 'absolute') continue;
    for (const r of el.getClientRects()) {
      if (r.width < 4 || r.height < 4) continue;
      boxes.push({el, r, t: (el.textContent || '').trim()});
    }
  }
  const hits = [];
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i], b = boxes[j];
      if (a.el === b.el) continue;
      if (a.el.contains(b.el) || b.el.contains(a.el)) continue;
      const ox = Math.min(a.r.right, b.r.right) - Math.max(a.r.left, b.r.left);
      const oy = Math.min(a.r.bottom, b.r.bottom) - Math.max(a.r.top, b.r.top);
      if (ox > 4 && oy > 4) {
        hits.push({a: a.t.slice(0, 28), b: b.t.slice(0, 28),
                   overlap: Math.round(ox) + 'x' + Math.round(oy)});
        if (hits.length > 5) return hits;
      }
    }
  }
  return hits;
}
"""

SIDEBAR = """
() => {
  const bar = document.querySelector('#quarto-sidebar');
  if (!bar) return {present: false};
  const r = bar.getBoundingClientRect();
  const logo = bar.querySelector('img.sidebar-logo');
  const head = bar.querySelector('.quarto-sidebar-header');
  const lr = logo ? logo.getBoundingClientRect() : null;
  const hr = head ? head.getBoundingClientRect() : null;
  return {
    present: true,
    visible: r.width > 0 && r.height > 0,
    width: Math.round(r.width),
    display: getComputedStyle(bar).display,
    logoH: lr ? Math.round(lr.height) : null,
    logoW: lr ? Math.round(lr.width) : null,
    // The whole reason the flex rule exists: logo must sit above the blurb.
    logoAboveBlurb: (lr && hr) ? lr.top < hr.top : null,
  };
}
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site", default="docs/_site", type=Path)
    args = ap.parse_args(argv)

    site = args.site.resolve()
    if not (site / "index.html").exists():
        print(f"{site}/index.html not found -- run `quarto render` in docs/ first")
        return 1

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(__doc__.split("Needs playwright")[1].split("Exits")[0].strip())
        return 1

    failures: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for width in WIDTHS:
            page = browser.new_page(viewport={"width": width, "height": 820})
            n_over = n_lap = 0

            for rel in PAGES:
                f = site / rel
                if not f.exists():
                    continue
                page.goto(f.as_uri(), wait_until="load")
                page.wait_for_timeout(200)

                o = page.evaluate(OVERFLOW)
                if o["scrollW"] > o["docW"] + 1 or o["items"]:
                    n_over += 1
                    failures.append(f"{rel} @{width}px scrolls sideways "
                                    f"({o['docW']} -> {o['scrollW']}px)")
                    for it in o["items"][:3]:
                        failures.append(f"      +{it['over']}px {it['what']} "
                                        f"{it['text']!r}")

                laps = page.evaluate(OVERLAP)
                if laps:
                    n_lap += 1
                    failures.append(f"{rel} @{width}px has "
                                    f"{len(laps)} overlapping text pair(s)")
                    for h in laps[:3]:
                        failures.append(f"      {h['overlap']} "
                                        f"{h['a']!r} over {h['b']!r}")

            # The sidebar, on the landing page, at this width.
            page.goto((site / "index.html").as_uri(), wait_until="load")
            page.wait_for_timeout(200)
            s = page.evaluate(SIDEBAR)
            desktop = width >= 992

            if not s.get("present"):
                failures.append(f"@{width}px: no #quarto-sidebar in the page")
            elif desktop:
                if not s["visible"]:
                    failures.append(f"@{width}px: sidebar should be docked and is not")
                elif s["logoAboveBlurb"] is False:
                    failures.append(f"@{width}px: logo is below the blurb")
            elif s["visible"]:
                failures.append(
                    f"@{width}px: sidebar is painted ({s['width']}px wide, "
                    f"display:{s['display']}) instead of collapsing behind the "
                    "toggle -- this is the rule that overlays the article")

            state = ("docked" if desktop and s.get("visible")
                     else "collapsed" if not s.get("visible") else "PAINTED")
            logo = (f"logo {s['logoW']}x{s['logoH']}px"
                    if s.get("logoH") else "logo hidden")
            print(f"@{width:>5}px  {state:<9} {logo:<22} "
                  f"overflow {n_over}/{len(PAGES)}  overlap {n_lap}/{len(PAGES)}")
            page.close()
        browser.close()

    if failures:
        print(f"\n{len(failures)} problem(s):")
        for f in failures:
            print(f"  {f}")
        return 1

    print(f"\nclean: {len(PAGES)} pages x {len(WIDTHS)} widths, "
          "no sideways scroll, no overlapping text, sidebar collapses below 992px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
