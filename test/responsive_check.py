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
    "gallery.html",
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

# Duplicated chrome. Geometry cannot see this: two search boxes that each fit
# the viewport and do not touch each other pass the overflow and overlap tests
# both, and the page still shows a reader two of everything. That is precisely
# what happened -- those two checks reported clean while the mobile header
# carried two menu toggles and two search boxes.
#
# Counted with the sidebar drawer OPEN as well as closed, because the second
# set only appears once something is expanded.
CHROME = """
() => {
  const vis = (el) => {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden'
           && cs.display !== 'none' && cs.opacity !== '0';
  };
  const count = (sel) => [...document.querySelectorAll(sel)].filter(vis).length;
  return {
    // Duplicate ids are invalid HTML before they are a layout problem, and
    // Quarto emits id="quarto-search" for both the navbar and the sidebar.
    searchIds: document.querySelectorAll('#quarto-search').length,
    searchVisible: count('#quarto-search, .sidebar-search, input[type=search]'),
    toggles: count('.navbar-toggler, .quarto-btn-toggle'),
    logos: count('img.sidebar-logo, img.navbar-logo'),
    // Which one, and whether the file actually resolved at this page depth --
    // a logo that 404s from guide/ still counts as one element.
    logoWhich: [...document.querySelectorAll('img.sidebar-logo, img.navbar-logo')]
      .filter(vis)
      .map(e => e.className.split(' ')[0] + ' ' +
                Math.round(e.getBoundingClientRect().height) + 'px' +
                (e.complete && e.naturalWidth > 0 ? '' : ' BROKEN'))
      .join(', ') || 'none',
    brandTitles: count('.navbar-title, .sidebar-title'),
    // Is the site's own name being cut off?
    //
    // This check reported clean at 320px while the navbar read "FALCO...".
    // Neither an overflow test nor an overlap test can see it: the clipped
    // element is inside its parent and touching nothing. Quarto puts
    // `overflow:hidden; text-overflow:ellipsis` on `.navbar-brand`, so the
    // failure is silent by construction and the only evidence is
    // scrollWidth > clientWidth on the element doing the clipping.
    clippedBrand: [...document.querySelectorAll('.navbar-brand, .navbar-title, .sidebar-title')]
      .filter(vis)
      .filter(e => e.scrollWidth > e.clientWidth + 1 && e.clientWidth > 0)
      .map(e => (e.className || 'brand').toString().split(' ')[0] +
                ' needs ' + e.scrollWidth + 'px, has ' + e.clientWidth + 'px')
      .join('; '),
    // HOW MANY SITE-CONTENTS LISTS ARE ON SCREEN AT ONCE.
    //
    // The site declares its reading order twice: as the sidebar spine, and as
    // a collapsed "Contents" menu in the navbar for the widths where that
    // column is hidden. Exactly one of the two must be reachable at any width.
    // Two is the duplication the sidebar was previously emptied to avoid; zero
    // is a site with no navigation, which is what happens if the CSS hiding
    // the sidebar on a phone lands without the navbar copy being shown.
    //
    // Counted, not reviewed: both failures pass an overflow test and an
    // overlap test, which is exactly how the earlier two-menu bug survived.
    // The navbar copy counts as present when its toggle is visible, because on
    // a phone the list itself is collapsed until tapped.
    navLists: (vis(document.querySelector('#quarto-sidebar .sidebar-menu-container'))
                 ? 1 : 0)
            + (vis(document.querySelector('#nav-menu-contents')) ? 1 : 0),
  };
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
            n_over = n_lap = n_img = 0

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

                # Every image the page asks for has to have arrived.
                #
                # The gallery PNGs are tracked once, under
                # test/output_figures/gallery/, and staged into docs/figures/
                # by docs/build_gallery.py before the render -- the staged copy
                # is deliberately not in git, so if that step is skipped or
                # renamed the site publishes with 26 empty boxes and nothing
                # else here would notice. A browser knows: naturalWidth is 0
                # for an <img> that failed to load.
                broken = page.evaluate("""() => [...document.images]
                    .filter(i => i.complete && i.naturalWidth === 0)
                    .map(i => i.getAttribute('src'))""")
                if broken:
                    n_img += 1
                    failures.append(f"{rel} @{width}px has {len(broken)} image(s) "
                                    f"that did not load")
                    for src in broken[:3]:
                        failures.append(f"      {src}")

                laps = page.evaluate(OVERLAP)
                if laps:
                    n_lap += 1
                    failures.append(f"{rel} @{width}px has "
                                    f"{len(laps)} overlapping text pair(s)")
                    for h in laps[:3]:
                        failures.append(f"      {h['overlap']} "
                                        f"{h['a']!r} over {h['b']!r}")

            # The sidebar and the chrome count, on the landing page.
            page.goto((site / "index.html").as_uri(), wait_until="load")
            page.wait_for_timeout(200)
            s = page.evaluate(SIDEBAR)
            desktop = width >= 992

            # Closed, then with everything expanded -- the duplicates only
            # appear once a drawer is open.
            chrome = page.evaluate(CHROME)
            for sel in (".navbar-toggler", ".quarto-btn-toggle"):
                try:
                    el = page.query_selector(sel)
                    if el and el.is_visible():
                        el.click(timeout=2000)
                        page.wait_for_timeout(500)
                except Exception:
                    pass
            opened = page.evaluate(CHROME)

            if chrome["searchIds"] > 1:
                failures.append(
                    f"@{width}px: id=\"quarto-search\" appears "
                    f"{chrome['searchIds']} times -- duplicate id, and two "
                    "search boxes. Turn off search on one of navbar/sidebar.")
            for state, c in (("closed", chrome), ("expanded", opened)):
                if c["searchVisible"] > 1:
                    failures.append(f"@{width}px ({state}): "
                                    f"{c['searchVisible']} search boxes visible")
                if c["toggles"] > 1:
                    failures.append(f"@{width}px ({state}): {c['toggles']} menu "
                                    "toggles visible -- a reader cannot tell "
                                    "which is the menu")
                if c["logos"] > 1:
                    failures.append(f"@{width}px ({state}): {c['logos']} logos "
                                    f"visible ({c['logoWhich']})")
                if c["logos"] == 0:
                    failures.append(
                        f"@{width}px ({state}): no logo anywhere. The sidebar "
                        "owns it above 992px and the navbar below; if both are "
                        "hidden the page has lost its mark entirely.")
                if "BROKEN" in c["logoWhich"]:
                    failures.append(f"@{width}px ({state}): logo does not load "
                                    f"at this page depth ({c['logoWhich']})")
                if c["brandTitles"] > 1:
                    failures.append(f"@{width}px ({state}): "
                                    f"{c['brandTitles']} site titles visible")
                if c["clippedBrand"]:
                    failures.append(
                        f"@{width}px ({state}): the site name is being cut off "
                        f"({c['clippedBrand']}). Shrink the wordmark at this "
                        "width rather than letting it ellipsise -- a truncated "
                        "name is not a name.")
                # Two contents lists at once is the duplication to prevent, and
                # it is a failure in either state. One is correct. Zero is only
                # correct in the closed state on a phone, where the navbar menu
                # is behind the hamburger by design, so that case is checked
                # below against the toggle rather than here.
                if c["navLists"] > 1:
                    failures.append(
                        f"@{width}px ({state}): the reading order is stated "
                        f"twice on one screen ({c['navLists']} contents lists). "
                        "The sidebar spine serves widths at or above 992px and "
                        "the navbar 'Contents' menu serves the rest; only one "
                        "of the two may be on screen.")
                if state == "expanded" and c["navLists"] == 0:
                    failures.append(
                        f"@{width}px (expanded): no contents list even with the "
                        "menu open, so the site cannot be navigated at this "
                        "width.")
                if state == "closed" and c["navLists"] == 0 and c["toggles"] == 0:
                    failures.append(
                        f"@{width}px: no contents list and no menu toggle. The "
                        "sidebar is hidden here and nothing stands in for it.")

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
                    f"display:{s['display']}) instead of being hidden -- below "
                    "992px it duplicates the navbar's chrome and overlays the "
                    "article")

            state = ("docked" if desktop and s.get("visible")
                     else "hidden" if not s.get("visible") else "PAINTED")
            print(f"@{width:>5}px  sidebar {state:<7} logo {opened['logoWhich']:<22} "
                  f"search {opened['searchVisible']}  menus {opened['toggles']}  "
                  f"overflow {n_over}/{len(PAGES)}  overlap {n_lap}/{len(PAGES)}  "
                  f"broken-img {n_img}/{len(PAGES)}")
            page.close()
        browser.close()

    if failures:
        print(f"\n{len(failures)} problem(s):")
        for f in failures:
            print(f"  {f}")
        return 1

    print(f"\nclean: {len(PAGES)} pages x {len(WIDTHS)} widths -- no sideways "
          "scroll, no overlapping text, no unloaded image, the site name not "
          "cut off, one search box, one menu, one title, and exactly one "
          "contents list at every width: the sidebar spine at or above 992px, "
          "the navbar menu below it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
