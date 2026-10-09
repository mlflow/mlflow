<!--
Adapted from the `artifact-diagramming` skill bundled with Claude Code. That skill does not load in
triage runs, so the parts that apply to a static PNG are kept here.
-->

# Drawing the visual aid

A visual aid in a triage comment lets a maintainer see a mechanism without assembling it from
prose: which call leads to the failing line, what each component holds or waits on, how a value
changes between layers. If a sentence says it faster, write the sentence instead.

## What to draw

- **Draw the mechanism, not its name.** A box labeled "scheduler" says less than the arrow showing
  what it submits every minute and where that row ends up. Show the parts the bug depends on (the
  call that opens a second session, the row that is never cleaned up) and leave out the rest.
- **Use real names.** Label boxes with the actual functions, classes, tables, and settings from the
  code, so a reader can search for them.
- **Label the arrows.** An unlabeled arrow means "related somehow". `calls`, `holds conn #1`,
  `submit_job every 60s`, or `re-enqueued on restart` is information.
- **Mark the failure.** Use one accent color (red) for the failing step and its result, and keep
  everything else neutral. When the fix is clear, a small box showing what changes is welcome.
- **Contrast when it helps.** For a race or deadlock, put the two actors side by side around the
  shared resource. For a wrong value, show expected and actual next to each other.
- **Match the size to the bug.** A one-hop bug is three boxes; a chain across several components
  needs each hop. Do not inventory the whole system.

## Mechanics

Write one self-contained HTML file with a single inline `<svg>`:

- Dark page background (for example `#0d1117`) with light text and strokes. Pick an accent red
  that stays readable on it (for example `#ff7b72`).
- Size it with `viewBox` and let CSS scale it (`width: 100%; height: auto`). Lay wide flows out
  left to right and layered stacks top to bottom.
- Draw with native shapes (`rect`, `line`, `path`, `polygon`) and `<text>`. Make arrowheads with
  a `<marker>` or a small `<polygon>`. No scripts, external fonts, or images.
- Keep labels short (a few words) and at least 13px at the drawn size. Put a one-line title at the
  top that states the claim, for example "log_spans() needs two pool connections at once".
- Align boxes to a grid with even gaps and shared baselines.
