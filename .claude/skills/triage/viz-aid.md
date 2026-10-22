# Visual aids in comments

Add an explanatory diagram when the verdict is `triage: reproduced` or `triage: already-fixed`,
unless screenshots or recordings already show the symptom. Make it after the reproduction and
any fix are verified, and skip it if fewer than 10 minutes of the run remain.

1. **Takeaway:** Pick one finding and state it in a plain-language headline of at most two lines
   that makes sense without the issue. Keep the conditions the evidence attaches to it (which
   input, environment, or threshold) instead of broadening it.
2. **Mechanism:** Make the cause visible in the drawing: relate the events, values, or resources
   through position, order, connections, or scale. Every value, name, and claim, footnotes
   included, must match the verified findings. Label unmeasured steps as illustrative where
   they appear.
3. **Legibility:** Design 800px wide with every label at least 14px; never wrap inside an
   identifier, path, or ID. Use a dark theme and set every text color explicitly (`fill` for SVG
   text). Text that explains the failure uses the main text color; only secondary labels may be
   dimmer, down to `#9da7b3`.

Create a self-contained HTML file with CSS and inline SVG under `$out_dir/work`: include
`<meta charset="utf-8">`, escape text copied from the issue, and load or send nothing. Open it
via `file://` in its own `agent-browser` session (`--session viz`) with an 800px-wide viewport at
device scale factor 2, screenshot just the figure element into `$out_dir/media`, look at it and
fix clipped or overlapping text, then close the session. Cite the screenshot in the Summary,
right after the verdict line, with alt text that states the takeaway.
