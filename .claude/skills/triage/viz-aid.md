# Visual aids in comments

Add an explanatory diagram when the verdict is `triage: reproduced` or `triage: already-fixed`,
unless screenshots or recordings already show the symptom. Make it after the reproduction and
any fix are verified, and skip it if fewer than 10 minutes of the run remain.

1. **Takeaway:** Choose one finding the reader should understand. State it in a short,
   plain-language headline that makes sense without reading the issue.
2. **Mechanism:** Identify the events, values, or resources the reader must relate to
   understand why the failure occurs. Choose the visual relationship first, then the shapes.
   Make the failure visible through position, order, connections, or scale. Use only
   verified values and names, and mark steps that were not measured as illustrative.
3. **Fix:** When proposing a PR, also show how the fix changes the failing path: put the
   fixed path next to the broken one with the same inputs, mark the point where the fix
   intervenes, and show the verified result after the fix. Do not reduce the fix to a
   caption.
4. **Size:** Design the figure 800px wide, the width GitHub shows it at, and keep every label
   at least 14px. Never break an identifier, path, or ID mid-token.
5. **Theme:** Always use a dark theme. Keep secondary text at least as light as `#9da7b3` so
   it stays readable on the dark background.

Create a self-contained HTML file with CSS and inline SVG under `$out_dir/work`. Include
`<meta charset="utf-8">`, set text colors explicitly (including `fill` for SVG text) so labels
contrast with their background, escape any text copied from the issue, and load or send
nothing. Open it with `file://` in its own `agent-browser` session (`--session viz`), set the
viewport to 800px wide at device scale factor 2, and screenshot just the figure element into
`$out_dir/media`. Check the screenshot for clipped or overlapping text and close the session.
Cite the screenshot in the Summary, right after the verdict line, with alt text that states
the takeaway.
When proposing a PR, also cite the same screenshot in `pull_request.body`, with the same alt
text.
