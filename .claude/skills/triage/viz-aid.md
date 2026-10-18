# Visual aids in comments

Skip these instructions when:

- Handling a maintainer follow-up.
- There are no verified reproduction results, including `not-reproduced`, `needs-info`,
  and `skipped` outcomes.
- Screenshots or recordings already explain the symptom.

Otherwise, include a visual aid in the comment's Summary, based on verified reproduction
results. Build it around these principles:

1. **Takeaway:** Choose one finding the reader should understand. State it in a short,
   plain-language headline that makes sense without reading the issue.
2. **Mechanism:** Identify the events, values, or resources the reader must relate to
   understand why the failure occurs. Choose the visual relationship first, then the shapes.
   Make the failure visible through position, order, connections, or scale.
3. **Evidence:** Put concrete reproduction inputs and results beside the relevant step.
   Label arrows with actions or data. Lead labels with behavior; use component and function
   names to connect that behavior to the code. Keep each measured result attached to its
   conditions; do not present results from separate setups as one run. Distinguish measured
   results from illustrative steps or inferred behavior. Keep secondary setup details and
   long code snippets outside the visual.
4. **Comprehension:** Aim for about 180 visible words, omitting prose that repeats the
   drawing. Use a label font size of at least 14px at a 1000px-wide viewing size. Preserve
   causal dependencies and measurement conditions, even if they need more words. Inspect
   the rendered image at its final size. With explanatory sentences hidden but labels and
   values retained, can you point to the visual feature that demonstrates the failure?
   If the conclusion exists only in the headline or caption, revise the layout. Correct
   clipping, overlapping labels, and weak contrast. Verify that every connector starts and
   ends on the intended component or lifeline, with arrowheads showing the direction.
   Move longer explanations into the comment.

Always use a dark theme for generated visuals.
Set text colors explicitly, including `fill` for SVG text, so labels contrast with their
background.
You can create a self-contained HTML file using CSS, JavaScript, and inline SVG as needed under
`$out_dir/work`. Open it with `agent-browser` using `file://`, or serve it locally if needed,
then capture a screenshot. Include `<meta charset="utf-8">` to avoid garbled characters.
Save the screenshot under `$out_dir/media` and cite its path in the comment's Summary
section. Label it as an explanatory diagram.
