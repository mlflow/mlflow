# Bug Triage

Reproduce the reported bug at the current checkout, show exactly how to reproduce it, and
suggest a fix.

## 1. Understand the issue

Extract from the issue:

- The MLflow version and the environment (Python version, OS, tracking/registry backend,
  framework and version, Databricks or OSS).
- The steps to reproduce, the expected behavior, and the actual behavior (error message,
  traceback, wrong output, UI symptom).

If the issue lacks what you need, still try a best-effort reproduction from what is there,
and list the missing details in the comment.

## 2. Reproduce

Trigger the bug in whatever form fits it: a CLI command, a short Python snippet, a request to a
local server, a pytest test, or UI steps. Use as little setup as possible: a local SQLite
store, tiny models and datasets, no external services. Put scratch files under
`$out_dir/work`. For UI bugs, seed the data the page needs through the Python client, then
follow "Running the UI" in `SKILL.md` and record the observed behavior. Capture
an image or short recording under `$out_dir/media` only when it helps a reader see
the symptom; cite the exact path in the comment as described in `SKILL.md`.

If the symptom is not obvious in the screenshot, use `agent-browser eval` to add temporary
JavaScript overlays such as an arrow, outline, or short label pointing out the bug, then
capture the annotated screenshot. Keep annotations limited to what helps explain the
observed symptom, preserving the UI and its displayed values. Save the annotated image
under `$out_dir/media` and cite its exact path in the comment. Remove the overlays before
continuing to interact with the page.

Run it against the current checkout, which is what a fix would land on. Only if it does not
reproduce there, run it against the reported version (when it is on PyPI) to tell "already
fixed" apart from "could not reproduce".

When the bug needs something the sandbox lacks (a cloud service, Databricks, a GPU, a paid API,
a large model), reproduce the closest local equivalent if one exists, such as calling the
failing function directly with the inputs from the traceback. Otherwise skip reproduction and
say why.

Shrink the reproduction once it works: drop every step the bug does not need, then rerun it to
confirm it still fails the same way. Stop after a few honest attempts. A clear "could not
reproduce, and here is what I tried" is a useful result.

## 3. Suggest a fix

Only when the bug reproduces at the current checkout. Find the line that is wrong and why, not
just where the error surfaces, then verify: patch the code, rerun the reproduction, and confirm the
symptom goes away without breaking the tests that cover that code.

Keep the fix minimal and in the style of the surrounding code. When the right fix is a design
decision (a public API, a default, or a storage schema), describe the options instead of
picking one.

## 4. Decide the verdict

Pick exactly one verdict using the label descriptions in `payload.schema.yml`, and set the
payload's `label` to that exact `const` value.

## 5. Comment

Write the comment using the template in `comment.description` of `payload.schema.yml`.

Prefer code to prose: give commands and code a reader can paste and run, not steps
to follow by hand. Keep the comment tight.

When the symptom is hard to grasp from text alone, add one diagram showing how the bug
happens: the path from the user's call to the failing line, a value changing across layers, or
expected vs actual side by side. Draw only what the bug depends on, use real function and field
names, and label every arrow. Skip it when a traceback or a sentence already makes it clear.

Draw it as an inline `<svg>` in a self-contained HTML file under `$out_dir/work` (no scripts or
external resources; white background, one accent color for the failing part, short labels), then
screenshot it and cite it like other media:

```bash
agent-browser open "file://$out_dir/work/diagram.html"
agent-browser set viewport 1280 720 2
agent-browser screenshot svg "$out_dir/media/diagram.png"
```
