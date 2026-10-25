# Bug reproduction and fix verification

Use these steps to investigate a bug or verify a proposed fix.

## 1. Understand the issue

Extract from the issue:

- The MLflow version and the environment (Python version, OS, tracking/registry backend,
  framework and version, Databricks or OSS).
- The steps to reproduce, the expected behavior, and the actual behavior (error message,
  traceback, wrong output, UI symptom).
- When the issue links to a failed GitHub Actions job or workflow run, or a pull request with
  failing checks, use the `analyze-ci` skill with the URL before reproducing the failure.

If the issue lacks what you need, still try a best-effort reproduction from what is there,
and list the missing details in the comment.

## 2. Reproduce

Trigger the bug in whatever form fits it: a CLI command, a short Python snippet, a request to a
local server, a pytest test, or UI steps. Use as little setup as possible: a local SQLite
store, tiny models and datasets, no external services. Put scratch files under
`$out_dir/work`. For UI bugs, seed the data the page needs through the Python client, then
drive the UI as described in "Running the UI" in [runtime.md](../runtime.md) and record the observed behavior.
For visible UI bugs, capture an image or short recording of the symptom under `$out_dir/media`;
cite the exact path in the comment as described in [output.md](../output.md).

For a bug in an external API integration, such as OpenAI or Anthropic, reproduce it with a live
call through the affected integration to the provider's real endpoint, and record the integration
package version and any applicable provider API or model version. A direct call to the handler
or implementation function, or inspection of the code alone, does not count as an endpoint
reproduction. If the report does not provide enough information to make that live call, ask the
reporter to share a minimal request and response against the provider's real endpoint, together
with those versions.

If the symptom is not obvious in the screenshot, use `agent-browser eval` to add temporary
JavaScript overlays such as an arrow, outline, or short label pointing out the bug, then
capture the annotated screenshot. Keep annotations limited to what helps explain the
observed symptom, preserving the UI and its displayed values. Save the annotated image
under `$out_dir/media` and cite its exact path in the comment. Remove the overlays before
continuing to interact with the page.

Before choosing a screenshot, ask: would a reviewer who sees only this image understand why the
state is wrong? If the answer depends on something done earlier (a selection, a search, a page
change, a Back press), record a short video instead of screenshots. Rehearse the steps with the
selectors you will put in `$out_dir/work/steps.json` (snapshot refs such as `@e12` don't carry over
to a new session), and hide side panels, popups, and onboarding hints that cover the page, then:

```bash
agent-browser --session <sess> --input-mode human --init-script .claude/skills/triage/scripts/init.js open <url>
agent-browser --session <sess> record start $out_dir/media/<name>.mp4 --cursor --contact-sheet
timeout 120 agent-browser --session <sess> batch --bail < $out_dir/work/steps.json  # [["click", "<sel>"], ["wait", "500"], ...]
agent-browser --session <sess> record stop
```

`init.js` defines `caption(text, theme = "light", position = "top")`, which shows a short caption
at the top of the page by default (`caption()` hides it; pass `"dark"` as the second argument when
the page uses a dark theme). Pass `"bottom"` as the third argument to place it at the bottom, e.g.
`caption("Step description", "light", "bottom")`.
Caption each step so viewers can follow along, especially actions that leave no visible trace,
such as the browser's Back button, a key press, or a wait: add
`["eval", "caption('Browser Back')"]` and a short wait before the action.

If the batch times out, retry once in a fresh session, then fall back to screenshots. Check the
contact sheet to confirm the video shows the bug before citing it.
When you propose a fix for a visible UI bug, record the same steps again after applying it,
against the same servers from "Running the UI" in [runtime.md](../runtime.md), and cite both videos in the
`pull_request` body as a before/after comparison. Adjust the captions for the after video if
needed.

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

For frontend fixes, verify against the dev servers from "Running the UI" in [runtime.md](../runtime.md), which
reload code changes. For simple visual or UI changes, manual browser inspection of the
reproduction before and after the fix can be sufficient verification. Run relevant existing
tests when they meaningfully cover the affected behavior.

Keep the fix minimal and in the style of the surrounding code. If a design choice remains
unresolved (a public API, a default, or a storage schema), describe the options instead of
picking one.
