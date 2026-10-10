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

## 4. Propose a pull request

The workflow opens a draft pull request from your fix only when all of these hold:

- The verdict is Reproduced and the issue is simple.
- The fix is straightforward, verified as in step 3, and needs no decisions: no design choice,
  no trade-off between plausible fixes, nothing a maintainer should weigh in on first.
- Every changed file is under `mlflow/` or `tests/`.
- The `PR_DISABLED` environment variable is not `true`.

Otherwise, skip this step: the comment's suggested fix is enough.

When it applies, add a regression test that fails without the fix and passes with it, next to
the existing tests for that code, and run it together with those tests. Then leave only the fix
and the test changed in the checkout (revert anything else you edited there) and write the
patch:

```bash
git add -N mlflow tests
git diff HEAD -- mlflow tests > "$out_dir/fix.patch"
git diff HEAD --stat
```

The last command must list only the files in the patch. Then add `pull_request` to the payload
using the template in `pull_request` of `payload.schema.yml`. The comment's suggested fix
still describes the change in full.

## 5. Decide the verdict

Pick exactly one verdict using the label descriptions in `payload.schema.yml`, and set the
payload's `label` to that exact `const` value.

## 6. Comment

Write the comment using the template in `comment.description` of `payload.schema.yml`.

Prefer code to prose: give commands and code a reader can paste and run, not steps
to follow by hand. Keep the comment tight.
