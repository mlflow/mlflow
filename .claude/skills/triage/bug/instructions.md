# Bug Triage

Reproduce the reported bug at the current checkout, tell a maintainer exactly how to reproduce
it, and suggest a fix, so they can confirm and fix it in minutes.

## 1. Understand the issue

Extract from the issue:

- The MLflow version and the environment (Python version, OS, tracking/registry backend,
  framework and version, Databricks or OSS).
- The steps to reproduce, the expected behavior, and the actual behavior (error message,
  traceback, wrong output, UI symptom).

If the issue lacks what you need, still try a best-effort reproduction from what is there,
and list the missing details in the comment.

## 2. Find the entry point

Search for the symbols and error messages from the issue (`rg "<message>" mlflow/`) to learn
which API, CLI command, or page the user hit and what inputs reach the failure. Read only
enough to write the reproduction.

## 3. Reproduce

Trigger the bug in whatever form fits it: a CLI command, a short Python snippet, a request to a
local server, a pytest test, or UI steps. Use as little setup as possible: a local SQLite or
file store, tiny models and datasets, no external services. Put scratch files under
`$out_dir/work`. For UI bugs, seed the data the page needs through the Python client, then
follow "Running the UI" in `SKILL.md` and capture a screenshot of the symptom.

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

## 4. Suggest a fix

Only when the bug reproduces at the current checkout. Find the line that is wrong and why, not
just where the error surfaces, then verify: patch the code, rerun the reproduction, and confirm the
symptom goes away without breaking the tests that cover that code. Save the patch with
`git diff > $out_dir/fix.patch`, then revert it; the fix goes in the comment, not the checkout.

Keep the fix minimal and in the style of the surrounding code. When the right fix is a design
decision (a public API, a default, or a storage schema), describe the options instead of
picking one.

## 5. Decide the verdict

Pick exactly one, and use its label as the payload's `label`:

| Verdict        | Label                    | When                                                                  |
| -------------- | ------------------------ | --------------------------------------------------------------------- |
| Reproduced     | `triage: reproduced`     | The bug exists at the current checkout.                               |
| Fixed          | `triage: fixed`          | The reported version reproduces it and the current checkout does not. |
| Not reproduced | `triage: not-reproduced` | Neither reproduces it with the information given.                     |
| Needs info     | `triage: needs-info`     | The issue is too incomplete to attempt a reproduction.                |
| Skipped        | `triage: skipped`        | It needs resources the sandbox lacks.                                 |

## 6. Comment

Fill in this template: replace each `<...>` placeholder, and drop a line or section that has
nothing to say, except the verdict.

```markdown
### Outcome

**<verdict>**: <one or two sentences on what happens and on which versions>.

- Expected: <what should happen>
- Actual: <what happens, in one line, e.g. the exception and its message>
- Fails at: <permalink to the line that raises or returns the wrong value>
- Missing information: <what the reporter should add>

### How to reproduce

<whatever a maintainer needs to see the bug themselves: the versions you tested on, then the
commands, scripts, or UI steps to run and the output or screenshot that shows the bug. Shape
it to fit the bug, putting anything to run in code blocks.>

### Suggested fix

<the root cause in a sentence or two, with permalinks>

<the fix: for a short change, the patch in a `diff` code block; for a longer or multi-file
change, what changes in each file and why, with permalinks>

### What I tried

<for any verdict other than Reproduced: the attempts and why each did not reproduce it>
```

Prefer code to prose: give commands and code the maintainer can paste and run, not steps
to follow by hand. Keep the body tight. A maintainer should be able to read it and rerun the
steps in a few minutes.
