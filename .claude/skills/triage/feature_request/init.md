# Initial feature request assessment

Assess the request first. If implementation looks cheap, attempt a minimal working proof
of concept as described below. Do not write `fix.patch` or propose a pull request.

Identify the user's goal, the requested behavior, and the affected MLflow components.
Check the current code and relevant documentation for existing support or a practical
workaround. Verify existing support with a small usage example when useful; do not
classify a missing feature as a reproduced bug.

## Cost and impact

Keep the assessment brief and limited to three things:

- Implementation effort: small, medium, or large, with the main changes and supporting code permalinks.
- Impact on existing features: the concrete behavior or compatibility risks found in the code.
- Checks needed: the key unresolved decisions and tests or manual checks before proceeding.

Mention uncertainty when it affects the conclusion, and distinguish checks performed
from checks still needed. Do not add separate estimates for operational cost, maintenance,
or user value.

## Optional proof of concept

When the effort is small and the intended behavior is clear, attempt the smallest working
proof of concept to refine the effort estimate and uncover implementation issues. Skip
the attempt if it would add little to the assessment. Stay within the shared runtime
budget. You may change code under `mlflow/`
and `tests/` or write a standalone example under `$out_dir/work`. Exercise the requested
behavior with a focused command, test, or browser interaction. Avoid expanding it into a
complete implementation. Stop if it needs unresolved design decisions or substantially
more work than expected, and report the blocker.

Briefly report what the attempt revealed about effort, impact, or blockers, together with
the verification command and observed result. Include a short code snippet only when it
helps explain a finding. State what remains incomplete; only call it working if
verification succeeded. Keep the assessment useful even when the attempt fails.

## Response

Read [init.schema.yml](init.schema.yml). Pick one `outcome` from its descriptions and
write `comment` using its report template. Do not add a triage label; the workflow already
applies `enhancement`. A proof of concept does not grant maintainer approval for a PR.
Follow the [shared guidance for issue titles](../output.md#optional-issue-title).
