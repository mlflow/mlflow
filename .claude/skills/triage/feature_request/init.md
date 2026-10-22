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

For a small, clear request, try a tiny prototype only to refine the estimate and find
implementation issues. Keep changes under `mlflow/`, `tests/`, or `$out_dir/work` and stay
within the runtime budget. Stop if design decisions or substantial work are needed.
Report the result and blockers in 1–2 sentences under Implementation effort; omit code
listings and only call it working if verification succeeded.

## Response

Read [init.schema.yml](init.schema.yml). Pick one `outcome` from its descriptions and
write `comment` using its report template. Do not add a triage label; the workflow already
applies `enhancement`. A proof of concept does not grant maintainer approval for a PR.
Follow the [shared guidance for issue titles](../output.md#optional-issue-title).
