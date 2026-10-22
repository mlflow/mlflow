# Initial feature request assessment

Assess the request without implementing it. Do not change source files, write `fix.patch`,
or propose a pull request.

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

## Response

Read [init.schema.yml](init.schema.yml). Pick one `outcome` from its descriptions and
write `comment` using its report template. Do not add a triage label; the workflow already
applies `enhancement`. An assessment does not grant maintainer approval to implement.
Follow the [shared guidance for issue titles](../output.md#optional-issue-title).
