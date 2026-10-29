# Feature request triage output

## Initial triage

Pick one `verdict` from the selected schema's descriptions, and write `comment` using
its report template. Keep the verdict unprefixed in the comment. Only write the payload;
the workflow adds `triage: ` and applies the verdict as an issue label.
Omit boilerplate about not running tests or not proposing a PR.

## Maintainer follow-up

Write a concise, free-form `comment` addressing the request, including changes to the
triage or implementation results and verification. Do not repeat the original report
or include a `verdict`.

## Pull request eligibility

Only propose a PR on a maintainer follow-up requesting implementation, when the behavior
is not already supported and the implementation is verified against a concrete usage
example and relevant tests. Include `pull_request` only with a verified `fix.patch` that
meets the [shared output requirements](../output.md).
