# regal ignore:directory-package-mismatch
package mlflow

import rego.v1

deny_jobs_without_permissions contains msg if {
	jobs := jobs_without_permissions(input.jobs)
	count(jobs) > 0
	msg := sprintf(
		"The following jobs are missing permissions: %s",
		[concat(", ", jobs)],
	)
}

deny_top_level_permissions contains msg if {
	input.permissions
	msg := "Do not use top-level permissions. Set permissions on the job level."
}

deny_unsafe_checkout contains msg if {
	# The "on" key gets transformed by conftest into "true" due to some legacy
	# YAML standards, see https://stackoverflow.com/q/42283732/2148786 - so
	# "on.push" becomes "true.push" which is why below statements use "true"
	# instead of "on".
	input["true"].pull_request_target
	some job in input.jobs
	some step in job.steps
	startswith(step.uses, "actions/checkout@")
	step["with"].ref
	msg := concat("", [
		"Explicit checkout in a pull_request_target workflow is unsafe. ",
		"See https://securitylab.github.com/resources/github-actions-preventing-pwn-requests for more information.",
	])
}

deny_unnecessary_github_token contains msg if {
	some job in input.jobs
	some step in job.steps
	startswith(step.uses, "actions/github-script@")
	regex.match(`\$\{\{\s*(secrets\.GITHUB_TOKEN|github\.token)\s*\}\}`, step["with"]["github-token"])
	msg := "Unnecessary use of github-token for actions/github-script."
}

deny_jobs_without_timeout contains msg if {
	jobs := jobs_without_timeout(input.jobs)
	count(jobs) > 0
	msg := sprintf(
		"The following jobs are missing timeout-minutes: %s",
		[concat(", ", jobs)],
	)
}

deny_unpinned_actions contains msg if {
	actions := unpinned_actions(input)
	count(actions) > 0
	msg := sprintf(
		concat("", [
			"The following actions are not pinned by full commit SHA: %s. ",
			"Use the full commit SHA instead ",
			"(e.g., actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683).",
		]),
		[concat(", ", actions)],
	)
}

###########################   RULE HELPERS   ##################################
jobs_without_permissions(jobs) := {job_id |
	some job_id, job in jobs
	not job.permissions
}

jobs_without_timeout(jobs) := {job_id |
	some job_id, job in jobs
	not job["timeout-minutes"]
}

is_step_unpinned(step) if {
	not startswith(step.uses, "./")
	not regex.match(`^[^@]+@[0-9a-f]{40}$`, step.uses)
}

unpinned_actions(inp) := unpinned if {
	# For workflow files with jobs
	inp.jobs
	unpinned := {step.uses |
		some job in inp.jobs
		some step in job.steps
		is_step_unpinned(step)
	}
}

unpinned_actions(inp) := unpinned if {
	# For composite action files with runs
	not inp.jobs
	inp.runs.steps
	unpinned := {step.uses |
		some step in inp.runs.steps
		is_step_unpinned(step)
	}
}
