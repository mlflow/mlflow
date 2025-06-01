package main

import future.keywords.in

deny_jobs_without_permissions[msg] {
    jobs_without_permissions := get_jobs_without_permissions(input.jobs)
    count(jobs_without_permissions) > 0
    msg := sprintf("The following jobs are missing permissions: %s",
        [concat(", ", jobs_without_permissions)])
}

deny_top_level_permissions[msg] {
    input.permissions
    msg := "Do not use top-level permissions. Set permissions on the job level."
}

deny_unsafe_checkout[msg] {
    # The "on" key gets transformed by conftest into "true" due to some legacy
    # YAML standards, see https://stackoverflow.com/q/42283732/2148786 - so
    # "on.push" becomes "true.push" which is why below statements use "true"
    # instead of "on".
    input["true"]["pull_request_target"]
    some job in input["jobs"]
    some step in job["steps"]
    startswith(step["uses"], "actions/checkout@")
    step["with"]["ref"]
    msg := "Explicit checkout in a pull_request_target workflow is unsafe. See https://securitylab.github.com/resources/github-actions-preventing-pwn-requests for more information."
}

deny_unnecessary_github_token[msg] {
    some job in input["jobs"]
    some step in job["steps"]
    startswith(step["uses"], "actions/github-script@")
    regex.match("\\$\\{\\{\\s*(secrets\\.GITHUB_TOKEN|github\\.token)\\s*\\}\\}", step["with"]["github-token"])
    msg := "Unnecessary use of github-token for actions/github-script."
}

deny_jobs_without_timeout[msg] {
    jobs_without_timeout := get_jobs_without_timeout(input.jobs)
    count(jobs_without_timeout) > 0
    msg := sprintf("The following jobs are missing timeout-minutes: %s",
        [concat(", ", jobs_without_timeout)])
}

deny_unpinned_actions[msg] {
    unpinned_actions := get_unpinned_actions(input)
    count(unpinned_actions) > 0
    msg := sprintf("The following actions are not pinned by full commit SHA: %s. Use the full commit SHA instead (e.g., actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683).",
        [concat(", ", unpinned_actions)])
}

###########################   RULE HELPERS   ##################################
get_jobs_without_permissions(jobs) = jobs_without_permissions {
    jobs_without_permissions := { job_id |
        job := jobs[job_id]
        not job["permissions"]
    }
}

get_jobs_without_timeout(jobs) = jobs_without_timeout {
    jobs_without_timeout := { job_id |
        job := jobs[job_id]
        not job["timeout-minutes"]
    }
}

is_step_unpinned(step) {
    step["uses"]
    not startswith(step["uses"], "./")
    not regex.match("^[^@]+@[0-9a-f]{40}$", step["uses"])
}

get_unpinned_actions(inp) = unpinned_actions {
    # For workflow files with jobs
    inp.jobs
    all_steps := [ step | job := inp.jobs[_]; step := job.steps[_] ]
    unpinned_actions := { step["uses"] |
        step := all_steps[_]
        is_step_unpinned(step)
    }
}

get_unpinned_actions(inp) = unpinned_actions {
    # For composite action files with runs
    inp.runs.steps
    not inp.jobs
    unpinned_actions := { step["uses"] |
        step := inp.runs.steps[_]
        is_step_unpinned(step)
    }
}
