package main

###########################   RULES   ##################################

deny_jobs_without_permissions contains msg if {
    jobs := get_jobs_without_permissions(input.jobs)
    count(jobs) > 0
    msg := sprintf(
        "The following jobs are missing permissions: %s",
        [concat(", ", jobs)]
    )
}

deny_top_level_permissions[msg] contains msg if{
    input.permissions != {}
    msg := "Do not use top-level permissions. Set permissions on the job level."
}

deny_unsafe_checkout[msg] contains msg if{
    input["true"]["pull_request_target"]
    some job_id
    job := input.jobs[job_id]
    some i
    step := job.steps[i]

    startswith(step.uses, "actions/checkout@")
    step.with.ref
    msg := "Explicit checkout in a pull_request_target workflow is unsafe. See https://securitylab.github.com/resources/github-actions-preventing-pwn-requests for more information."
}

deny_unnecessary_github_token[msg] contains msg if{
    some job_id
    job := input.jobs[job_id]
    some i
    step := job.steps[i]
    startswith(step.uses, "actions/github-script@")
    regex.match("\\$\\{\\{\\s*(secrets\\.GITHUB_TOKEN|github\\.token)\\s*\\}\\}", step.with["github-token"])
    msg := "Unnecessary use of github-token for actions/github-script."
}

deny_jobs_without_timeout[msg] contains msg if{
    jobs := get_jobs_without_timeout(input.jobs)
    count(jobs) > 0
    msg := sprintf(
        "The following jobs are missing timeout-minutes: %s",
        [concat(", ", jobs)]
    )
}

deny_unpinned_actions[msg] contains msg if{
    actions := get_unpinned_actions(input)
    count(actions) > 0
    msg := sprintf(
        "The following actions are not pinned by full commit SHA: %s. Use the full commit SHA instead (e.g., actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683).",
        [concat(", ", actions)]
    )
}

###########################   HELPERS   ##################################

get_jobs_without_permissions(jobs) = jobs_without_permissions if{
    jobs_without_permissions := { job_id |
        job := jobs[job_id]
        not job.permissions
    }
}

get_jobs_without_timeout(jobs) = jobs_without_timeout if{
    jobs_without_timeout := { job_id |
        job := jobs[job_id]
        not job["timeout-minutes"]
    }
}

is_step_unpinned(step) if{
    step.uses
    not startswith(step.uses, "./")
    not regex.match("^[^@]+@[0-9a-f]{40}$", step.uses)
}

get_unpinned_actions(inp) = unpinned_actions if{
    inp.jobs
    all_steps := [ step | job := inp.jobs[_]; step := job.steps[_] ]
    unpinned_actions := { step.uses |
        step := all_steps[_]
        is_step_unpinned(step)
    }
}

get_unpinned_actions(inp) = unpinned_actions if{
    inp.runs.steps
    not inp.jobs
    unpinned_actions := { step.uses |
        step := inp.runs.steps[_]
        is_step_unpinned(step)
    }
}
