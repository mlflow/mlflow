package main

deny[msg] {
    count(jobs_without_timeout_minutes(input.jobs)) > 0

    msg := sprintf("The following jobs are missing permissions: %s",
        [concat(", ", jobs_without_timeout_minutes(input.jobs))])
}


deny[msg] {
    input.permissions

    msg := "Do not use top-level permissions"
}

###########################   RULE HELPERS   ##################################
jobs_without_timeout_minutes(jobs) = jobs_without_timeout_minutes {
    jobs_without_timeout_minutes := { job_id |
        job := jobs[job_id]
        not job["permissions"]
    }
}
