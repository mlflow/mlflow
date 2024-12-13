package main

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

###########################   RULE HELPERS   ##################################
get_jobs_without_permissions(jobs) = jobs_without_permissions {
    jobs_without_permissions := { job_id |
        job := jobs[job_id]
        not job["permissions"]
    }
}
