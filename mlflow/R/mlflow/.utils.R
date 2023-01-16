
# This script defines utility functions only used during development.

should_enable_cran_incoming_checks <- function() {
    # The CRAN incoming feasibility check performs a package recency check (this is undocumented)
    # that examines the number of days since the last release and raises a NOTE if it's < 7.
    # Relevant code:
    # https://github.com/wch/r-source/blob/4561aea946a75425ddcc8869cdb129ed5e27af97/src/library/tools/R/QC.R#L8005-L8008
    # This check needs to be disabled for a week after releasing a new version.
    desc_url <- url("https://cran.r-project.org/web/packages/mlflow/DESCRIPTION")
    field <- "Date/Publication"
    desc <- read.dcf(desc_url, fields = c(field))
    close(desc_url)
    publication_date <- as.Date(unname(desc[1, field]))
    today <- Sys.Date()
    days_since_last_release <- as.numeric(difftime(today, publication_date, units="days"))
    days_since_last_release > 7
}
