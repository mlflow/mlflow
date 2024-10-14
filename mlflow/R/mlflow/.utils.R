
# This script defines utility functions only used during development.

should_enable_cran_incoming_checks <- function() {
    # The CRAN incoming feasibility check performs a package recency check (this is undocumented)
    # that examines the number of days since the last release and raises a NOTE if it's < 7.
    # Relevant code:
    # https://github.com/wch/r-source/blob/4561aea946a75425ddcc8869cdb129ed5e27af97/src/library/tools/R/QC.R#L8005-L8008
    # This check needs to be disabled for a week after releasing a new version.
    cran_url <- getOption("repos")["CRAN"]
    desc_url <- url(paste0(cran_url, "/web/packages/mlflow/DESCRIPTION"))
    field <- "Date/Publication"
    desc <- read.dcf(desc_url, fields = c(field))
    close(desc_url)
    publication_date <- as.Date(unname(desc[1, field]))
    today <- Sys.Date()
    days_since_last_release <- as.numeric(difftime(today, publication_date, units="days"))
    if (days_since_last_release < 7) {
        return(FALSE)
    }

    # Skip the release frequency check if the number of releases in the last 180 days exceeds 6.
    url <- "https://crandb.r-pkg.org/mlflow/all"
    response <- httr::GET(url)
    json_data <- httr::content(response, "parsed")
    release_dates <- as.Date(sapply(json_data$timeline, function(x) substr(x, 1, 10)))
    today <- Sys.Date()
    days_ago_180 <- as.Date(today) - 180
    recent_releases <- sum(release_dates >= days_ago_180)
    if (recent_releases > 6) {
        return(FALSE)
    }

    return(TRUE)
}
