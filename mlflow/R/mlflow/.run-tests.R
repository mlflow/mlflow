parent_dir <- dir("../", full.names = TRUE)
package <- parent_dir[grepl("mlflow_", parent_dir)]

library(reticulate)
use_condaenv(mlflow:::mlflow_conda_env_name())

# Disable CRAN incoming feasibility check within a week after the latest release because it fails.
#
# Relevant code:
# https://github.com/wch/r-source/blob/4561aea946a75425ddcc8869cdb129ed5e27af97/src/library/tools/R/QC.R#L8005-L8008
install.packages(c("xml2", "rvest"))
library(xml2)
library(rvest)

URL <- "https://cran.r-project.org/web/packages/mlflow/index.html"
html <- read_html(URL)
xpath <- '//td[text()="Published:"]/following-sibling::td[1]/text()'
published_date <- as.Date(html_text(html_nodes(html, xpath=xpath)))
today <- Sys.Date()
days_since_last_release <- difftime(today, published_date, units="days")
remote <- as.numeric(days_since_last_release) > 7

devtools::check_built(path = package, remote = remote, error_on = "note", args = "--no-tests")
source("testthat.R")
