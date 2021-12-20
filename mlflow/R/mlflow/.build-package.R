source(".utils.R")

# Increase the timeout length for `utils::download.file` because the default value (60 seconds)
# could be too short to download large packages such as h2o.
options(timeout=300)
# Install dependencies required for the submission check.
devtools::install_deps(".", dependencies = TRUE)
# Bundle up the package into a .tar.gz file. This file will be submitted to CRAN.
package_path <- devtools::build(".")
# Run the submission check against the built package.
devtools::check_built(
    package_path,
    remote = should_enable_cran_incoming_checks(),
    error_on = "note",
    args = c("--no-tests", "--as-cran"),
)
