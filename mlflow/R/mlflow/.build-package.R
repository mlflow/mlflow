source(".utils.R")

# Bundle up the package into a .tar.gz file. This file will be submitted to CRAN.
package_path <- devtools::build(".", path = ".")
# Run the submission check against the built package.
devtools::check_built(
    path = normalizePath(package_path),
    remote = should_enable_cran_incoming_checks(),
    error_on = "note",
    args = c("--no-tests", "--as-cran"),
)
