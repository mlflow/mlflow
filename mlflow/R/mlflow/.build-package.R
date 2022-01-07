source(".utils.R")

# Bundle up the package into a .tar.gz file. This file will be submitted to CRAN.
package_path <- devtools::build(".", path = ".")
# Run the submission check against the built package.
devtools::check_built(
    path = package_path,
    cran = TRUE,
    remote = should_enable_cran_incoming_checks(),
    error_on = "note",
    check_dir = getwd(),
    args = "--no-tests",
)
# This runs checks that are disabled when `cran` is TRUE (e.g. unused import check).
devtools::check_built(
    path = package_path,
    cran = FALSE,
    error_on = "note",
    check_dir = getwd(),
    args = "--no-tests",
)
