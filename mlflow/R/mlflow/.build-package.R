source(".utils.R")

# Bundle up the package into a .tar.gz file.
package_path <- devtools::build(".", path = ".")

# Hack to get around this issue:
# https://stat.ethz.ch/pipermail/r-package-devel/2020q3/005930.html
#
# The system clock check during `R CMD check` relies on two external web APIs and fails
# when they are unavailable. By setting `_R_CHECK_SYSTEM_CLOCK_` to FALSE, we can skip it:
# https://github.com/wch/r-source/blob/59a1965239143ca6242b9cc948d8834e1194e84a/src/library/tools/R/check.R#L511
Sys.setenv("_R_CHECK_SYSTEM_CLOCK_" = "FALSE")

# Run the check with `cran = TRUE`
devtools::check_built(
    path = package_path,
    cran = TRUE,
    remote = should_enable_cran_incoming_checks(),
    error_on = "note",
    check_dir = getwd(),
    args = "--no-tests",
)

# Run the check with `cran = FALSE` to detect unused imports:
# https://github.com/wch/r-source/blob/b12ffba7584825d6b11bba8b7dbad084a74c1c20/src/library/tools/R/check.R#L6070
devtools::check_built(
    path = package_path,
    cran = FALSE,
    error_on = "note",
    check_dir = getwd(),
    args = "--no-tests",
)
