# extract a file path from CLI output with consideration of interop requirements
# on Windows
extract_path <- function(output) {
  path <- gsub("\r|\n", "", output)

  normalizePath(path, winslash = "/")
}

strip_prefix <- function(x, prefix) {
  if (identical(substr(x, 1, nchar(prefix)), prefix)) {
    substr(x, nchar(prefix) + 1, nchar(x))
  } else {
    x
  }
}
