python_unix_binary <- function(bin) {
  locations <- file.path(c("/usr/bin", "/usr/local/bin", path.expand("~/.local/bin")), bin)
  locations <- locations[file.exists(locations)]
  if (length(locations) > 0)
    locations[[1]]
  else
    NULL
}

#' @importFrom processx run
python_run <- function(command, ..., echo = TRUE) {
  args <- list(...)

  # find command usually switch between python 2 and 3.
  candidates <- sapply(command, python_unix_binary)
  candidates_valid <- sapply(candidates, is.null)
  if (all(candidates_valid)) stop("Could not find ", paste(command, sep = " or "), ".")
  command <- command[which(!candidates_valid)]

  # execute command
  path <- python_unix_binary(command)
  result <- run(command, args = unlist(args), echo = echo)

  invisible(result)
}
