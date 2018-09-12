wait_for <- function(f, wait, sleep) {
  command_start <- Sys.time()

  success <- FALSE
  while (!success && Sys.time() < command_start + wait) {
    success <- suppressWarnings({
      tryCatch({
        f()
        TRUE
      }, error = function(err) {
        FALSE
      })
    })

    if (!success) Sys.sleep(sleep)
  }

  if (!success) {
    stop("Operation failed after waiting for ", wait, " seconds")
  }
}
