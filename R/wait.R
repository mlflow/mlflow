wait_for <- function(f, wait, sleep) {
  command_start <- Sys.time()

  success <- FALSE
  while (!success && Sys.time() < command_start + wait) {
    tryCatch({
      suppressWarnings({
        f()
      })

      success <- TRUE
    }, error = function(err) {
    })

    if (!success) Sys.sleep(sleep)
  }
}
