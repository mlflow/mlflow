# Cast utility functions to replace deprecated forge functions
# These functions use modern rlang/vctrs functions instead of deprecated ones

cast_string <- function(x, allow_na = FALSE) {
  if (is.null(x)) {
    if (allow_na) return(NA_character_) else stop("Value cannot be NULL")
  }
  if (is.na(x) && !allow_na) {
    stop("Value cannot be NA")
  }
  as.character(x)
}

cast_nullable_string <- function(x) {
  if (is.null(x)) return(NULL)
  as.character(x)
}

cast_scalar_double <- function(x, allow_na = FALSE) {
  if (is.null(x)) {
    if (allow_na) return(NA_real_) else stop("Value cannot be NULL")
  }
  if (length(x) != 1) {
    stop("Value must be a scalar (length 1)")
  }
  if (is.na(x) && !allow_na) {
    stop("Value cannot be NA")
  }
  as.numeric(x)
}

cast_nullable_scalar_double <- function(x) {
  if (is.null(x)) return(NULL)
  if (length(x) != 1) {
    stop("Value must be a scalar (length 1)")
  }
  as.numeric(x)
}

cast_nullable_scalar_integer <- function(x) {
  if (is.null(x)) return(NULL)
  if (length(x) != 1) {
    stop("Value must be a scalar (length 1)")
  }
  as.integer(x)
}

cast_string_list <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.list(x)) {
    lapply(x, as.character)
  } else {
    as.list(as.character(x))
  }
}

cast_choice <- function(x, choices, allow_null = FALSE) {
  if (is.null(x)) {
    if (allow_null) return(NULL) else stop("Value cannot be NULL")
  }
  x <- as.character(x)
  if (!x %in% choices) {
    stop(sprintf("Value must be one of: %s", paste(choices, collapse = ", ")))
  }
  x
}