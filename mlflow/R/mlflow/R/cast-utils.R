# Cast utility functions to replace deprecated forge functions
# These functions use modern rlang/vctrs functions instead of deprecated ones

#' Cast to string
#' @param x Value to cast
#' @param allow_na Whether to allow NA values
cast_string <- function(x, allow_na = FALSE) {
  if (is.null(x)) {
    if (allow_na) return(NA_character_) else stop("Value cannot be NULL")
  }
  if (is.na(x) && !allow_na) {
    stop("Value cannot be NA")
  }
  as.character(x)
}

#' Cast to nullable string
#' @param x Value to cast
cast_nullable_string <- function(x) {
  if (is.null(x)) return(NULL)
  as.character(x)
}

#' Cast to scalar double
#' @param x Value to cast
#' @param allow_na Whether to allow NA values
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

#' Cast to nullable scalar double
#' @param x Value to cast
cast_nullable_scalar_double <- function(x) {
  if (is.null(x)) return(NULL)
  if (length(x) != 1) {
    stop("Value must be a scalar (length 1)")
  }
  as.numeric(x)
}

#' Cast to nullable scalar integer
#' @param x Value to cast
cast_nullable_scalar_integer <- function(x) {
  if (is.null(x)) return(NULL)
  if (length(x) != 1) {
    stop("Value must be a scalar (length 1)")
  }
  as.integer(x)
}

#' Cast to string list
#' @param x Value to cast
cast_string_list <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.list(x)) {
    lapply(x, as.character)
  } else {
    as.list(as.character(x))
  }
}

#' Cast to choice (validate against allowed values)
#' @param x Value to cast
#' @param choices Valid choices
#' @param allow_null Whether to allow NULL values
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