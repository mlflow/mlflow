# TODO: Use crate from CRAN!

# Remove after rlang 0.3.0 is released
locally <- function(..., .env = env(caller_env())) {
  dots <- exprs(...)
  nms <- names(dots)
  out <- NULL

  for (i in seq_along(dots)) {
    out <- eval_bare(dots[[i]], .env)

    nm <- nms[[i]]
    if (nm != "") {
      .env[[nm]] <- out
    }
  }

  out
}

#' @import rlang
NULL

#' Crate a function to share with another process
#'
#' @description
#'
#' `crate()` creates functions in a self-contained environment
#' (technically, a child of the base environment). This has two
#' advantages:
#'
#' * They can easily be executed in another process.
#'
#' * Their effects are reproducible. You can run them locally with the
#'   same results as on a different process.
#'
#' Creating self-contained functions requires some care, see section
#' below.
#'
#'
#' @section Creating self-contained functions:
#'
#' * They should call package functions with an explicit `::`
#'   namespace. This includes packages in the default search path with
#'   the exception of the base package. For instance `var()` from the
#'   stats package must be called with its namespace prefix:
#'   `stats::var(x)`.
#'
#' * They should declare any data they depend on. You can declare data
#'   by supplying additional arguments or by unquoting objects with `!!`.
#'
#' @param .fn A fresh formula or function. "Fresh" here means that
#'   they should be declared in the call to `crate()`. See examples if
#'   you need to crate a function that is already defined. Formulas
#'   are converted to purrr-like lambda functions using
#'   [rlang::as_function()].
#' @param ... Arguments to declare in the environment of `.fn`. If a
#'   name is supplied, the object is assigned to that name. Otherwise
#'   the argument is automatically named after itself.
#'
#' @export
#' @examples
#' # You can create functions using the ordinary notation:
#' crate(function(x) stats::var(x))
#'
#' # Or the formula notation:
#' crate(~stats::var(.x))
#'
#' # Declare data by supplying named arguments. You can test you have
#' # declared all necessary data by calling your crated function:
#' na_rm <- TRUE
#' fn <- crate(~stats::var(.x, na.rm = na_rm))
#' try(fn(1:10))
#'
#' # Arguments are automatically named after themselves so that the
#' # following are equivalent:
#' crate(~stats::var(.x, na.rm = na_rm), na_rm = na_rm)
#' crate(~stats::var(.x, na.rm = na_rm), na_rm)
#'
#' # However if you supply a complex expression, do supply a name!
#' crate(~stats::var(.x, na.rm = na_rm), !na_rm)
#' crate(~stats::var(.x, na.rm = na_rm), na_rm = na_rm)
#'
#' # For small data it is handy to unquote instead. Unquoting inlines
#' # objects inside the function. This is less verbose if your
#' # function depends on many small objects:
#' fn <- crate(~stats::var(.x, na.rm = !!na_rm))
#' fn(1:10)
#'
#' # One downside is that the individual sizes of unquoted objects
#' # won't be shown in the crate printout:
#' fn
#'
#'
#' # The function or formula you pass to crate() should defined inside
#' # the crate() call, i.e. you can't pass an already defined
#' # function:
#' fn <- function(x) toupper(x)
#' try(crate(fn))
#'
#' # If you really need to crate an existing function, you can
#' # explicitly set its environment to the crate environment with the
#' # set_env() function from rlang:
#' crate(rlang::set_env(fn))
crate <- function(.fn, ...) {
  # Evaluate arguments in a child of the caller so the caller context
  # is in scope and new data is created in a separate child
  env <- child_env(caller_env())
  dots <- exprs(..., .named = TRUE)
  locally(!!!dots, .env = env)

  # Quote and evaluate in the local env to avoid capturing execution
  # envs when user passed an unevaluated function or formula
  fn <- eval_bare(enexpr(.fn), env)

  # Isolate the evaluation environment from the search path
  env_poke_parent(env, base_env())

  if (is_formula(fn)) {
    fn <- as_function(fn)
  } else if (!is_function(fn)) {
    abort("`.fn` must evaluate to a function")
  }

  if (!is_reference(get_env(fn), env)) {
    abort("The function must be defined inside the `crate()` call")
  }

  new_crate(fn)
}


new_crate <- function(crate) {
  if (!is_function(crate)) {
    abort("`crate` must be a function")
  }

  structure(crate, class = "crate")
}

#' Is an object a crate?
#'
#' @param x An object to test.
#' @export
is_crate <- function(x) {
  inherits(x, "crate")
}

# Unexported until the `bytes` class is moved to lobstr (and probably
# becomes `lobstr_bytes`)
crate_sizes <- function(crate) {
  bare_fn <- unclass(crate)
  environment(bare_fn) <- global_env()

  bare_size <- pryr::object_size(bare_fn)

  env <- fn_env(crate)
  nms <- ls(env)

  n <- length(nms) + 1
  out <- new_list(n, c("function", nms))
  out[[1]] <- bare_size

  index <- seq2(2, n)
  get_size <- function(nm) pryr::object_size(env[[nm]])
  out[index] <- lapply(nms, get_size)

  # Sort data by decreasing size but keep function first
  order <- order(as.numeric(out[-1]), decreasing = TRUE)
  out <- out[c(1, order + 1)]

  out
}


#' @export
print.crate <- function(x, ...) {
  sizes <- crate_sizes(x)

  total_size <- format(pryr::object_size(x), ...)
  cat(sprintf("<crate> %s\n", total_size))

  fn_size <- format(sizes[[1]], ...)
  cat(sprintf("* function: %s\n", fn_size))

  nms <- names(sizes)
  for (i in seq2_along(2, sizes)) {
    nm <- nms[[i]]
    size <- format(sizes[[i]], ...)
    cat(sprintf("* `%s`: %s\n", nm, size))
  }

  # Print function without the environment tag
  bare_fn <- unclass(x)
  environment(bare_fn) <- global_env()
  print(bare_fn, ...)

  invisible(x)
}

# From pryr
format.bytes <- function(x, digits = 3, ...) {
  power <- min(floor(log(abs(x), 1000)), 4)
  if (power < 1) {
    unit <- "B"
  } else {
    unit <- c("kB", "MB", "GB", "TB")[[power]]
    x <- x / (1000 ^ power)
  }

  x <- signif(x, digits = digits)
  fmt <- format(unclass(x), big.mark = ",", scientific = FALSE)
  paste(fmt, unit)
}
