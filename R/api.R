mlflow_url <- function(mc) {
  mc$url
}

mlflow_api_get <- function(url) {

}

mlflow_api_path <- function(version) {
  switch(
    version,
    "2.0" = "ajax-api/2.0/preview/mlflow"
  )
}

#' @importFrom httr content
#' @importFrom httr GET
#' @importFrom jsonlite fromJSON
mlflow_api <- function(mc, ..., data = NULL, verb = "GET", version = "2.0") {
  args <- list(...)
  url <- mlflow_url(mc)

  api_url <- file.path(
    mlflow_url(mc),
    mlflow_api_path(version),
    paste(args, collapse = "/")
  )

  response <- switch(
    verb,
    GET = GET(api_url),
    stop("Verb '", verb, "' is unsupported.")
  )

  text <- content(response, "text", encoding = "UTF-8")
  fromJSON(text)
}
