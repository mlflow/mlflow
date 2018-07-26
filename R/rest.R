mlflow_rest_path <- function(version) {
  switch(
    version,
    "2.0" = "ajax-api/2.0/preview/mlflow"
  )
}

mlflow_rest_body <- function(data) {
  paste0(
    "\"",
    gsub(
      "\\\"",
      "\\\\\"",
      as.character(
        jsonlite::toJSON(data, auto_unbox = TRUE)
      )
    ),
    "\""
  )
}

#' @importFrom httr add_headers
mlflow_rest_headers <- function() {
  add_headers("Content-Type" = "application/json")
}

#' @importFrom httr content
#' @importFrom httr GET
#' @importFrom httr POST
#' @importFrom jsonlite fromJSON
mlflow_rest <- function(..., data = NULL, verb = "GET", version = "2.0") {
  args <- list(...)

  api_url <- file.path(
    mlflow_tracking_url_get(),
    mlflow_rest_path(version),
    paste(args, collapse = "/")
  )

  response <- switch(
    verb,
    GET = GET(api_url),
    POST = POST(api_url,
                body = mlflow_rest_body(data),
                mlflow_rest_headers()),
    stop("Verb '", verb, "' is unsupported.")
  )

  text <- content(response, "text", encoding = "UTF-8")
  fromJSON(text)
}
