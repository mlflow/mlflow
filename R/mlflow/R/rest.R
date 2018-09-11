mlflow_rest_path <- function(version) {
  switch(
    version,
    "2.0" = "ajax-api/2.0/preview/mlflow"
  )
}

mlflow_rest_body <- function(data) {
  data <- Filter(length, data)
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

#' @importFrom httr timeout
mlflow_rest_timeout <- function() {
  timeout(getOption("mlflow.rest.timeout", 1))
}

#' @importFrom httr content
#' @importFrom httr GET
#' @importFrom httr POST
#' @importFrom jsonlite fromJSON
#' @importFrom xml2 as_list
mlflow_rest <- function(..., query = NULL, data = NULL, verb = "GET", version = "2.0") {
  args <- list(...)

  if (is.null(args$mc)) {
    mc <- mlflow_get_or_create_active_connection()
  }
  else {
    mc <- args$mc
    args$mc <- NULL
  }

  api_url <- file.path(
    mc$tracking_uri,
    mlflow_rest_path(version),
    paste(args, collapse = "/")
  )

  response <- switch(
    verb,
    GET = GET(api_url, query = query, mlflow_rest_timeout()),
    POST = POST(api_url,
                body = mlflow_rest_body(data),
                mlflow_rest_headers(),
                mlflow_rest_timeout()),
    stop("Verb '", verb, "' is unsupported.")
  )

  if (identical(response$status_code, 500L)) {
    stop(as_list(content(response))$html$body$p[[1]])
  }

  text <- content(response, "text", encoding = "UTF-8")
  jsonlite::fromJSON(text)
}
