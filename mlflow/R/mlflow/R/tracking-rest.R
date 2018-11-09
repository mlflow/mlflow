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
  timeout(getOption("mlflow.rest.timeout", 60))
}

#' @importFrom httr content
#' @importFrom httr GET
#' @importFrom httr POST
#' @importFrom jsonlite fromJSON
mlflow_rest <- function(..., client, query = NULL, data = NULL, verb = "GET", version = "2.0") {
  args <- list(...)
  tracking_url <- client$server_url

  api_url <- file.path(
    tracking_url,
    mlflow_rest_path(version),
    paste(args, collapse = "/")
  )

  response <- switch(
    verb,
    GET = GET(api_url, query = query, mlflow_rest_timeout()),
    POST = POST(
      api_url,
      body = mlflow_rest_body(data),
      mlflow_rest_headers(),
      mlflow_rest_timeout()
    ),
    stop("Verb '", verb, "' is unsupported.")
  )

  if (identical(response$status_code, 500L)) {
    stop(xml2::as_list(content(response))$html$body$p[[1]])
  }

  text <- content(response, "text", encoding = "UTF-8")
  jsonlite::fromJSON(text)
}
