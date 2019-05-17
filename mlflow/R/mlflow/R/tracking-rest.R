mlflow_rest_path <- function(version) {
  switch(
    version,
    "2.0" = "api/2.0/preview/mlflow"
  )
}

#' @importFrom httr timeout
mlflow_rest_timeout <- function() {
  timeout(getOption("mlflow.rest.timeout", 60))
}

try_parse_response_as_text <- function(response) {
  raw_content <- content(response, type = "raw")
  tryCatch({
    rawToChar(raw_content)
  }, error = function(e) {
    do.call(paste, as.list(raw_content))
  })
}

#' @importFrom base64enc base64encode
get_rest_config <- function(host_creds) {
  headers <- list()
  auth_header <- if (!is.na(host_creds$username) && !is.na(host_creds$password)) {
    basic_auth_str <- paste(host_creds$username, host_creds$password, sep = ":")
    paste("Basic", base64encode(charToRaw(basic_auth_str)), sep = " ")
  } else if (!is.na(host_creds$token)) {
    paste("Bearer", host_creds$token, sep = " ")
  } else {
    NA
  }
  if (!is.na(auth_header)) {
    headers$Authorization <- auth_header
  }

  headers$`User-Agent` <- paste("mlflow-r-client", utils::packageVersion("mlflow"), sep = "/")

  is_insecure <- list(true = TRUE, false = FALSE)[[tolower(host_creds$insecure)]]
  list(
    headers = headers,
    config = if (is_insecure) {
      httr::config(ssl_verifypeer = 0, ssl_verifyhost = 0)
    } else {
      list()
    }
  )
}

#' @importFrom httr GET POST add_headers config content
mlflow_rest <- function( ..., client, query = NULL, data = NULL, verb = "GET", version = "2.0") {
  host_creds <- client$get_host_creds()
  rest_config <- get_rest_config(host_creds)
  args <- list(...)
  api_url <- file.path(
    host_creds$host,
    mlflow_rest_path(version),
    paste(args, collapse = "/")
  )

  response <- switch(
    verb,
    GET = GET(
      api_url,
      query = query,
      mlflow_rest_timeout(),
      config = rest_config$config,
      do.call(add_headers, rest_config$headers)),
    POST = POST(
      api_url,
      body = if (is.null(data)) NULL else rapply(data, as.character, how = "replace"),
      encode = "json",
      mlflow_rest_timeout(),
      config = rest_config$config,
      do.call(add_headers, rest_config$headers)
    ),
    stop("Verb '", verb, "' is unsupported.", call. = FALSE)
  )
  if (response$status_code != 200) {
    message_body <- tryCatch(
      paste(content(response, "parsed", type = "application/json"), collapse = "; "),
      error = function(e) {
        try_parse_response_as_text(response)
      }
    )
    msg <- paste("API request to endpoint '",
                 paste(args, collapse = "/"),
                 "' failed with error code ",
                 response$status_code,
                 ". Reponse body: '",
                 message_body,
                 "'",
                 sep = "")
    stop(msg, call. = FALSE)
  }
  text <- content(response, "text", encoding = "UTF-8")
  jsonlite::fromJSON(text, simplifyVector = FALSE)
}
