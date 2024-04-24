# nocov start

#' Serve an RFunc MLflow Model
#'
#' Serves an RFunc MLflow model as a local REST API server. This interface provides similar
#' functionality to ``mlflow models serve`` cli command, however, it can only be used to deploy
#' models that include RFunc flavor. The deployed server supports standard mlflow models interface
#' with /ping and /invocation endpoints. In addition, R function models also support deprecated
#' /predict endpoint for generating predictions. The /predict endpoint will be removed in a future
#' version of mlflow.
#'
#' @template roxlate-model-uri
#' @param host Address to use to serve model, as a string.
#' @param port Port to use to serve model, as numeric.
#' @param daemonized Makes `httpuv` server daemonized so R interactive sessions
#'   are not blocked to handle requests. To terminate a daemonized server, call
#'   `httpuv::stopDaemonizedServer()` with the handle returned from this call.
#' @param ... Optional arguments passed to `mlflow_predict()`.
#' @param browse Launch browser with serving landing page?
#'
#' @examples
#' \dontrun{
#' library(mlflow)
#'
#' # save simple model with constant prediction
#' mlflow_save_model(function(df) 1, "mlflow_constant")
#'
#' # serve an existing model over a web interface
#' mlflow_rfunc_serve("mlflow_constant")
#'
#' # request prediction from server
#' httr::POST("http://127.0.0.1:8090/predict/")
#' }
#' @importFrom httpuv runServer
#' @importFrom httpuv startDaemonizedServer
#' @importFrom jsonlite fromJSON
#' @import swagger
#' @export
mlflow_rfunc_serve <- function(model_uri,
                               host = "127.0.0.1",
                               port = 8090,
                               daemonized = FALSE,
                               browse = !daemonized,
                               ...) {
  model_path <- mlflow_download_artifacts_from_uri(model_uri)
  httpuv_start <- if (daemonized) httpuv::startDaemonizedServer else httpuv::runServer
  serve_run(model_path, host, port, httpuv_start, browse && interactive(), ...)
}

serve_content_type <- function(file_path) {
  file_split <- strsplit(file_path, split = "\\.")[[1]]
  switch(file_split[[length(file_split)]],
    "css" = "text/css",
    "html" = "text/html",
    "js" = "application/javascript",
    "json" = "application/json",
    "map" = "text/plain",
    "png" = "image/png"
  )
}

serve_static_file_response <- function(package, file_path, replace = NULL) {
  mlflow_verbose_message("Serving static file: ", file_path)

  file_path <- system.file(file_path, package = package)
  file_contents <- if (file.exists(file_path)) readBin(file_path, "raw", n = file.info(file_path)$size) else NULL

  if (!is.null(remove)) {
    contents <- rawToChar(file_contents)
    for (r in names(replace)) {
      contents <- sub(r, replace[[r]], contents)
    }
    file_contents <- charToRaw(enc2utf8(contents))
  }

  list(
    status = 200L,
    headers = list(
      "Content-Type" = paste0(serve_content_type(file_path))
    ),
    body = file_contents
  )
}

serve_invalid_request <- function(message = NULL) {
  list(
    status = 404L,
    headers = list(
      "Content-Type" = "text/plain; charset=UTF-8"
    ),
    body = charToRaw(enc2utf8(
      paste(
        "Invalid Request. ",
        message
      )
    ))
  )
}

serve_prediction <- function(json_raw, model, ...) {
  mlflow_verbose_message("Serving prediction: ", json_raw)
  df <- data.frame()
  if (length(json_raw) > 0) {
    df <- jsonlite::fromJSON(
      rawToChar(json_raw),
      simplifyDataFrame = FALSE,
      simplifyMatrix = FALSE
    )
  }

  df <- as.data.frame(df)

  mlflow_predict(model, df, ...)
}

serve_empty_page <- function(req, sess, model) {
  list(
    status = 200L,
    headers = list(
      "Content-Type" = "text/html"
    ),
    body = "<html></html>"
  )
}

serve_handlers <- function(host, port, ...) {
  handlers <- list(
    "^/swagger.json" = function(req, model) {
      list(
        status = 200L,
        headers = list(
          "Content-Type" = paste0(serve_content_type("json"), "; charset=UTF-8")
        ),
        body = charToRaw(enc2utf8(
          mlflow_swagger()
        ))
      )
    },
    "^/$" = function(req, model) {
      serve_static_file_response(
        "swagger",
        "dist/index.html",
        list(
          "http://petstore\\.swagger\\.io/v2" = "",
          "layout: \"StandaloneLayout\"" = "layout: \"StandaloneLayout\",\nvalidatorUrl : false"
        )
      )
    },
    "^/predict" = function(req, model) {
      json_raw <- req$rook.input$read()

      results <- serve_prediction(json_raw, model, ...)

      list(
        status = 200L,
        headers = list(
          "Content-Type" = paste0(serve_content_type("json"), "; charset=UTF-8")
        ),
        body = charToRaw(enc2utf8(
          jsonlite::toJSON(results, auto_unbox = TRUE, digits = NA)
        ))
      )
    },
    "^/ping" = function(req, model) {
      if (!is.na(model) && !is.null(model)) {
        res <- list(status = 200L,
             headers = list(
                 "Content-Type" = paste0(serve_content_type("json"), "; charset=UTF-8")
             ),
             body = ""
        )
        res
      } else {
        list(status = 404L,
             headers = list(
                 "Content-Type" = paste0(serve_content_type("json"), "; charset=UTF-8")
             )
        )
      }
    },
    "^/invocation" = function(req, model) {
      data_raw <- rawToChar(req$rook.input$read())
      headers <- strsplit(req$HEADERS, "\n")
      content_type <- headers$`content-type` %||% "application/json"
      df <- switch( content_type,
        "application/json" = parse_json(data_raw),
        "text/csv" = utils::read.csv(text = data_raw, stringsAsFactors = FALSE),
        stop("Unsupported input format.")
      )
      results <- mlflow_predict(model, df, ...)
      list(
        status = 200L,
        headers = list(
          "Content-Type" = paste0(serve_content_type("json"), "; charset=UTF-8")
        ),
        body = charToRaw(enc2utf8(
          jsonlite::toJSON(results, auto_unbox = TRUE, digits = NA, simplifyVector = TRUE)
        ))
      )
    },
    "^/[^/]*$" = function(req, model) {
      serve_static_file_response("swagger", file.path("dist", req$PATH_INFO))
    },
    ".*" = function(req, sess, model) {
      stop("Invalid path.")
    }
  )

  if (!getOption("mlflow.swagger", default = TRUE)) {
    handlers[["^/swagger.json"]] <- serve_empty_page
    handlers[["^/$"]] <- serve_empty_page
  }

  handlers
}

message_serve_start <- function(host, port, model) {
  hostname <- paste("http://", host, ":", port, sep = "")

  message()
  message("Starting serving endpoint: ", hostname)
}

#' @importFrom utils browseURL
serve_run <- function(model_path, host, port, start, browse, ...) {
  model <- mlflow_load_model(model_path)

  message_serve_start(host, port, model)

  if (browse) browseURL(paste0("http://", host, ":", port))

  handlers <- serve_handlers(host, port, ...)

  start(host, port, list(
    onHeaders = function(req) {
      NULL
    },
    call = function(req) {
      tryCatch({
        matches <- sapply(names(handlers), function(e) grepl(e, req$PATH_INFO))
        handlers[matches][[1]](req, model)
      }, error = function(e) {
        serve_invalid_request(e$message)
      })
    },
    onWSOpen = function(ws) {
      NULL
    }
  ))
}

# nocov end
