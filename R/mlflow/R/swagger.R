#' @import jsonlite
# nocov start

mlflow_swagger <- function() {
  def <- c(
    swagger_header(),
    swagger_paths(),
    swagger_defs()
  )

  jsonlite::toJSON(def)
}

swagger_header <- function() {
  list(
    swagger = unbox("2.0"),
    info = list(
      description = unbox("API to MLflow Model."),
      version = unbox("1.0.0"),
      title = unbox("MLflow Model")
    ),
    basePath = unbox("/"),
    schemes = list(
      unbox("http")
    )
  )
}

swagger_path <- function() {
  list(
    post = list(
      summary = unbox(paste0("Perform prediction")),
      description = unbox(""),
      consumes = list(
        unbox("application/json")
      ),
      produces = list(
        unbox("application/json")
      ),
      parameters = list(
        list(
          "in" = unbox("body"),
          name = unbox("body"),
          description = unbox(paste0("Prediction instances for model")),
          required = unbox(TRUE),
          schema = list(
            "$ref" = unbox(paste0("#/definitions/Type"))
          )
        )
      ),
      responses = list(
        "200" = list(
          description = unbox("Success")
        )
      )
    )
  )
}

swagger_paths <- function() {
  list(
    paths = list(
      "/predict/" = swagger_path()
    )
  )
}

swagger_defs <- function() {
  list(
    definitions = list(
      Type = list(
        type = unbox("object")
      )
    )
  )
}

# nocov end
