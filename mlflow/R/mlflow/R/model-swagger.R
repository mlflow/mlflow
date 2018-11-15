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
    swagger = jsonlite::unbox("2.0"),
    info = list(
      description = jsonlite::unbox("API to MLflow Model."),
      version = jsonlite::unbox("1.0.0"),
      title = jsonlite::unbox("MLflow Model")
    ),
    basePath = jsonlite::unbox("/"),
    schemes = list(
      jsonlite::unbox("http")
    )
  )
}

swagger_path <- function() {
  list(
    post = list(
      summary = jsonlite::unbox(paste0("Perform prediction")),
      description = jsonlite::unbox(""),
      consumes = list(
        jsonlite::unbox("application/json")
      ),
      produces = list(
        jsonlite::unbox("application/json")
      ),
      parameters = list(
        list(
          "in" = jsonlite::unbox("body"),
          name = jsonlite::unbox("body"),
          description = jsonlite::unbox(paste0("Prediction instances for model")),
          required = jsonlite::unbox(TRUE),
          schema = list(
            "$ref" = jsonlite::unbox(paste0("#/definitions/Type"))
          )
        )
      ),
      responses = list(
        "200" = list(
          description = jsonlite::unbox("Success")
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
        type = jsonlite::unbox("object")
      )
    )
  )
}

# nocov end
