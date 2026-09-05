#' Wrap an R object in preparation for upload to `MLflow`
#'
#' `rblob` is a generic flavor that can store any R object. Only one object can
#' be stored - if you want to store multiple objects you can put them in a `list()`.
#'
#' @param object The object to wrap and store in MLflow.
#'
#' @export
#'
#' @return An `rblob` object ready for use with MLflow.
rblob <- function(object) {
  class(object) <- c("rblob", class(object))
  object
}

#' @export
mlflow_save_model.rblob <- function(model, path, model_spec = list(), ...) {
  if (dir.exists(path)) unlink(path, recursive = TRUE)
  dir.create(path)

  serialized <- serialize(model, NULL)
  saveRDS(serialized, file.path(path, "rblob.Rds"))

  model_spec$flavors <- append(
    model_spec$flavors,
    list(
      rblob = list(
        version = "0.0.1",
        model = "rblob.Rds"
      )
    )
  )

  mlflow_write_model_spec(path, model_spec)
  model_spec
}

#' @export
mlflow_load_flavor.mlflow_flavor_rblob <- function(flavor, model_path) {
  unserialize(readRDS(file.path(model_path, "rblob.Rds")))
}
