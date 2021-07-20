#' Create registered model
#'
#' Creates a new registered model in the model registry
#'
#' @param name The name of the model to create.
#' @param tags Additional metadata for the registered model (Optional).
#' @param description Description for the registered model (Optional).
#' @template roxlate-client
#' @export
mlflow_create_registered_model <- function(name, tags = NULL, description = NULL, client = NULL) {
  client <- resolve_client(client)

  response <- mlflow_rest(
    "registered-models",
    "create",
    client = client,
    verb = "POST",
    version = "2.0",
    data = list(
      name = forge::cast_string(name),
      tags = tags,
      description = description
    )
  )

  return(response$registered_model)
}

#' Delete registered model
#'
#' Deletes an existing registered model by name
#'
#' @param name The name of the model to delete
#' @template roxlate-client
#' @export
mlflow_delete_registered_model <- function(name, client = NULL) {
  client <- resolve_client(client)

  response <- mlflow_rest(
    "registered-models",
    "delete",
    client = client,
    verb = "DELETE",
    version = "2.0",
    data = list(name = forge::cast_string(name))
  )
}
