#' Create Experiment
#'
#' Creates an MLflow experiment and returns its id.
#'
#' @param name The name of the experiment to create.
#' @param artifact_location Location where all artifacts for this experiment are stored. If
#'   not provided, the remote server will select an appropriate default.
#' @param tags Experiment tags to set on the experiment upon experiment creation.
#' @template roxlate-client
#' @export
mlflow_create_experiment <- function(name, artifact_location = NULL, client = NULL, tags = NULL) {
  client <- resolve_client(client)
  name <- cast_string(name)

  tags <- if (!is.null(tags)) tags %>%
    purrr::imap(~ list(key = .y, value = .x)) %>%
    unname()

  response <- mlflow_rest(
    "experiments", "create",
    client = client, verb = "POST",
    data = list(
      name = name,
      artifact_location = artifact_location,
      tags = tags
    )
  )
  response$experiment_id
}

#' Search Experiments
#'
#' Search for experiments that satisfy specified criteria.
#'
#' @param filter A filter expression used to identify specific experiments.
#'   The syntax is a subset of SQL which allows only ANDing together binary operations.
#'   Examples: "attribute.name = 'MyExperiment'", "tags.problem_type = 'iris_regression'"
#' @param experiment_view_type Experiment view type. Only experiments matching this view type are
#'   returned.
#' @param order_by List of properties to order by. Example: "attribute.name".
#' @param max_results Maximum number of experiments to retrieve.
#' @param page_token Pagination token to go to the next page based on a
#'   previous query.
#' @template roxlate-client
#' @export
mlflow_search_experiments <- function(filter = NULL,
                                      experiment_view_type = c(
                                        "ACTIVE_ONLY", "DELETED_ONLY", "ALL"
                                      ),
                                      max_results = 1000,
                                      order_by = list(),
                                      page_token = NULL,
                                      client = NULL) {
  client <- resolve_client(client)
  experiment_view_type <- match.arg(experiment_view_type)
  response <- mlflow_rest("experiments", "search", client = client, verb = "POST", data = list(
    filter = filter,
    view_type = experiment_view_type,
    max_results = max_results,
    order_by = cast_string_list(order_by),
    page_token = page_token
  ))

  # Return `NULL` if no experiments
  if (!length(response)) return(NULL)
  experiments <- purrr::map(response$experiments, function(x) {
    x$tags <- parse_run_data(x$tags)
    tibble::as_tibble(x)
  }) %>%
    do.call(rbind, .)

  return(list(
    experiments=experiments,
    next_page_token = response$next_page_token
  ))
}

#' Set Experiment Tag
#'
#' Sets a tag on an experiment with the specified ID. Tags are experiment metadata that can be updated.
#'
#' @param key Name of the tag. All storage backends are guaranteed to support
#'   key values up to 250 bytes in size. This field is required.
#' @param value String value of the tag being logged. All storage backends are
#'   guaranteed to support key values up to 5000 bytes in size. This field is required.
#' @param experiment_id ID of the experiment.
#' @template roxlate-client
#' @export
mlflow_set_experiment_tag <- function(key, value, experiment_id = NULL, client = NULL) {
  key <- cast_string(key)
  value <- cast_string(value)
  client <- resolve_client(client)

  experiment_id <- resolve_experiment_id(experiment_id)
  experiment_id <- cast_string(experiment_id)
  response <- mlflow_rest("experiments", "set-experiment-tag", client = client, verb = "POST", data = list(
      experiment_id = experiment_id,
      key = key,
      value = value
  ))

  invisible(NULL)
}

#' Get Experiment
#'
#' Gets metadata for an experiment and a list of runs for the experiment. Attempts to obtain the
#' active experiment if both `experiment_id` and `name` are unspecified.
#'
#' @param experiment_id ID of the experiment.
#' @param name The experiment name. Only one of `name` or `experiment_id` should be specified.
#' @template roxlate-client
#' @export
mlflow_get_experiment <- function(experiment_id = NULL, name = NULL, client = NULL) {
  if (!is.null(name) && !is.null(experiment_id)) {
    stop("Only one of `name` or `experiment_id` should be specified.", call. = FALSE)
  }
  client <- resolve_client(client)
  response <- if (!is.null(name)) {
    mlflow_rest("experiments", "get-by-name",client = client,
                query = list(experiment_name = name))
  } else {
    experiment_id <- resolve_experiment_id(experiment_id)
    experiment_id <- cast_string(experiment_id)
    response <- mlflow_rest(
      "experiments", "get",
      client = client, query = list(experiment_id = experiment_id)
    )
  }
  response$experiment$tags <- parse_run_data(response$experiment$tags)
  response$experiment %>%
    new_mlflow_experiment()
}

#' Delete Experiment
#'
#' Marks an experiment and associated runs, params, metrics, etc. for deletion. If the
#'   experiment uses FileStore, artifacts associated with experiment are also deleted.
#'
#' @param experiment_id ID of the associated experiment. This field is required.
#' @template roxlate-client
#' @export
mlflow_delete_experiment <- function(experiment_id, client = NULL) {
  if (identical(experiment_id, mlflow_get_active_experiment_id()))
    stop("Cannot delete an active experiment.", call. = FALSE)

  client <- resolve_client(client)
  mlflow_rest(
    "experiments", "delete",
    verb = "POST", client = client,
    data = list(experiment_id = experiment_id)

  )
  invisible(NULL)
}



#' Restore Experiment
#'
#' Restores an experiment marked for deletion. This also restores associated metadata,
#'   runs, metrics, and params. If experiment uses FileStore, underlying artifacts
#'   associated with experiment are also restored.
#'
#' Throws `RESOURCE_DOES_NOT_EXIST` if the experiment was never created or was permanently deleted.
#'
#' @param experiment_id ID of the associated experiment. This field is required.
#' @template roxlate-client
#' @export
mlflow_restore_experiment <- function(experiment_id, client = NULL) {
  client <- resolve_client(client)
  mlflow_rest(
    "experiments", "restore",
    client = client, verb = "POST",
    data = list(experiment_id = experiment_id)
  )
  invisible(NULL)
}

#' Rename Experiment
#'
#' Renames an experiment.
#'
#' @template roxlate-client
#' @param experiment_id ID of the associated experiment. This field is required.
#' @param new_name The experiment's name will be changed to this. The new name must be unique.
#' @export
mlflow_rename_experiment <- function(new_name, experiment_id = NULL, client = NULL) {
  experiment_id <- resolve_experiment_id(experiment_id)

  client <- resolve_client(client)
  mlflow_rest(
    "experiments", "update",
    client = client, verb = "POST",
    data = list(
      experiment_id = experiment_id,
      new_name = new_name
    )
  )
  invisible(NULL)
}

#' Set Experiment
#'
#' Sets an experiment as the active experiment. Either the name or ID of the experiment can be provided.
#'   If the a name is provided but the experiment does not exist, this function creates an experiment
#'   with provided name. Returns the ID of the active experiment.
#'
#' @param experiment_name Name of experiment to be activated.
#' @param experiment_id ID of experiment to be activated.
#' @param artifact_location Location where all artifacts for this experiment are stored. If
#'   not provided, the remote server will select an appropriate default.
#' @export
mlflow_set_experiment <- function(experiment_name = NULL, experiment_id = NULL, artifact_location = NULL) {
  if (!is.null(experiment_name) && !is.null(experiment_id)) {
    stop("Only one of `experiment_name` or `experiment_id` should be specified.",
         call. = FALSE
    )
  }

  if (is.null(experiment_name) && is.null(experiment_id)) {
    stop("Exactly one of `experiment_name` or `experiment_id` should be specified.",
         call. = FALSE)
  }

  client <- mlflow_client()

  final_experiment_id <- if (!is.null(experiment_name)) {
    tryCatch(
      mlflow_id(mlflow_get_experiment(client = client, name = experiment_name)),
      error = function(e) {
        if (grep("RESOURCE_DOES_NOT_EXIST", e$message, fixed = TRUE)) {
          message("Experiment `", experiment_name, "` does not exist. Creating a new experiment.")
          mlflow_create_experiment(client = client, name = experiment_name, artifact_location = artifact_location)
        } else {
          stop(e)
        }
      }
    )
  } else {
    experiment_id
  }
  invisible(mlflow_set_active_experiment_id(final_experiment_id))
  return(final_experiment_id)
}
