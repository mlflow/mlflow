# Convert named `tags` lists into REST key/value tag objects.
mlflow_registry_tags_payload <- function(tags) {
  if (is.null(tags)) return(NULL)
  if (is.list(tags) && !is.null(names(tags))) {
    return(unname(purrr::imap(tags, ~ list(
      key = cast_string(.y),
      value = cast_string(.x, allow_na = TRUE)
    ))))
  }
  tags
}

mlflow_python_tags <- function(tags) {
  tags <- mlflow_registry_tags_payload(tags)
  if (is.null(tags)) return(NULL)

  result <- list()
  for (tag in tags) {
    result[[cast_string(tag$key)]] <- cast_string(tag$value, allow_na = TRUE)
  }
  result
}

mlflow_uc_create_model_version_code <- function() {
  paste(c(
    "import json",
    "import sys",
    "from mlflow.tracking import MlflowClient",
    "",
    "with open(sys.argv[1], encoding='utf-8') as handle:",
    "    payload = json.load(handle)",
    "",
    "kwargs = {",
    "    key: value",
    "    for key, value in {",
    "        'name': payload['name'],",
    "        'source': payload['source'],",
    "        'run_id': payload.get('run_id'),",
    "        'tags': payload.get('tags'),",
    "        'run_link': payload.get('run_link'),",
    "        'description': payload.get('description'),",
    "    }.items()",
    "    if value is not None",
    "}",
    "model_version = MlflowClient().create_model_version(**kwargs)",
    "keys = [",
    "    'name', 'version', 'creation_timestamp', 'last_updated_timestamp',",
    "    'description', 'user_id', 'current_stage', 'source', 'run_id',",
    "    'status', 'status_message', 'run_link', 'aliases', 'model_id',",
    "]",
    "result = {}",
    "for key in keys:",
    "    value = getattr(model_version, key, None)",
    "    if value is not None:",
    "        result[key] = list(value) if isinstance(value, tuple) else value",
    "if getattr(model_version, 'tags', None):",
    "    result['tags'] = model_version.tags",
    "print(json.dumps(result))"
  ), collapse = "\n")
}

mlflow_python_json <- function(code, payload, client) {
  payload_file <- tempfile(fileext = ".json")
  on.exit(unlink(payload_file), add = TRUE)
  jsonlite::write_json(payload, payload_file, auto_unbox = TRUE, null = "null")

  env <- if (is.null(client)) list() else client$get_cli_env()
  tracking_uri <- if (is.null(client)) {
    mlflow_get_tracking_uri()
  } else {
    client$tracking_uri$raw_uri %||% mlflow_get_tracking_uri()
  }
  registry_uri <- if (is.null(client)) {
    mlflow_get_registry_uri()
  } else {
    client$registry_uri$raw_uri %||% mlflow_get_registry_uri()
  }
  env <- modifyList(list(
    MLFLOW_TRACKING_URI = tracking_uri,
    MLFLOW_REGISTRY_URI = registry_uri
  ), env)

  response <- tryCatch({
    withr::with_envvar(env, {
      run(
        python_bin(),
        c("-c", code, payload_file),
        echo = mlflow_is_verbose(),
        echo_cmd = mlflow_is_verbose()
      )
    })
  }, error = function(e) {
    stop("Python MLflow failed to handle Unity Catalog model artifacts: ",
         conditionMessage(e), call. = FALSE)
  })

  jsonlite::fromJSON(response$stdout, simplifyVector = FALSE)
}

mlflow_download_uc_model_version <- function(name, version, client = NULL) {
  client <- resolve_client(client)

  response <- mlflow_cli(
    "artifacts", "download",
    "--artifact-uri", sprintf("models:/%s/%s", name, cast_string(version)),
    client = client,
    echo = mlflow_is_verbose()
  )
  lines <- strsplit(response$stdout, "\n", fixed = TRUE)[[1]]
  lines <- lines[nzchar(lines)]
  path <- tail(lines, 1)
  if (length(path) == 0 || !nchar(path)) {
    stop("Python MLflow did not return a downloaded Unity Catalog model path.", call. = FALSE)
  }
  path
}

mlflow_uc_create_model_version <- function(name, source, run_id = NULL, tags = NULL, run_link = NULL,
                                           description = NULL, client = NULL) {
  client <- resolve_client(client)
  mlflow_python_json(
    mlflow_uc_create_model_version_code(),
    list(
      name = name,
      source = source,
      run_id = run_id,
      tags = mlflow_python_tags(tags),
      run_link = run_link,
      description = description
    ),
    client = client
  )
}

mlflow_uc_stage_error <- function(method) {
  stop(
    sprintf("`%s()` is unsupported for Unity Catalog models. ", method),
    "Use registered model aliases instead of stages.",
    call. = FALSE
  )
}

#' Create registered model
#'
#' Creates a new registered model in the model registry
#'
#' @param name The name of the model to create.
#' @param tags Additional metadata for the registered model (Optional).
#' @param description Description for the registered model (Optional).
#' @template roxlate-client
#' @export
mlflow_create_registered_model <- function(name, tags = NULL,
                                           description = NULL, client = NULL) {
  client <- resolve_client(client)

  response <- mlflow_registry_rest(
    "registered-models",
    "create",
    client = client,
    verb = "POST",
    version = "2.0",
    data = list(
      name = cast_string(name),
      tags = mlflow_registry_tags_payload(tags),
      description = description
    )
  )

  return(response$registered_model)
}

#' Get a registered model
#'
#' Retrieves a registered model from the Model Registry.
#'
#' @param name The name of the model to retrieve.
#' @template roxlate-client
#' @export
mlflow_get_registered_model <- function(name, client = NULL) {
  client <- resolve_client(client)

  response <- mlflow_registry_rest(
    "registered-models",
    "get",
    client = client,
    verb = "GET",
    version = "2.0",
    query = list(name = name)
  )

  return(response$registered_model)
}

#' Rename a registered model
#'
#' Renames a model in the Model Registry.
#'
#' @param name The current name of the model.
#' @param new_name The new name for the model.
#' @template roxlate-client
#' @export
mlflow_rename_registered_model <- function(name, new_name, client = NULL) {
  client <- resolve_client(client)

  response <- mlflow_registry_rest(
    "registered-models",
    "rename",
    client = client,
    verb = "POST",
    version = "2.0",
    data = list(
      name = cast_string(name),
      new_name = cast_string(new_name)
    )
  )

  return(response$registered_model)
}

#' Update a registered model
#'
#' Updates a model in the Model Registry.
#'
#' @param name The name of the registered model.
#' @param description The updated description for this registered model.
#' @template roxlate-client
#' @export
mlflow_update_registered_model <- function(name, description, client = NULL) {
  client <- resolve_client(client)

  response <- mlflow_registry_rest(
    "registered-models",
    "update",
    client = client,
    verb = "PATCH",
    version = "2.0",
    data = list(
      name = cast_string(name),
      description = cast_string(description)
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

  response <- mlflow_registry_rest(
    "registered-models",
    "delete",
    client = client,
    verb = "DELETE",
    version = "2.0",
    data = list(name = cast_string(name))
  )
}

#' List registered models
#'
#' Retrieves a list of registered models.
#'
#' @param filter A filter expression used to identify specific registered models.
#'   The syntax is a subset of SQL which allows only ANDing together binary operations.
#'   Example: "name = 'my_model_name' and tag.key = 'value1'"
#' @param max_results Maximum number of registered models to retrieve.
#' @param page_token Pagination token to go to the next page based on a
#'   previous query.
#' @param order_by List of registered model properties to order by. Example: "name".
#' @template roxlate-client
#' @export
mlflow_search_registered_models <- function(filter = NULL,
                                            max_results = 100,
                                            order_by = list(),
                                            page_token = NULL,
                                            client = NULL) {
  client <- resolve_client(client)

  if (is_uc_registry_uri(client)) {
    if (!is.null(filter)) {
      stop("Unity Catalog registered model search does not support `filter`.", call. = FALSE)
    }
    if (length(order_by) > 0) {
      stop("Unity Catalog registered model search does not support `order_by`.", call. = FALSE)
    }
    response <- mlflow_registry_rest(
      "registered-models",
      "search",
      client = client,
      verb = "GET",
      version = "2.0",
      query = list(
        max_results = max_results,
        page_token = page_token
      )
    )
    return(list(
      registered_models = response$registered_models %||% response$registered_model,
      next_page_token = response$next_page_token
    ))
  }

  response <- mlflow_registry_rest(
    "registered-models",
    "search",
    client = client,
    verb = "POST",
    version = "2.0",
    data = list(
      filter = filter,
      max_results = max_results,
      order_by = cast_string_list(order_by),
      page_token = page_token
    )
  )

  return(list(
    registered_models = response$registered_models %||% response$registered_model,
    next_page_token = response$next_page_token
  ))
}

#' Get latest model versions
#'
#' Retrieves a list of the latest model versions for a given model.
#'
#' @param name Name of the model.
#' @param stages A list of desired stages. If the input list is NULL, return
#'   latest versions for ALL_STAGES.
#' @template roxlate-client
#' @export
mlflow_get_latest_versions <- function(name, stages = list(), client = NULL) {
  client <- resolve_client(client)

  if (is_uc_registry_uri(client)) {
    mlflow_uc_stage_error("mlflow_get_latest_versions")
  }

  response <- mlflow_registry_rest(
    "registered-models",
    "get-latest-versions",
    client = client,
    verb = "GET",
    version = "2.0",
    query = list(
      name = cast_string(name),
      stages = cast_string_list(stages)
    )
  )

  return(response$model_versions)
}

#' Create a model version
#'
#' @param name Register model under this name.
#' @param source URI indicating the location of the model artifacts.
#' @param run_id MLflow run ID for correlation, if `source` was generated
#'   by an experiment run in MLflow Tracking.
#' @param tags Additional metadata.
#' @param run_link MLflow run link - This is the exact link of the run that
#'   generated this model version.
#' @param description Description for model version.
#' @template roxlate-client
#' @export
mlflow_create_model_version <- function(name, source, run_id = NULL,
                                        tags = NULL, run_link = NULL,
                                        description = NULL, client = NULL) {
  client <- resolve_client(client)

  if (is_uc_registry_uri(client)) {
    return(mlflow_uc_create_model_version(
      name = name,
      source = source,
      run_id = run_id,
      tags = tags,
      run_link = run_link,
      description = description,
      client = client
    ))
  }

  response <- mlflow_registry_rest(
    "model-versions",
    "create",
    client = client,
    verb = "POST",
    version = "2.0",
    data = list(
      name = name,
      source = source,
      run_id = run_id,
      run_link = run_link,
      tags = mlflow_registry_tags_payload(tags),
      description = description
    )
  )

  return(response$model_version)
}

#' Get a model version
#'
#' @param name Name of the registered model.
#' @param version Model version number.
#' @template roxlate-client
#' @export
mlflow_get_model_version <- function(name, version, client = NULL) {
  client <- resolve_client(client)

  response <- mlflow_registry_rest(
    "model-versions",
    "get",
    client = client,
    verb = "GET",
    version = "2.0",
    query = list(
      name = name,
      version = version
    )
  )

  return(response$model_version)
}

mlflow_get_model_version_download_uri <- function(name, version, client = NULL) {
  client <- resolve_client(client)
  response <- mlflow_registry_rest(
    "model-versions",
    "get-download-uri",
    client = client,
    verb = "GET",
    version = "2.0",
    query = list(
      name = name,
      version = cast_string(version)
    )
  )
  response$artifact_uri %||% response$artifactUri
}

#' Register model URI under a model name
#'
#' @param model_uri URI indicating the location of model artifacts.
#' @param name Register model under this name.
#' @param run_id MLflow run ID for correlation, if `model_uri` was generated
#'   by an experiment run in MLflow Tracking.
#' @param tags Additional metadata.
#' @param description Description for model version.
#' @param ... Additional arguments forwarded to `mlflow_create_model_version()`.
#' @template roxlate-client
#' @export
mlflow_register_model <- function(model_uri, name, run_id = NULL, tags = NULL,
                                  description = NULL, client = NULL, ...) {
  mlflow_create_model_version(
    name = name,
    source = model_uri,
    run_id = run_id,
    tags = tags,
    description = description,
    client = client,
    ...
  )
}

#' Set a model alias
#'
#' @param name Name of the registered model.
#' @param alias Alias to set.
#' @param version Model version number.
#' @param ... Reserved for future options.
#' @template roxlate-client
#' @export
mlflow_set_registered_model_alias <- function(name, alias, version, client = NULL, ...) {
  client <- resolve_client(client)
  mlflow_registry_rest(
    "registered-models",
    "alias",
    client = client,
    verb = "POST",
    version = "2.0",
    data = list(
      name = cast_string(name),
      alias = cast_string(alias),
      version = cast_string(version)
    )
  )
  invisible(NULL)
}

#' Get model version by alias
#'
#' @param name Name of the registered model.
#' @param alias Alias to resolve.
#' @param ... Reserved for future options.
#' @template roxlate-client
#' @export
mlflow_get_model_version_by_alias <- function(name, alias, client = NULL, ...) {
  client <- resolve_client(client)
  response <- mlflow_registry_rest(
    "registered-models",
    "alias",
    client = client,
    verb = "GET",
    version = "2.0",
    query = list(
      name = cast_string(name),
      alias = cast_string(alias)
    )
  )
  response$model_version %||% response
}

#' Update model version
#'
#' Updates a model version
#'
#' @param name Name of the registered model.
#' @param version Model version number.
#' @param description Description of this model version.
#' @template roxlate-client
#' @export
mlflow_update_model_version <- function(name, version, description,
                                        client = NULL) {
  client <- resolve_client(client)

  response <- mlflow_registry_rest(
    "model-versions",
    "update",
    client = client,
    verb = "PATCH",
    version = "2.0",
    data = list(
      name = name,
      version = version,
      description = description
    )
  )

  return(response$model_version)
}

#' Delete a model version
#'
#' @param name Name of the registered model.
#' @param version Model version number.
#' @template roxlate-client
#' @export
mlflow_delete_model_version <- function(name, version, client = NULL) {
  client <- resolve_client(client)

  response <- mlflow_registry_rest(
    "model-versions",
    "delete",
    client = client,
    verb = "DELETE",
    version = "2.0",
    data = list(
      name = cast_string(name),
      version = cast_string(version)
    )
  )
}

#' Transition ModelVersion Stage
#'
#' Transition a model version to a different stage.
#'
#' @param name Name of the registered model.
#' @param version Model version number.
#' @param stage Transition `model_version` to this stage.
#' @param archive_existing_versions (Optional)
#' @template roxlate-client
#' @export
mlflow_transition_model_version_stage <- function(name, version, stage,
                                                  archive_existing_versions = FALSE,
                                                  client = NULL) {
  client <- resolve_client(client)

  if (is_uc_registry_uri(client)) {
    mlflow_uc_stage_error("mlflow_transition_model_version_stage")
  }

  response <- mlflow_registry_rest(
    "model-versions",
    "transition-stage",
    client = client,
    verb = "POST",
    version = "2.0",
    data = list(
      name = name,
      version = version,
      stage = stage,
      archive_existing_versions = archive_existing_versions
    )
  )

  return(response$model_version)
}

#' Set Model version tag
#'
#' Set a tag for the model version.
#' When stage is set, tag will be set for latest model version of the stage.
#' Setting both version and stage parameter will result in error.
#'
#' @param name Registered model name.
#' @param version Registered model version.
#' @param key Tag key to log. key is required.
#' @param value Tag value to log. value is required.
#' @param stage Registered model stage.
#' @template roxlate-client
#' @export
mlflow_set_model_version_tag <- function(name, version = NULL, key = NULL, value = NULL, stage = NULL, client = NULL) {
  if (!is.null(version) && !is.null(stage)) {
    stop("version and stage cannot be set together",
      call. = FALSE
    )
  }

  if (is.null(version) && is.null(stage)) {
    stop("version or stage must be set",
      call. = FALSE
    )
  }

  client <- resolve_client(client)

  if (!is.null(stage)) {
    latest_versions <- mlflow_get_latest_versions(name = name, stages = list(stage), client = client)
    if (is.null(latest_versions)) {
      stop(sprintf("Could not find any model version for %s stage", stage),
        call. = FALSE
      )
    }
    version <- latest_versions[[1]]$version
  }

  response <- mlflow_registry_rest(
    "model-versions", "set-tag",
    client = client, verb = "POST",
    data = list(
      name = name,
      version = version,
      key = key,
      value = value
    )
  )
  invisible(NULL)
}
