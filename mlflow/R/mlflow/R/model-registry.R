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

mlflow_model_has_uc_signature <- function(model_spec) {
  sig <- model_spec$signature
  if (is.null(sig) || is.null(sig$outputs)) {
    return(FALSE)
  }
  if (is.character(sig$outputs)) {
    return(any(!is.na(sig$outputs) & nzchar(sig$outputs)))
  }
  length(sig$outputs) > 0
}

mlflow_validate_uc_model_signature <- function(model_dir) {
  spec_file <- fs::path(model_dir, "MLmodel")
  if (!file.exists(spec_file)) {
    stop("Model directory must include an MLmodel file for Unity Catalog registration.", call. = FALSE)
  }
  spec <- yaml::read_yaml(spec_file)
  if (!mlflow_model_has_uc_signature(spec)) {
    stop(
      "Unity Catalog model versions must include a model signature with output type specifications.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

mlflow_materialize_local_model <- function(source, client) {
  if (dir.exists(source)) {
    return(normalizePath(source, winslash = "/", mustWork = TRUE))
  }

  mlflow_download_artifacts_from_uri(source, client = client)
}

mlflow_registry_headers <- function(headers) {
  named_headers <- character()
  for (h in headers %||% list()) {
    key <- h$key %||% h$name
    value <- h$value
    if (!is.null(key) && !is.null(value)) {
      named_headers[key] <- value
    }
  }
  named_headers
}

mlflow_upload_file_to_signed_url <- function(url, local_file, headers = character()) {
  req_headers <- httr::add_headers(.headers = headers)
  body <- httr::upload_file(local_file, type = "application/octet-stream")
  resp <- httr::PUT(url, body = body, mlflow_rest_timeout(), req_headers)
  if (resp$status_code >= 400) {
    stop(sprintf("Signed URL upload failed with HTTP %s.", resp$status_code), call. = FALSE)
  }
  invisible(TRUE)
}

mlflow_download_file_from_signed_url <- function(url, local_file, headers = character()) {
  req_headers <- httr::add_headers(.headers = headers)
  dir.create(dirname(local_file), recursive = TRUE, showWarnings = FALSE)
  write_disk <- httr::write_disk(local_file, overwrite = TRUE)
  resp <- httr::GET(url, mlflow_rest_timeout(), write_disk, req_headers)
  if (resp$status_code >= 400) {
    unlink(local_file)
    stop(sprintf("Signed URL download failed with HTTP %s.", resp$status_code), call. = FALSE)
  }
  local_file
}

mlflow_uc_databricks_file_path <- function(model_name, version, artifact_path = NULL) {
  parts <- c("/Models", strsplit(model_name, "\\.")[[1]], cast_string(version))
  if (!is.null(artifact_path) && nchar(artifact_path)) {
    parts <- c(parts, artifact_path)
  }
  paste(parts, collapse = "/")
}

mlflow_uc_uses_default_storage <- function(credentials) {
  storage_mode <- credentials$storage_mode %||% credentials$storageMode
  identical(storage_mode, "DEFAULT_STORAGE")
}

mlflow_uc_sdk_models_artifact_repository_enabled <- function(client) {
  tryCatch({
    response <- mlflow_registry_rest(
      "registered-models:is-databricks-sdk-models-artifact-repository-enabled",
      client = client,
      verb = "GET",
      version = "2.0"
    )
    isTRUE(
      response$is_databricks_sdk_models_artifact_repository_enabled %||%
        response$isDatabricksSdkModelsArtifactRepositoryEnabled
    )
  }, error = function(e) {
    FALSE
  })
}

mlflow_uc_model_version_storage <- function(model_version) {
  model_version$storage_location %||% model_version$storageLocation %||% NULL
}

mlflow_uc_local_model_files <- function(model_dir) {
  if (!dir.exists(model_dir)) {
    stop("Unity Catalog upload requires a local model directory.", call. = FALSE)
  }
  files <- list.files(model_dir, recursive = TRUE, all.files = TRUE, full.names = TRUE, no.. = TRUE)
  files <- files[file.info(files)$isdir == FALSE]
  if (length(files) == 0) {
    stop("Unity Catalog upload model directory is empty.", call. = FALSE)
  }
  files
}

mlflow_uc_model_version_credentials <- function(name, version, operation, client) {
  response <- mlflow_registry_rest(
    "model-versions",
    "generate-temporary-credentials",
    client = client,
    verb = "POST",
    version = "2.0",
    data = list(
      name = name,
      version = cast_string(version),
      operation = operation
    )
  )
  response$credentials %||% list()
}

mlflow_uc_create_upload_url <- function(client, path) {
  response <- mlflow_rest(
    "fs",
    "create-upload-url",
    client = client,
    verb = "POST",
    version = "2.0",
    path_prefix = "api/2.0",
    data = list(path = path)
  )
  list(
    url = response$url,
    headers = mlflow_registry_headers(response$headers)
  )
}

mlflow_uc_create_download_url <- function(client, path) {
  response <- mlflow_rest(
    "fs",
    "create-download-url",
    client = client,
    verb = "POST",
    version = "2.0",
    path_prefix = "api/2.0",
    data = list(path = path)
  )
  list(
    url = response$url,
    headers = mlflow_registry_headers(response$headers)
  )
}

mlflow_uc_list_databricks_file_dir <- function(client, path, page_token = NULL) {
  data <- if (!is.null(page_token) && nchar(page_token)) {
    list(page_token = page_token)
  } else {
    NULL
  }
  mlflow_rest(
    "fs",
    "directories",
    sub("^/+", "", path),
    client = client,
    verb = "GET",
    version = "2.0",
    path_prefix = "api/2.0",
    data = data
  )
}

mlflow_uc_databricks_files <- function(client, root) {
  collect_dir <- function(path) {
    files <- character()
    next_token <- NULL
    repeat {
      response <- mlflow_uc_list_databricks_file_dir(client, path, page_token = next_token)
      for (entry in response$contents %||% list()) {
        entry_path <- entry$path
        if (is.null(entry_path) || !nchar(entry_path)) next

        if (isTRUE(entry$is_directory %||% entry$isDirectory)) {
          files <- c(files, collect_dir(sub("/+$", "", entry_path)))
        } else {
          files <- c(files, entry_path)
        }
      }

      next_token <- response$next_page_token %||% response$nextPageToken
      if (is.null(next_token) || !nchar(next_token)) break
    }
    files
  }

  collect_dir(root)
}

mlflow_upload_model_dir_to_uc_databricks_files <- function(model_dir, model_version, client, files) {
  if (is.null(model_version$name) || is.null(model_version$version)) {
    stop("Unity Catalog Databricks file upload requires model name and version.", call. = FALSE)
  }
  registry_client <- resolve_registry_client(client)
  for (f in files) {
    rel <- fs::path_rel(f, start = model_dir)
    rel <- gsub("\\\\", "/", rel)
    remote <- mlflow_uc_databricks_file_path(model_version$name, model_version$version, rel)
    upload <- mlflow_uc_create_upload_url(registry_client, remote)
    mlflow_upload_file_to_signed_url(upload$url, f, headers = upload$headers)
  }
  invisible(TRUE)
}

mlflow_download_model_dir_from_uc_databricks_files <- function(model_version, client) {
  if (is.null(model_version$name) || is.null(model_version$version)) {
    stop("Unity Catalog Databricks file download requires model name and version.", call. = FALSE)
  }

  registry_client <- resolve_registry_client(client)
  root <- mlflow_uc_databricks_file_path(model_version$name, model_version$version)
  files <- mlflow_uc_databricks_files(registry_client, root)
  if (length(files) == 0) {
    stop(sprintf("No Unity Catalog model files found under `%s`.", root), call. = FALSE)
  }

  output_dir <- tempfile(pattern = "mlflow-uc-model-")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  prefix <- paste0(sub("/+$", "", root), "/")

  for (remote in files) {
    rel <- if (startsWith(remote, prefix)) {
      substr(remote, nchar(prefix) + 1, nchar(remote))
    } else {
      basename(remote)
    }
    rel <- sub("^/+", "", rel)
    if (!nchar(rel)) next

    local_file <- file.path(output_dir, rel)
    download <- mlflow_uc_create_download_url(registry_client, remote)
    mlflow_download_file_from_signed_url(download$url, local_file, headers = download$headers)
  }

  output_dir
}

mlflow_uc_python_artifact_code <- function() {
  paste(c(
    "import json",
    "import os",
    "import sys",
    "",
    "def configure_mlflow():",
    "    import mlflow",
    "    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')",
    "    registry_uri = os.environ.get('MLFLOW_REGISTRY_URI')",
    "    if tracking_uri:",
    "        mlflow.set_tracking_uri(tracking_uri)",
    "    if registry_uri:",
    "        mlflow.set_registry_uri(registry_uri)",
    "    return mlflow",
    "",
    "def model_version_to_dict(mv):",
    "    keys = [",
    "        'name', 'version', 'creation_timestamp', 'last_updated_timestamp',",
    "        'description', 'user_id', 'current_stage', 'source', 'run_id',",
    "        'status', 'status_message', 'run_link', 'aliases', 'model_id',",
    "    ]",
    "    result = {}",
    "    for key in keys:",
    "        value = getattr(mv, key, None)",
    "        if value is not None:",
    "            result[key] = list(value) if isinstance(value, tuple) else value",
    "    tags = getattr(mv, 'tags', None)",
    "    if tags:",
    "        result['tags'] = [{'key': str(k), 'value': str(v)} for k, v in tags.items()]",
    "    return result",
    "",
    "def upload_and_finalize(payload):",
    "    configure_mlflow()",
    "    from mlflow.protos.databricks_uc_registry_messages_pb2 import GetModelVersionRequest",
    "    from mlflow.tracking import MlflowClient",
    "    from mlflow.utils._unity_catalog_utils import model_version_from_uc_proto",
    "    from mlflow.utils.proto_json_utils import message_to_json",
    "    name = payload['name']",
    "    version = str(payload['version'])",
    "    client = MlflowClient()",
    "    store = client._get_registry_client().store",
    "    request = GetModelVersionRequest(name=name, version=version)",
    "    response = store._call_endpoint(GetModelVersionRequest, message_to_json(request))",
    "    model_version = response.model_version",
    "    repo = store._get_artifact_repo(model_version, name)",
    "    repo.log_artifacts(local_dir=payload['local_model_path'], artifact_path='')",
    "    finalized = store._finalize_model_version(name=name, version=str(model_version.version))",
    "    return {'model_version': model_version_to_dict(model_version_from_uc_proto(finalized))}",
    "",
    "def download(payload):",
    "    mlflow = configure_mlflow()",
    "    model_uri = 'models:/{}/{}'.format(payload['name'], payload['version'])",
    "    return {'path': mlflow.artifacts.download_artifacts(artifact_uri=model_uri)}",
    "",
    "with open(sys.argv[1], encoding='utf-8') as handle:",
    "    payload = json.load(handle)",
    "if payload['operation'] == 'upload_and_finalize':",
    "    result = upload_and_finalize(payload)",
    "elif payload['operation'] == 'download':",
    "    result = download(payload)",
    "else:",
    "    raise ValueError('Unsupported operation: {}'.format(payload['operation']))",
    "with open(sys.argv[2], 'w', encoding='utf-8') as handle:",
    "    json.dump(result, handle)"
  ), collapse = "\n")
}

mlflow_uc_python_process_uri <- function(uri, default) {
  if (is.null(uri)) return(default)
  scheme <- uri$scheme %||% NA_character_
  path <- uri$path %||% NA_character_
  if (!is.na(scheme) && scheme %in% c("databricks", "databricks-uc") &&
      (is.na(path) || !nchar(path))) {
    return(scheme)
  }
  uri$raw_uri %||% default
}

mlflow_uc_python_process_env <- function(client) {
  env <- if (is.null(client)) list() else client$get_cli_env()
  tracking_uri <- if (is.null(client)) {
    mlflow_get_tracking_uri()
  } else {
    mlflow_uc_python_process_uri(client$tracking_uri, mlflow_get_tracking_uri())
  }
  registry_uri <- if (is.null(client)) {
    mlflow_get_registry_uri()
  } else {
    mlflow_uc_python_process_uri(client$registry_uri, mlflow_get_registry_uri())
  }
  modifyList(list(
    MLFLOW_TRACKING_URI = tracking_uri,
    MLFLOW_REGISTRY_URI = registry_uri
  ), env)
}

mlflow_uc_run_python_artifact_bridge <- function(payload, client) {
  payload_file <- tempfile(fileext = ".json")
  output_file <- tempfile(fileext = ".json")
  on.exit(unlink(c(payload_file, output_file)), add = TRUE)

  jsonlite::write_json(payload, payload_file, auto_unbox = TRUE, null = "null")

  tryCatch({
    withr::with_envvar(mlflow_uc_python_process_env(client), {
      processx::run(
        python_bin(),
        c("-c", mlflow_uc_python_artifact_code(), payload_file, output_file),
        echo = mlflow_is_verbose(),
        echo_cmd = mlflow_is_verbose()
      )
    })
  }, error = function(e) {
    stop(
      "Python MLflow failed to handle Unity Catalog model artifacts: ",
      conditionMessage(e),
      call. = FALSE
    )
  })

  if (!file.exists(output_file)) {
    stop("Python MLflow did not return a Unity Catalog artifact result.", call. = FALSE)
  }

  jsonlite::fromJSON(output_file, simplifyVector = FALSE)
}

mlflow_upload_and_finalize_uc_model_version_with_python <- function(model_dir, model_version,
                                                                    client) {
  if (is.null(model_version$name) || is.null(model_version$version)) {
    stop("Unity Catalog Python upload requires model name and version.", call. = FALSE)
  }

  response <- mlflow_uc_run_python_artifact_bridge(
    list(
      operation = "upload_and_finalize",
      name = model_version$name,
      version = cast_string(model_version$version),
      local_model_path = model_dir
    ),
    client = client
  )

  response$model_version %||% model_version
}

mlflow_download_uc_model_version_with_python <- function(name, version, client) {
  response <- mlflow_uc_run_python_artifact_bridge(
    list(
      operation = "download",
      name = name,
      version = cast_string(version)
    ),
    client = client
  )
  path <- response$path
  if (is.null(path) || !nchar(path)) {
    stop("Python MLflow did not return a downloaded Unity Catalog model path.", call. = FALSE)
  }
  path
}

mlflow_download_uc_model_version <- function(name, version, client = NULL) {
  client <- resolve_client(client)
  model_version <- mlflow_get_model_version(name, version, client = client)

  if (mlflow_uc_sdk_models_artifact_repository_enabled(client)) {
    return(mlflow_download_model_dir_from_uc_databricks_files(model_version, client))
  }

  credentials <- mlflow_uc_model_version_credentials(
    name = name,
    version = model_version$version %||% version,
    operation = "MODEL_VERSION_OPERATION_READ",
    client = client
  )

  if (mlflow_uc_uses_default_storage(credentials)) {
    return(mlflow_download_model_dir_from_uc_databricks_files(model_version, client))
  }

  if (length(credentials) > 0) {
    return(
      mlflow_download_uc_model_version_with_python(
        model_version$name %||% name,
        model_version$version %||% version,
        client = client
      )
    )
  }

  storage <- mlflow_uc_model_version_storage(model_version) %||%
    mlflow_get_model_version_download_uri(name, model_version$version %||% version, client = client)

  mlflow_download_artifacts_from_uri(storage, client = client)
}

mlflow_upload_model_dir_for_uc <- function(model_dir, model_version, credentials, client = NULL) {
  files <- mlflow_uc_local_model_files(model_dir)

  if (mlflow_uc_uses_default_storage(credentials)) {
    return(mlflow_upload_model_dir_to_uc_databricks_files(
      model_dir = model_dir,
      model_version = model_version,
      client = client,
      files = files
    ))
  }

  stop(
    "Unity Catalog returned non-default model-version storage credentials. ",
    "The model version was not finalized because direct R uploads require ",
    "Databricks file URLs. Use Python MLflow for customer-managed UC storage.",
    call. = FALSE
  )
}

mlflow_uc_create_model_version <- function(name, source, run_id = NULL, tags = NULL, run_link = NULL,
                                           description = NULL, client = NULL) {
  client <- resolve_client(client)
  local_model <- mlflow_materialize_local_model(source, client = client)
  mlflow_validate_uc_model_signature(local_model)

  create_resp <- mlflow_registry_rest(
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

  model_version <- create_resp$model_version
  model_version$name <- model_version$name %||% name

  if (mlflow_uc_sdk_models_artifact_repository_enabled(client)) {
    mlflow_upload_model_dir_to_uc_databricks_files(
      model_dir = local_model,
      model_version = model_version,
      client = client,
      files = mlflow_uc_local_model_files(local_model)
    )
  } else {
    credentials <- mlflow_uc_model_version_credentials(
      name = name,
      version = model_version$version,
      operation = "MODEL_VERSION_OPERATION_READ_WRITE",
      client = client
    )

    if (!mlflow_uc_uses_default_storage(credentials) && length(credentials) > 0) {
      return(mlflow_upload_and_finalize_uc_model_version_with_python(
        local_model,
        model_version,
        client = client
      ))
    }

    mlflow_upload_model_dir_for_uc(local_model, model_version, credentials, client = client)
  }

  finalize_resp <- mlflow_registry_rest(
    "model-versions",
    "finalize",
    client = client,
    verb = "POST",
    version = "2.0",
    data = list(
      name = name,
      version = cast_string(model_version$version)
    )
  )
  finalize_resp$model_version %||% model_version
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
