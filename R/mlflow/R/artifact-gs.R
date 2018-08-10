mlflow_store_gs <- function(path, artifact_uri) {
  if (!"googleCloudStorageR" %in% installed.packages()) {
    stop("The package 'googleCloudStorageR' is currently required but not installed, ",
         "please install using install.packages(\"googleCloudStorageR\").")
  }

  gcs_upload <- get("gcs_upload", envir = asNamespace("googleCloudStorageR"))

  file_name <- basename(path)

  bucket <- mlflow_parse_bucket(artifact_uri)

  gcs_upload(file = filename,
             bucket = bucket$name,
             name = fs::path(bucket$path, file_name))

  invisible(NULL)
}
