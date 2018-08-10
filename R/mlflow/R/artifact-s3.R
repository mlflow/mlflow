mlflow_store_s3 <- function(path, artifact_uri) {
  file_name <- basename(path)

  bucket <- mlflow_parse_bucket(artifact_uri)

  aws.s3::put_object(
    path,
    object = fs::path(bucket$path, file_name),
    bucket = bucket$name,
    check_region = TRUE
  )

  invisible(NULL)
}
