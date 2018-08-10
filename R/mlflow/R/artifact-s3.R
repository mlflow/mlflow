mlflow_store_s3 <- function(path, artifact_uri) {
  file_name <- basename(path)

  match <- regexec("[^:]+://([^/]+)(.*)", artifact_uri)
  results <- regmatches(artifact_uri, match)

  bucket <- results[[1]][2]
  bucket_path <- results[[1]][3]

  aws.s3::put_object(
    path,
    object = fs::path(bucket_path, file_name),
    bucket = bucket,
    check_region = TRUE
  )

  invisible(NULL)
}
