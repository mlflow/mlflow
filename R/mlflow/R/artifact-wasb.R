mlflow_store_wasb <- function(path, artifact_uri) {
  file_name <- basename(path)
  bucket <- mlflow_parse_bucket(artifact_uri)

  processx::run("az",
                "storage",
                "blob",
                "upload",
                "--container-name",
                bucket$name,
                " --file",
                filename,
                "--name",
                bucket$path,
                echo = TRUE)

  invisible(NULL)
}
