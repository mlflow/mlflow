test_that("mlflow_log_artifact and mlflow_list_artifacts work", {
  with(mlflow_start_run(), {
    # List artifacts for run without artifacts, result should be empty
    empty_artifact_list <- mlflow_list_artifacts()
    # Create some dummy artifact files/directories
    expect_equal(nrow(empty_artifact_list), 0)
    source_dir <- file.path(tempdir(), "temp-directory")
    dir.create(source_dir)
    ## file 1
    file_path1 <- file.path(source_dir, "a-my-file")
    contents <- "File contents\n"
    cat(contents, file = file_path1, sep = "")
    ## file 2
    file_path2 <- file.path(source_dir, "a-my-file-2")
    contents <- "File contents\n"
    cat(contents, file = file_path2, sep = "")

    # Log file, file with path, directory with path argument
    mlflow_log_artifact(file_path1)
    mlflow_log_artifact(file_path2)
    mlflow_log_artifact(file_path1, "directory_for_file")
    mlflow_log_artifact(source_dir, "artifact_subdirectory")
    # Verify logged files
    artifact_list0 <- mlflow_list_artifacts()
    expect_equal(nrow(artifact_list0), 4)
    logged_file0 <- artifact_list0[artifact_list0$path == "a-my-file", ]
    expect_equal(nrow(logged_file0), 1)
    expect_equal(logged_file0$is_dir, FALSE)
    logged_file1 <- artifact_list0[artifact_list0$path == "directory_for_file", ]
    expect_equal(nrow(logged_file1), 1)
    expect_equal(logged_file1$is_dir, TRUE)
    logged_dir0 <- artifact_list0[artifact_list0$path == "artifact_subdirectory", ]
    expect_equal(nrow(logged_dir0), 1)
    expect_equal(logged_dir0$is_dir, TRUE)
    # Verify contents of logged directory
    artifact_list1 <- mlflow_list_artifacts("artifact_subdirectory")
    expect_equal(nrow(artifact_list1), 2)
    logged_file2 <- artifact_list1[artifact_list1$path ==
      paste("artifact_subdirectory", "a-my-file", sep = "/"), ]
    expect_equal(nrow(logged_file2), 1)
    expect_equal(logged_file2$is_dir, FALSE)
    expect_equal(strtoi(logged_file2$file_size), nchar(contents))
    # Verify contents of file logged under directory
    artifact_list2 <- mlflow_list_artifacts("directory_for_file")
    expect_equal(nrow(artifact_list2), 1)
    logged_file3 <- artifact_list2[artifact_list2$path ==
    paste("directory_for_file", "a-my-file", sep = "/"), ]
    expect_equal(nrow(logged_file3), 1)
    expect_equal(logged_file3$is_dir, FALSE)
    expect_equal(strtoi(logged_file3$file_size), nchar(contents))
  })
})