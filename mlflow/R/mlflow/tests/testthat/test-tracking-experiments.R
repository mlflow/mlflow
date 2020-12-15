context("Tracking - Experiments")

teardown({
  mlflow_clear_test_dir("mlruns")
})

test_that("mlflow_create/get_experiment() basic functionality (fluent)", {
  mlflow_clear_test_dir("mlruns")

  experiment_1_id <- mlflow_create_experiment("exp_name", "art_loc")
  experiment_1a <- mlflow_get_experiment(experiment_id = experiment_1_id)
  experiment_1b <- mlflow_get_experiment(name = "exp_name")

  expect_identical(experiment_1a, experiment_1b)
  expect_identical(experiment_1a$artifact_location, "art_loc")
  expect_identical(experiment_1a$name, "exp_name")
})

test_that("mlflow_create/get_experiment() basic functionality (client)", {
  mlflow_clear_test_dir("mlruns")

  client <- mlflow_client()

  experiment_1_id <- mlflow_create_experiment(client = client, "exp_name", "art_loc")
  experiment_1a <- mlflow_get_experiment(client = client, experiment_id = experiment_1_id)
  experiment_1b <- mlflow_get_experiment(client = client, name = "exp_name")

  expect_identical(experiment_1a, experiment_1b)
  expect_identical(experiment_1a$artifact_location, "art_loc")
  expect_identical(experiment_1a$name, "exp_name")
})

test_that("mlflow_get_experiment() not found error", {
  mlflow_clear_test_dir("mlruns")

  expect_error(
    mlflow_get_experiment(experiment_id = "42"),
    "Could not find experiment with ID 42"
  )
})

test_that("mlflow_list_experiments() works properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  mlflow_create_experiment(client = client, "foo1", "art_loc1")
  mlflow_create_experiment(client = client, "foo2", "art_loc2")

  # client
  experiments_list <- mlflow_list_experiments(client = client)
  expect_setequal(experiments_list$experiment_id, c("0", "1", "2"))
  expect_setequal(experiments_list$name, c("Default", "foo1", "foo2"))
  default_artifact_loc <- file.path(getwd(), "mlruns", "0", fsep = "/")
  expect_setequal(experiments_list$artifact_location, c(default_artifact_loc,
                                                        "art_loc1",
                                                        "art_loc2"))

  # fluent
  experiments_list <- mlflow_list_experiments()
  expect_setequal(experiments_list$experiment_id, c("0", "1", "2"))
  expect_setequal(experiments_list$name, c("Default", "foo1", "foo2"))
  default_artifact_loc <- file.path(getwd(), "mlruns", "0", fsep = "/")
  expect_setequal(experiments_list$artifact_location, c(default_artifact_loc,
                                                        "art_loc1",
                                                        "art_loc2"))

  # Returns NULL when no experiments found
  expect_null(mlflow_list_experiments("DELETED_ONLY"))

  # `view_type` is respected
  mlflow_delete_experiment(experiment_id = "1")
  deleted_experiments <- mlflow_list_experiments("DELETED_ONLY")
  expect_identical(deleted_experiments$name, "foo1")
})

test_that("mlflow_set_experiment_tag() works correctly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  experiment_id <- mlflow_create_experiment(client = client, "setExperimentTagTestExperiment", "art_exptag_loc")
  mlflow_set_experiment_tag("dataset", "imagenet1K", experiment_id, client = client)
  experiment <- mlflow_get_experiment(experiment_id = experiment_id, client = client)
  tags <- experiment$tags[[1]]
  expect_true("dataset" %in% tags$key)
  expect_identical("imagenet1K", tags$value[tags$key == "dataset"])

  # test that updating a tag works
  mlflow_set_experiment_tag("dataset", "birdbike", experiment_id, client = client)
  experiment <- mlflow_get_experiment(experiment_id = experiment_id, client = client)
  tags <- experiment$tags[[1]]
  expect_true("dataset" %in% tags$key)
  expect_identical("birdbike", tags$value[tags$key == "dataset"])

  # test that setting a tag on 1 experiment does not impact another experiment.
  experiment_id_2 <- mlflow_create_experiment(client = client, "setExperimentTagTestExperiment2", "art_exptag_loc2")
  experiment_2 <- mlflow_get_experiment(experiment_id = experiment_id_2, client = client)
  expect_false("tags" %in% colnames(experiment_2))

  # test that setting a tag on different experiments maintain different values across experiments
  mlflow_set_experiment_tag("dataset", "birds200", experiment_id_2, client = client)
  experiment <- mlflow_get_experiment(experiment_id = experiment_id, client = client)
  tags <- experiment$tags[[1]]
  experiment_2 <- mlflow_get_experiment(experiment_id = experiment_id_2, client = client)
  tags_2 <- experiment_2$tags[[1]]
  expect_true("dataset" %in% tags$key)
  expect_identical("birdbike", tags$value[tags$key == "dataset"])
  expect_true("dataset" %in% tags_2$key)
  expect_identical("birds200", tags_2$value[tags_2$key == "dataset"])

  # test can set multi-line tags
  mlflow_set_experiment_tag("multiline tag", "value2\nvalue2\nvalue2", experiment_id, client = client)
  experiment <- mlflow_get_experiment(experiment_id = experiment_id, client = client)
  for (subtag in experiment$tags) {
    if (subtag$key == "multiline tag") {
      expect_identical("value2\nvalue2\nvalue2", subtag$value)
    }
  }
})


test_that("mlflow_get_experiment_by_name() works properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  expect_error(
    mlflow_get_experiment(client = client, name = "exp"),
    "Could not find experiment with name 'exp'"
  )
  experiment_id <- mlflow_create_experiment(client = client, "exp", "art")
  experiment <- mlflow_get_experiment(client = client, name = "exp")
  expect_identical(experiment_id, experiment$experiment_id)
  expect_identical(experiment$name, "exp")
  expect_identical(experiment$artifact_location, "art")
})

test_that("infer experiment id works properly", {
  mlflow_clear_test_dir("mlruns")
  experiment_id <- mlflow_create_experiment("test")
  Sys.setenv(MLFLOW_EXPERIMENT_NAME = "test")
  expect_true(experiment_id == mlflow_infer_experiment_id())
  Sys.unsetenv("MLFLOW_EXPERIMENT_NAME")
  Sys.setenv(MLFLOW_EXPERIMENT_ID = experiment_id)
  expect_true(experiment_id == mlflow_infer_experiment_id())
  Sys.unsetenv("MLFLOW_EXPERIMENT_ID")
  mlflow_set_experiment("test")
  expect_true(experiment_id == mlflow_infer_experiment_id())
})

test_that("experiment setting works", {
  mlflow_clear_test_dir("mlruns")
  exp1_id <- mlflow_create_experiment("exp1")
  exp2_id <- mlflow_create_experiment("exp2")
  mlflow_set_experiment(experiment_name = "exp1")
  expect_identical(exp1_id, mlflow_get_active_experiment_id())
  expect_identical(mlflow_get_experiment(exp1_id), mlflow_get_experiment())
  mlflow_set_experiment(experiment_id = exp2_id)
  expect_identical(exp2_id, mlflow_get_active_experiment_id())
  expect_identical(mlflow_get_experiment(exp2_id), mlflow_get_experiment())
})

test_that("mlflow_set_experiment() creates experiments", {
  mlflow_clear_test_dir("mlruns")
  mlflow_set_experiment(experiment_name = "foo", artifact_location = "artifact/location")
  experiment <- mlflow_get_experiment()
  expect_identical(experiment$artifact_location, "artifact/location")
  expect_identical(experiment$name, "foo")
})
