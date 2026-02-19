context("Tracking - Experiments")

teardown({
  mlflow_clear_test_dir("mlruns")
})

test_that("mlflow_create/set/get_experiment() basic functionality (fluent)", {
  mlflow_clear_test_dir("mlruns")
  artifact_relative_path <- "art_loc"
  experiment_1_id <- mlflow_create_experiment("exp_name_1", artifact_relative_path)
  experiment_2_id <- mlflow_set_experiment(experiment_name = "exp_name_2", artifact_location = artifact_relative_path)
  experiment_1a <- mlflow_get_experiment(experiment_id = experiment_1_id)
  experiment_1b <- mlflow_get_experiment(name = "exp_name_1")
  experiment_2a <- mlflow_get_experiment(name = "exp_name_2")

  expect_identical(experiment_1a, experiment_1b)
  expected_artifact_location <- sprintf("%s/%s", getwd(), artifact_relative_path)
  expect_identical(experiment_1a$artifact_location, expected_artifact_location)
  expect_identical(experiment_1a$name, "exp_name_1")
  expect_identical(experiment_2a$name, "exp_name_2")
  expect_identical(experiment_2a$artifact_location, expected_artifact_location)
})

test_that("mlflow_create/get_experiment() basic functionality (client)", {
  mlflow_clear_test_dir("mlruns")

  client <- mlflow_client()
  artifact_relative_path <- "art_loc"
  experiment_1_id <- mlflow_create_experiment(
    client = client,
    name = "exp_name",
    artifact_location = artifact_relative_path,
    tags = list(foo = "bar", foz = "baz", fiz = "biz")
  )
  experiment_1a <- mlflow_get_experiment(client = client, experiment_id = experiment_1_id)
  experiment_1b <- mlflow_get_experiment(client = client, name = "exp_name")

  expect_identical(experiment_1a, experiment_1b)
  expected_artifact_location <- sprintf("%s/%s", getwd(), artifact_relative_path)
  expect_identical(experiment_1a$artifact_location, expected_artifact_location)
  expect_identical(experiment_1a$name, "exp_name")

  expect_true(
    all(purrr::transpose(experiment_1b$tags[[1]]) %in%
      list(
        list(key = "foz", value = "baz"),
        list(key = "foo", value = "bar"),
        list(key = "fiz", value = "biz")
      )
    )
  )
})

test_that("mlflow_get_experiment() not found error", {
  mlflow_clear_test_dir("mlruns")

  expect_error(
    mlflow_get_experiment(experiment_id = "42"),
    "Could not find experiment with ID 42"
  )
})

test_that("mlflow_search_experiments() works properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  ex1 <- mlflow_create_experiment(client = client, "foo1", "art_loc1")
  ex2 <- mlflow_create_experiment(client = client, "foo2", "art_loc2")
  ex3 <- mlflow_create_experiment(client = client, "foo3", "art_loc3")

  mlflow_set_experiment_tag("expgroup", "group1", experiment_id = ex1)
  mlflow_set_experiment_tag("expgroup", "group1", experiment_id = ex3)

  allexperiments_result <- mlflow_search_experiments(client = client)
  allexperiments <- allexperiments_result$experiments
  expect_setequal(allexperiments$experiment_id, c("0", ex1, ex2, ex3))
  expect_setequal(allexperiments$name, c("Default", "foo1", "foo2", "foo3"))
  default_artifact_loc <- file.path(getwd(), "mlruns", "0", fsep = "/")
  expect_setequal(allexperiments$artifact_location, c(default_artifact_loc,
                                                      sprintf("%s/%s", getwd(), "art_loc1"),
                                                      sprintf("%s/%s", getwd(), "art_loc2"),
                                                      sprintf("%s/%s", getwd(), "art_loc3")))
  expect_null(allexperiments_result$next_page_token)

  ex1_result = mlflow_search_experiments(filter = "attribute.name = 'foo1'")
  expect_setequal(ex1_result$experiments$experiment_id, c(ex1))
  expect_null(ex1_result$next_page_token)

  exgroup1_result = mlflow_search_experiments(filter = "tags.expgroup = 'group1'")
  expect_setequal(exgroup1_result$experiments$experiment_id, c(ex1, ex3))
  expect_null(exgroup1_result$next_page_token)

  mlflow_delete_experiment(experiment_id = ex1)
  deleted_experiments_result <- mlflow_search_experiments(experiment_view_type = "DELETED_ONLY")
  expect_setequal(deleted_experiments_result$experiments$experiment_id, c(ex1))
  expect_null(deleted_experiments_result$next_page_token)

  # By default, only active experiments should be returned
  active_experiments_result <- mlflow_search_experiments()
  expect_setequal(active_experiments_result$experiments$experiment_id, c("0", ex2, ex3))
  expect_null(active_experiments_result$next_page_token)

  order_limit_result1 <- mlflow_search_experiments(
    max_results = 2,
    order_by = c("attribute.name desc"),
    experiment_view_type="ALL",
  )
  expect_setequal(order_limit_result1$experiments$name, c("foo3", "foo2"))

  order_limit_result2 <- mlflow_search_experiments(
    max_results = 2,
    order_by = c("attribute.name desc"),
    page_token = order_limit_result1$next_page_token,
    experiment_view_type="ALL",
  )
  expect_setequal(order_limit_result2$experiments$name, c("foo1", "Default"))
  expect_null(order_limit_result2$next_page_token)
})

test_that("mlflow_search_experiments() works properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  art_loc_1 <- "art_loc1"
  art_loc_2 <- "art_loc2"
  art_loc_3 <- "art_loc3"
  ex1 <- mlflow_create_experiment(client = client, "foo1", art_loc_1)
  ex2 <- mlflow_create_experiment(client = client, "foo2", art_loc_2)
  ex3 <- mlflow_create_experiment(client = client, "foo3", art_loc_3)

  mlflow_set_experiment_tag("expgroup", "group1", experiment_id = ex1)
  mlflow_set_experiment_tag("expgroup", "group1", experiment_id = ex3)

  allexperiments_result <- mlflow_search_experiments(client = client)
  allexperiments <- allexperiments_result$experiments
  expect_setequal(allexperiments$experiment_id, c("0", ex1, ex2, ex3))
  expect_setequal(allexperiments$name, c("Default", "foo1", "foo2", "foo3"))
  default_artifact_loc <- file.path(getwd(), "mlruns", "0", fsep = "/")
  expect_setequal(allexperiments$artifact_location, c(default_artifact_loc,
                                                      sprintf("%s/%s", getwd(), art_loc_1),
                                                      sprintf("%s/%s", getwd(), art_loc_2),
                                                      sprintf("%s/%s", getwd(), art_loc_3)))
  expect_null(allexperiments_result$next_page_token)

  ex1_result = mlflow_search_experiments(filter = "attribute.name = 'foo1'")
  expect_setequal(ex1_result$experiments$experiment_id, c(ex1))
  expect_null(ex1_result$next_page_token)

  exgroup1_result = mlflow_search_experiments(filter = "tags.expgroup = 'group1'")
  expect_setequal(exgroup1_result$experiments$experiment_id, c(ex1, ex3))
  expect_null(exgroup1_result$next_page_token)

  mlflow_delete_experiment(experiment_id = ex1)
  deleted_experiments_result <- mlflow_search_experiments(experiment_view_type = "DELETED_ONLY")
  expect_setequal(deleted_experiments_result$experiments$experiment_id, c(ex1))
  expect_null(deleted_experiments_result$next_page_token)

  # By default, only active experiments should be returned
  active_experiments_result <- mlflow_search_experiments()
  expect_setequal(active_experiments_result$experiments$experiment_id, c("0", ex2, ex3))
  expect_null(active_experiments_result$next_page_token)

  order_limit_result1 <- mlflow_search_experiments(
    max_results = 2,
    order_by = c("attribute.name desc"),
    experiment_view_type="ALL",
  )
  expect_setequal(order_limit_result1$experiments$name, c("foo3", "foo2"))

  order_limit_result2 <- mlflow_search_experiments(
    max_results = 2,
    order_by = c("attribute.name desc"),
    page_token = order_limit_result1$next_page_token,
    experiment_view_type="ALL",
  )
  expect_setequal(order_limit_result2$experiments$name, c("foo1", "Default"))
  expect_null(order_limit_result2$next_page_token)
})

test_that("mlflow_set_experiment_tag() works correctly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  experiment_id <- mlflow_create_experiment(client = client, "setExperimentTagTestExperiment", "art_exptag_loc")
  mlflow_set_experiment_tag("dataset", "imagenet1K", experiment_id, client = client)
  experiment <- mlflow_get_experiment(experiment_id = experiment_id, client = client)
  tags <- experiment$tags[[1]]
  expect_identical(tags, tibble::tibble(key = 'dataset', value = 'imagenet1K'))
  expect_identical("imagenet1K", tags$value[tags$key == "dataset"])

  # test that updating a tag works
  mlflow_set_experiment_tag("dataset", "birdbike", experiment_id, client = client)
  experiment <- mlflow_get_experiment(experiment_id = experiment_id, client = client)
  expect_equal(experiment$tags, list(tibble::tibble(key = 'dataset', value = 'birdbike')))

  # test that setting a tag on 1 experiment does not impact another experiment.
  experiment_id_2 <- mlflow_create_experiment(client = client, "setExperimentTagTestExperiment2", "art_exptag_loc2")
  experiment_2 <- mlflow_get_experiment(experiment_id = experiment_id_2, client = client)
  expect_equal(experiment_2$tags, NA)

  # test that setting a tag on different experiments maintain different values across experiments
  mlflow_set_experiment_tag("dataset", "birds200", experiment_id_2, client = client)
  experiment <- mlflow_get_experiment(experiment_id = experiment_id, client = client)
  tags <- experiment$tags[[1]]
  experiment_2 <- mlflow_get_experiment(experiment_id = experiment_id_2, client = client)
  tags_2 <- experiment_2$tags[[1]]
  expect_equal(tags, tibble::tibble(key = 'dataset', value = 'birdbike'))
  expect_equal(tags_2, tibble::tibble(key = 'dataset', value = 'birds200'))

  # test can set multi-line tags
  mlflow_set_experiment_tag("multiline tag", "value2\nvalue2\nvalue2", experiment_id, client = client)
  experiment <- mlflow_get_experiment(experiment_id = experiment_id, client = client)
  expect_identical(
        tibble::tibble(
          key = c('dataset', 'multiline tag'),
          value= c("birdbike", "value2\nvalue2\nvalue2")
        ),
        experiment$tags[[1]][order(experiment$tags[[1]]$key),]
  )
})


test_that("mlflow_get_experiment_by_name() works properly", {
  mlflow_clear_test_dir("mlruns")
  client <- mlflow_client()
  expect_error(
    mlflow_get_experiment(client = client, name = "exp"),
    "Could not find experiment with name 'exp'"
  )
  artifact_relative_path <- "art"
  experiment_id <- mlflow_create_experiment(client = client, "exp", artifact_relative_path)
  experiment <- mlflow_get_experiment(client = client, name = "exp")
  expect_identical(experiment_id, experiment$experiment_id)
  expect_identical(experiment$name, "exp")
  expect_identical(experiment$artifact_location, sprintf("%s/%s", getwd(), artifact_relative_path))
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
  artifact_relative_path <- "artifact/location"
  mlflow_set_experiment(experiment_name = "foo", artifact_location = artifact_relative_path)
  experiment <- mlflow_get_experiment()
  expected_artifact_location <- sprintf("%s/%s", getwd(), artifact_relative_path)
  expect_identical(experiment$artifact_location, expected_artifact_location)
  expect_identical(experiment$name, "foo")
})
