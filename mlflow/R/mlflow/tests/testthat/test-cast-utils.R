context("Cast utils")

test_that("cast_string works correctly", {
  expect_equal(cast_string("test"), "test")
  expect_equal(cast_string(123), "123")
  expect_equal(cast_string(TRUE), "TRUE")
  
  expect_equal(cast_string(NA, allow_na = TRUE), NA_character_)
  expect_error(cast_string(NA, allow_na = FALSE))
  expect_error(cast_string(NULL, allow_na = FALSE))
})

test_that("cast_nullable_string works correctly", {
  expect_equal(cast_nullable_string("test"), "test")
  expect_equal(cast_nullable_string(123), "123")
  expect_null(cast_nullable_string(NULL))
})

test_that("cast_scalar_double works correctly", {
  expect_equal(cast_scalar_double(123), 123)
  expect_equal(cast_scalar_double("123.5"), 123.5)
  
  expect_equal(cast_scalar_double(NA, allow_na = TRUE), NA_real_)
  expect_error(cast_scalar_double(NA, allow_na = FALSE))
  expect_error(cast_scalar_double(NULL, allow_na = FALSE))
  expect_error(cast_scalar_double(c(1, 2)))
})

test_that("cast_nullable_scalar_double works correctly", {
  expect_equal(cast_nullable_scalar_double(123), 123)
  expect_equal(cast_nullable_scalar_double("123.5"), 123.5)
  expect_null(cast_nullable_scalar_double(NULL))
  expect_error(cast_nullable_scalar_double(c(1, 2)))
})

test_that("cast_nullable_scalar_integer works correctly", {
  expect_equal(cast_nullable_scalar_integer(123), 123L)
  expect_equal(cast_nullable_scalar_integer("123"), 123L)
  expect_null(cast_nullable_scalar_integer(NULL))
  expect_error(cast_nullable_scalar_integer(c(1, 2)))
})

test_that("cast_string_list works correctly", {
  expect_equal(cast_string_list(c("a", "b")), list("a", "b"))
  expect_equal(cast_string_list(list("a", "b")), list("a", "b"))
  expect_equal(cast_string_list(c(1, 2)), list("1", "2"))
  expect_null(cast_string_list(NULL))
})

test_that("cast_choice works correctly", {
  expect_equal(cast_choice("a", c("a", "b")), "a")
  expect_equal(cast_choice(1, c("1", "2")), "1")
  
  expect_null(cast_choice(NULL, c("a", "b"), allow_null = TRUE))
  expect_error(cast_choice(NULL, c("a", "b"), allow_null = FALSE))
  expect_error(cast_choice("c", c("a", "b")))
})