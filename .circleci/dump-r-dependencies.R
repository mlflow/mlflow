print(R.version)
install.packages('remotes')
saveRDS(remotes::dev_package_deps("mlflow/R/mlflow", dependencies = TRUE), ".circleci/depends.Rds", version = 2)
writeLines(sprintf("R-%i.%i", getRversion()$major, getRversion()$minor), ".circleci/R-version")
