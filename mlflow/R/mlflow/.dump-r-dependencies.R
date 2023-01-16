# Make sure the current working directory is 'mlflow/R/mlflow'
print(R.version)
install.packages('remotes')
saveRDS(remotes::dev_package_deps(".", dependencies = TRUE), "depends.Rds", version = 2)
writeLines(sprintf("R-%i.%i", getRversion()$major, getRversion()$minor), "R-version")
