# Make sure the current working directory is 'mlflow/R/mlflow'
print(R.version)
repos = getOption("repos")
repos["CRAN"] = "http://cran.us.r-project.org"
options(repos = repos)
install.packages('remotes')
saveRDS(remotes::dev_package_deps(".", dependencies = TRUE), "depends.Rds", version = 2)
writeLines(sprintf("R-%i.%i", getRversion()$major, getRversion()$minor), "R-version")
