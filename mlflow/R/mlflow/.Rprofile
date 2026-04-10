repos <- getOption("repos")
# https://packagemanager.rstudio.com provides pre-compiled package binaries for Linux
# that can be installed significantly faster than uncompiled package sources.
if (Sys.which("lsb_release") != "") {
    ubuntu_codename <- tolower(system("lsb_release -cs", intern = TRUE))
    repo_name <- sprintf("https://packagemanager.rstudio.com/cran/__linux__/%s/latest", ubuntu_codename)
    options(repos = c(repos, REPO_NAME = repo_name))
}
