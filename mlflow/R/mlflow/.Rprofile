# https://packagemanager.rstudio.com provides pre-compiled package binaries for Linux
# that can be installed significantly faster than uncompiled package sources.
if (Sys.which("lst_release") != "") {
    ubuntu_code_name <- tolower(gsub(" .*", "", system("lsb_release -cs", intern = TRUE)))
    repo_name <- sprintf("https://packagemanager.rstudio.com/cran/__linux__/%s/latest", ubuntu_code_name)
    options(repos = c(REPO_NAME = repo_name))
}
