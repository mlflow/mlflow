# Generate docs as markdown
Rd2md::ReferenceManual()

# Remove markdown package description
markdown_doc <- readLines("Reference_Manual_mlflow.md")
first_function <- which(grepl("active_experiment", markdown_doc))[[1]]
markdown_fixed <- markdown_doc[first_function:length(markdown_doc)]
writeLines(markdown_fixed, "Reference_Manual_mlflow.md")

# Clear Sphinx docs and tree to correctly generate sections
if (dir.exists("../../docs/build")) {
  unlink("../../docs/build", recursive = TRUE)
}

# Generate reStructuredText documentation
rmarkdown::pandoc_convert("Reference_Manual_mlflow.md", output = "../../docs/source/R-api.rst")

# Add R API header to RST docs
rst_header <- ".. _R-api:

========
R API
========
"
rst_doc <- readLines("../../docs/source/R-api.rst")
rst_doc <- c(rst_header, rst_doc)
writeLines(rst_doc, "../../docs/source/R-api.rst")

# Generate docs by using an mlflow virtualenv and running `make` from `mlflow/docs`


