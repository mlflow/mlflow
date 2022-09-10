FROM condaforge/miniforge3

RUN apt-get update -y && apt-get install build-essential -y
RUN echo foo
RUN conda --version
# Without `charset-normalizer=2.0.12`, `conda install` below would fail with:
# CondaHTTPError: HTTP 404 NOT FOUND for url <https://conda.anaconda.org/conda-forge/noarch/charset-normalizer-2.0.11-pyhd8ed1ab_0.conda>
RUN conda install python=3.7 charset-normalizer=2.0.12
RUN pip install mlflow
