### Train and Publish Locally With RAPIDS and MLFlow
**[RAPIDS](https://rapids.ai/)** is a suite of open source libraries for GPU-accelerated analytics. 

**[RAPIDS cuML](https://github.com/rapidsai/cuml)** matches the scikit-learn API, so it can build on MLFlow's existing support for scikit-learn-like models to support 
persistence and deployment."

#### Jupyter Notebook Workflow
[Jupyter Notebook](notebooks/rapids_mlflow.ipynb)

#### CLI Based Workflow
1. Create a new conda environment.
    1. `$ conda create -f envs/conda.yaml`
1. Train the model
    1. `$ cd mlflow_project`
    1. The example project is described in [MLProject](https://www.mlflow.org/docs/latest/projects.html) file.
        1. This can be edited to allow additional command line variables, specify conda environments, and training
        parameters (see link for additional information).
    1. Publish to local tracking server
        1. Here we instruct mlflow to run our training routine locally, and publish the results to the local file system.
        1. In your shell, run:
```shell script
# Download the file
wget -N https://rapidsai-cloud-ml-sample-data.s3-us-west-2.amazonaws.com/airline_small.parquet
# Launch the job
mlflow run . -b local -e hyperopt \
         -P conda-env=$PWD/envs/conda.yaml \
         -P fpath=airline_small.parquet
```

1. Deploy your model
    1. Locate the model's 'Full Path'
        1. In your shell, run `mlflow ui`
        1. Locate the model path using the mlflow ui at localhost:5000
    1. Select the successful run and find the 'Full Path' element
    1. Deploy your model
        1. `$ mlflow models serve --no-conda -m [PATH_TO_MODEL] -p 55755`

1. Query the deployed model with test data `src/sample_server_query.sh` example script.
    1. `bash src/sample_server_query.sh`
