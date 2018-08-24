# `active_experiment`: Active Experiment

## Description


 Retrieve or set the active experiment.


## Usage

```r
mlflow_active_experiment()
mlflow_set_active_experiment(experiment_id)
```


## Arguments

Argument      |Description
------------- |----------------
```experiment_id```     |     Identifer to get an experiment.

# `active_run`: Active Run

## Description


 Retrieves or sets the active run.


## Usage

```r
mlflow_active_run()
mlflow_set_active_run(run)
```


## Arguments

Argument      |Description
------------- |----------------
```run```     |     The run object to make active.

# `mlflow_cli`: MLflow Command

## Description


 Executes a generic MLflow command through the commmand line interface.


## Usage

```r
mlflow_cli(..., background = FALSE, echo = TRUE)
```


## Arguments

Argument      |Description
------------- |----------------
```...```     |     The parameters to pass to the command line.
```background```     |     Should this command be triggered as a background task? Defaults to `FALSE` .
```echo```     |     Print the standard output and error to the screen? Defaults to `TRUE` , does not apply to background tasks.

## Value


 A `processx` task.


## Examples

```r 
 list("\n", "library(mlflow)\n", "mlflow_install()\n", "\n", "mlflow_cli(\"server\", \"--help\")\n") 
 
 ``` 

# `mlflow_connect`: Connect to MLflow

## Description


 Connect to local or remote MLflow instance.


## Usage

```r
mlflow_connect(x = NULL, activate = TRUE, ...)
```


## Arguments

Argument      |Description
------------- |----------------
```x```     |     (Optional) Either a URL to the remote MLflow server or the file store, i.e. the root of the backing file store for experiment and run data. If not specified, will launch and connect to a local instance listening on a random port.
```activate```     |     Whether to set the connction as the active connection, defaults to `TRUE`.
```...```     |     Optional arguments passed to `mlflow_server()`.

# `mlflow_create_experiment`: Create Experiment

## Description


 Creates an MLflow experiment.


## Usage

```r
mlflow_create_experiment(name, activate = TRUE)
```


## Arguments

Argument      |Description
------------- |----------------
```name```     |     The name of the experiment to create.
```activate```     |     Whether to set the created experiment as the active experiment. Defaults to `TRUE`.

## Examples

```r 
 list("\n", "library(mlflow)\n", "mlflow_install()\n", "\n", "# create local experiment\n", "mlflow_create_experiment(\"My Experiment\")\n", "\n", "# create experiment in remote MLflow server\n", "mlflow_set_tracking_uri(\"http://tracking-server:5000\")\n", "mlflow_create_experiment(\"My Experiment\")\n") 
 
 ``` 

# `mlflow_create_run`: Create Run

## Description


 reate a new run within an experiment. A run is usually a single execution of a machine learning or data ETL pipeline.


## Usage

```r
mlflow_create_run(user_id = NULL, run_name = NULL,
  source_type = NULL, source_name = NULL, status = NULL,
  start_time = NULL, end_time = NULL, source_version = NULL,
  artifact_uri = NULL, entry_point_name = NULL, run_tags = NULL,
  experiment_id = NULL)
```


## Arguments

Argument      |Description
------------- |----------------
```user_id```     |     User ID or LDAP for the user executing the run.
```run_name```     |     Human readable name for run.
```source_type```     |     Originating source for this run. One of Notebook, Job, Project, Local or Unknown.
```source_name```     |     String descriptor for source. For example, name or description of the notebook, or job name.
```status```     |     Current status of the run. One of RUNNING, SCHEDULE, FINISHED, FAILED, KILLED.
```start_time```     |     Unix timestamp of when the run started in milliseconds.
```end_time```     |     Unix timestamp of when the run ended in milliseconds.
```source_version```     |     Git version of the source code used to create run.
```artifact_uri```     |     URI of the directory where artifacts should be uploaded This can be a local path (starting with “/”), or a distributed file system (DFS) path, like s3://bucket/directory or dbfs:/my/directory. If not set, the local ./mlruns directory will be chosen by default.
```entry_point_name```     |     Name of the entry point for the run.
```run_tags```     |     Additional metadata for run in key-value pairs.
```experiment_id```     |     Unique identifier for the associated experiment.

## Details


 MLflow uses runs to track Param, Metric, and RunTag, associated with a single execution.


# `mlflow_disconnect`: Disconnect from MLflow

## Description


 Disconnects from a local MLflow instance.


## Usage

```r
mlflow_disconnect(mc)
```


## Arguments

Argument      |Description
------------- |----------------
```mc```     |     The MLflow connection created using `mlflow_connect()` .

# `mlflow_end_run`: End Run

## Description


 End the active run.


## Usage

```r
mlflow_end_run(status = "FINISHED")
```


## Arguments

Argument      |Description
------------- |----------------
```status```     |     Ending status of the run, defaults to `FINISHED`.

# `mlflow_get_experiment`: Get Experiment

## Description


 Get meta data for experiment and a list of runs for this experiment.


## Usage

```r
mlflow_get_experiment(experiment_id)
```


## Arguments

Argument      |Description
------------- |----------------
```experiment_id```     |     Identifer to get an experiment.

# `mlflow_get_metric_history`: Get Metric History

## Description


 For cases that a metric is logged more than once during a run, this API can be used
 to retrieve all logged values for this metric.


## Usage

```r
mlflow_get_metric_history(metric_key, run_uuid = NULL)
```


## Arguments

Argument      |Description
------------- |----------------
```metric_key```     |     Name of the metric.
```run_uuid```     |     Unique ID for the run for which metric is recorded.

# `mlflow_get_metric`: Get Metric

## Description


 API to retrieve the logged value for a metric during a run. For a run, if this
 metric is logged more than once, this API will retrieve only the latest value logged.


## Usage

```r
mlflow_get_metric(metric_key, run_uuid = NULL)
```


## Arguments

Argument      |Description
------------- |----------------
```metric_key```     |     Name of the metric.
```run_uuid```     |     Unique ID for the run for which metric is recorded.

# `mlflow_get_run`: Get Run

## Description


 Get meta data, params, tags, and metrics for run. Only last logged value for each metric is returned.


## Usage

```r
mlflow_get_run(run_uuid)
```


## Arguments

Argument      |Description
------------- |----------------
```run_uuid```     |     Unique ID for the run.

# `mlflow_install`: Install MLflow

## Description


 Installs MLflow for individual use.


## Usage

```r
mlflow_install()
```


## Details


 Notice that MLflow requires Python and Conda to be installed,
 see [https://www.python.org/getit/](https://www.python.org/getit/) and [https://conda.io/docs/installation.html](https://conda.io/docs/installation.html) .


## Examples

```r 
 list("\n", "library(mlflow)\n", "mlflow_install()\n") 
 
 ``` 

# `mlflow_list_experiments`: List Experiments

## Description


 Retrieves MLflow experiments as a data frame.


## Usage

```r
mlflow_list_experiments()
```


## Examples

```r 
 list("\n", "library(mlflow)\n", "mlflow_install()\n", "\n", "# list local experiments\n", "mlflow_list_experiments()\n", "\n", "# list experiments in remote MLflow server\n", "mlflow_set_tracking_uri(\"http://tracking-server:5000\")\n", "mlflow_list_experiments()\n") 
 
 ``` 

# `mlflow_log_artifact`: Log Artifact

## Description


 Logs an specific file or directory as an artifact.


## Usage

```r
mlflow_log_artifact(path, artifact_path = NULL)
```


## Arguments

Argument      |Description
------------- |----------------
```path```     |     The file or directory to log as an artifact.
```artifact_path```     |     Destination path within the run’s artifact URI.

## Details


 When logging to Amazon S3, ensure that the user has a proper policy
 attach to it, for instance:
 
 `` 
 
 Additionally, at least the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` 
 environment variables must be set to the corresponding key and secrets provided
 by Amazon IAM.


# `mlflow_log_metric`: Log Metric

## Description


 API to log a metric for a run. Metrics key-value pair that record a single float measure.
 During a single execution of a run, a particular metric can be logged several times.
 Backend will keep track of historical values along with timestamps.


## Usage

```r
mlflow_log_metric(key, value, timestamp = NULL, run_uuid = NULL)
```


## Arguments

Argument      |Description
------------- |----------------
```key```     |     Name of the metric.
```value```     |     Float value for the metric being logged.
```timestamp```     |     Unix timestamp in milliseconds at the time metric was logged.
```run_uuid```     |     Unique ID for the run.

# `mlflow_log_model`: Log Model

## Description


 Logs a model in the given run. Similar to `mlflow_save_model()`
 but stores model only as an artifact within the active run.


## Usage

```r
mlflow_log_model(f, path = "model")
```


## Arguments

Argument      |Description
------------- |----------------
```f```     |     The serving function that will perform a prediction.
```path```     |     Destination path where this MLflow compatible model will be saved.

# `mlflow_log_param`: Log Parameter

## Description


 API to log a parameter used for this run. Examples are params and hyperparams
 used for ML training, or constant dates and values used in an ETL pipeline.
 A params is a STRING key-value pair. For a run, a single parameter is allowed
 to be logged only once.


## Usage

```r
mlflow_log_param(key, value, run_uuid = NULL)
```


## Arguments

Argument      |Description
------------- |----------------
```key```     |     Name of the parameter.
```value```     |     String value of the parameter.
```run_uuid```     |     Unique ID for the run for which parameter is recorded.

# `mlflow_param`: Read Command Line Parameter

## Description


 Reads a command line parameter.


## Usage

```r
mlflow_param(name, default = NULL, type = NULL, description = NULL)
```


## Arguments

Argument      |Description
------------- |----------------
```name```     |     The name for this parameter.
```default```     |     The default value for this parameter.
```type```     |     Type of this parameter. Required if `default` is not set. If specified, must be one of "numeric", "integer", or "string".
```description```     |     Optional description for this parameter.

# `mlflow_restore_snapshot`: Restore Snapshot

## Description


 Restores a snapshot of all dependencies required to run the files in the
 current directory


## Usage

```r
mlflow_restore_snapshot()
```


# `mlflow_rfunc_predict`: Predict using RFunc MLflow Model

## Description


 Predict using an RFunc MLflow Model from a file or data frame.


## Usage

```r
mlflow_rfunc_predict(model_dir, data, output_file = NULL,
  restore = FALSE)
```


## Arguments

Argument      |Description
------------- |----------------
```model_dir```     |     The path to the MLflow model, as a string.
```data```     |     Data frame, 'JSON' or 'CSV' file to be used for prediction.
```output_file```     |     'JSON' or 'CSV' file where the prediction will be written to.
```restore```     |     Should `mlflow_restore_snapshot()` be called before serving?

## Examples

```r 
 list("\n", "library(mlflow)\n", "\n", "# save simple model which roundtrips data as prediction\n", "mlflow_save_model(function(df) df, \"mlflow_roundtrip\")\n", "\n", "# save data as json\n", "jsonlite::write_json(iris, \"iris.json\")\n", "\n", "# predict existing model from json data\n", "mlflow_rfunc_predict(\"mlflow_roundtrip\", \"iris.json\")\n") 
 
 ``` 

# `mlflow_rfunc_serve`: Serve an RFunc MLflow Model

## Description


 Serve an RFunc MLflow Model as a local web api under [http://localhost:8090](http://localhost:8090) .


## Usage

```r
mlflow_rfunc_serve(model_dir, host = "127.0.0.1", port = 8090,
  daemonized = FALSE, browse = !daemonized, restore = FALSE)
```


## Arguments

Argument      |Description
------------- |----------------
```model_dir```     |     The path to the MLflow model, as a string.
```host```     |     Address to use to serve model, as a string.
```port```     |     Port to use to serve model, as numeric.
```daemonized```     |     Makes 'httpuv' server daemonized so R interactive sessions are not blocked to handle requests. To terminate a daemonized server, call 'httpuv::stopDaemonizedServer()' with the handle returned from this call.
```browse```     |     Launch browser with serving landing page?
```restore```     |     Should `mlflow_restore_snapshot()` be called before serving?

## Examples

```r 
 list("\n", "library(mlflow)\n", "\n", "# save simple model with constant prediction\n", "mlflow_save_model(function(df) 1, \"mlflow_constant\")\n", "\n", "# serve an existing model over a web interface\n", "mlflow_rfunc_serve(\"mlflow_constant\")\n", "\n", "# request prediction from server\n", "httr::POST(\"http://127.0.0.1:8090/predict/\")\n") 
 ``` 

# `mlflow_run`: Run in MLflow

## Description


 Wrapper for `mlflow run`.


## Usage

```r
mlflow_run(uri, entry_point = NULL, version = NULL,
  param_list = NULL, experiment_id = NULL, mode = NULL,
  cluster_spec = NULL, git_username = NULL, git_password = NULL,
  no_conda = FALSE, storage_dir = NULL)
```


## Arguments

Argument      |Description
------------- |----------------
```uri```     |     A directory or an R script.
```entry_point```     |     Entry point within project, defaults to `main` if not specified.
```version```     |     Version of the project to run, as a Git commit reference for Git projects.
```param_list```     |     A list of parameters.
```experiment_id```     |     ID of the experiment under which to launch the run.
```mode```     |     Execution mode to use for run.
```cluster_spec```     |     Path to JSON file describing the cluster to use when launching a run on Databricks.
```git_username```     |     Username for HTTP(S) Git authentication.
```git_password```     |     Password for HTTP(S) Git authentication.
```no_conda```     |     If specified, assume that MLflow is running within a Conda environment with the necessary dependencies for the current project instead of attempting to create a new conda environment. Only valid if running locally.
```storage_dir```     |     Only valid when `mode` is local. MLflow downloads artifacts from distributed URIs passed to parameters of type 'path' to subdirectories of storage_dir.

# `mlflow_save_model`: Save Model for MLflow

## Description


 Saves model in MLflow's format that can later be used
 for prediction and serving.


## Usage

```r
mlflow_save_model(f, path = "model")
```


## Arguments

Argument      |Description
------------- |----------------
```f```     |     The serving function that will perform a prediction.
```path```     |     Destination path where this MLflow compatible model will be saved.

# `mlflow_server`: Run the MLflow Tracking Server

## Description


 Wrapper for `mlflow server`.


## Usage

```r
mlflow_server(file_store = "mlruns", default_artifact_root = NULL,
  host = "127.0.0.1", port = 5000, workers = 4,
  static_prefix = NULL)
```


## Arguments

Argument      |Description
------------- |----------------
```file_store```     |     The root of the backing file store for experiment and run data.
```default_artifact_root```     |     Local or S3 URI to store artifacts in, for newly created experiments.
```host```     |     The network address to listen on (default: 127.0.0.1).
```port```     |     The port to listen on (default: 5000).
```workers```     |     Number of gunicorn worker processes to handle requests (default: 4).
```static_prefix```     |     A prefix which will be prepended to the path of all static paths.

# `mlflow_set_tracking_uri`: Set Remote Tracking URI

## Description


 Specifies the URI to the remote MLflow server that will be used
 to track experiments.


## Usage

```r
mlflow_set_tracking_uri(uri)
```


## Arguments

Argument      |Description
------------- |----------------
```uri```     |     The URI to the remote MLflow server.

# `mlflow_snapshot`: Dependencies Snapshot

## Description


 Creates a snapshot of all dependencies required to run the files in the
 current directory.


## Usage

```r
mlflow_snapshot()
```


# `mlflow_source`: Source a Script with MLflow Params

## Description


 This function should not be used interactively. It is designed to be called via `Rscript` from
 the terminal or through the MLflow CLI.


## Usage

```r
mlflow_source(uri)
```


## Arguments

Argument      |Description
------------- |----------------
```uri```     |     Path to an R script, can be a quoted or unquoted string.

# `mlflow_start_run`: Start Run

## Description


 Starts a new run within an experiment, should be used within a `with` block.


## Usage

```r
mlflow_start_run(run_uuid = NULL, experiment_id = NULL,
  source_name = NULL, source_version = NULL, entry_point_name = NULL,
  source_type = "LOCAL")
```


## Arguments

Argument      |Description
------------- |----------------
```run_uuid```     |     If specified, get the run with the specified UUID and log metrics and params under that run. The run's end time is unset and its status is set to running, but the run's other attributes remain unchanged.
```experiment_id```     |     Used only when ``run_uuid`` is unspecified. ID of the experiment under which to create the current run. If unspecified, the run is created under a new experiment with a randomly generated name.
```source_name```     |     Name of the source file or URI of the project to be associated with the run. Defaults to the current file if none provided.
```source_version```     |     Optional Git commit hash to associate with the run.
```entry_point_name```     |     Optional name of the entry point for to the current run.
```source_type```     |     Integer enum value describing the type of the run  ("local", "project", etc.).

## Examples

```r 
 list("\n", "with(mlflow_start_run(), {\n", "  mlflow_log(\"test\", 10)\n", "})\n") 
 
 ``` 

# `mlflow_tracking_uri`: Get Remote Tracking URI

## Description


 Get Remote Tracking URI


## Usage

```r
mlflow_tracking_uri()
```


# `mlflow_ui`: MLflow User Interface

## Description


 Launches MLflow user interface.


## Usage

```r
mlflow_ui(x, ...)
```


## Arguments

Argument      |Description
------------- |----------------
```x```     |     If specified, can be either an `mlflow_connection` object or a string specifying the file store, i.e. the root of the backing file store for experiment and run data.
```...```     |     Optional arguments passed to `mlflow_server()` when `x` is a path to a file store.

## Examples

```r 
 list("\n", "library(mlflow)\n", "mlflow_install()\n", "\n", "# launch mlflow ui locally\n", "mlflow_ui()\n", "\n", "# launch mlflow ui for existing mlflow server\n", "mlflow_set_tracking_uri(\"http://tracking-server:5000\")\n", "mlflow_ui()\n") 
 
 ``` 

# `mlflow_update_run`: Update Run

## Description


 Update Run


## Usage

```r
mlflow_update_run(status = c("FINISHED", "SCHEDULED", "FAILED",
  "KILLED"), end_time = NULL, run_uuid = NULL)
```


## Arguments

Argument      |Description
------------- |----------------
```status```     |     Updated status of the run. Defaults to `FINISHED`.
```end_time```     |     Unix timestamp of when the run ended in milliseconds.
```run_uuid```     |     Unique identifier for the run.

# `reexports`: Objects exported from other packages

## Description


 These objects are imported from other packages. Follow the links
 below to see their documentation.
 
 list("\n", "  ", list(list("purrr"), list(list(list("%>%")))), "\n", "\n", "  ", list(list("rlang"), list(list(list("%||%")))), "\n")

