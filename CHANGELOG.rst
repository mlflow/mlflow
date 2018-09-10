Changelog
=========

0.6.0 (2018-09-10)
------------------

MLflow 0.6.0 introduces several major features:

- A Java client API, available on Maven
- Support for saving and serving SparkML models as MLeap for low-latency serving
- Support for tagging runs with metadata, during and after the run completion
- Support for deleting (and restoring deleted) experiments

In addition to these features, there are a host of improvements and bugfixes to the REST API, Python API, tracking UI, and documentation. The `examples/ <https://github.com/mlflow/mlflow/tree/master/examples>`_ subdirectory has also been revamped to make it easier to jump in, and examples demonstrating multistep workflows and hyperparameter tuning have been added.

Breaking changes:

We fixed a few inconsistencies in the the ``mlflow.tracking`` API, as introduced in 0.5.0:

- ``MLflowService`` has been renamed ``MlflowClient`` (#461, @mparkhe)
- You get an ``MlflowClient`` by calling ``mlflow.tracking.MlflowClient()`` (previously, this was ``mlflow.tracking.get_service()``) (#461, @mparkhe)
- ``MlflowService.list_runs`` was changed to ``MlflowService.list_run_infos`` to reflect the information actually returned by the call. It now returns a ``RunInfo`` instead of a ``Run`` (#334, @aarondav)
- ``MlflowService.log_artifact`` and ``MlflowService.log_artifacts`` now take a ``run_id`` instead of ``artifact_uri``. This now matches ``list_artifacts`` and ``download_artifacts``  (#444, @aarondav)

Features:

- Java client API added with support for the MLflow Tracking API (analogous to ``mlflow.tracking``), allowing users to create and manage experiments, runs, and artifacts. The release includes a `usage example <https://github.com/mlflow/mlflow/blob/master/mlflow/java/client/src/main/java/org/mlflow/tracking/samples/QuickStartDriver.java>`_ and `Javadocs <https://mlflow.org/docs/latest/java_api/index.html>`_. The client is published to Maven under ``mlflow:mlflow`` (#380, #394, #398, #409, #410, #430, #452, @aarondav)
- SparkML models are now also saved in MLeap format (https://github.com/combust/mleap), when applicable. Model serving platforms can choose to serve using this format instead of the SparkML format to dramatically decrease prediction latency. SageMaker now does this by default (#324, #327, #331, #395, #428, #435, #438, @dbczumar)
- [API] Experiments can now be deleted and restored via REST API, Python Tracking API, and MLflow CLI (#340, #344, #367, @mparkhe)
- [API] Tags can now be set via a SetTag API, and they have been moved to ``RunData`` from ``RunInfo`` (#342, @aarondav)
- [API] Added ``list_artifacts`` and ``download_artifacts`` to ``MlflowService`` to interact with a run's artifactory (#350, @andrewmchen)
- [API] Added ``get_experiment_by_name`` to Python Tracking API, and equivalent to Java API (#373, @vfdev-5)
- [API/Python] Version is now exposed via ``mlflow.__version__``.
- [API/CLI] Added ``mlflow artifacts`` CLI to list, download, and upload to run artifact repositories (#391, @aarondav)
- [UI] Added icons to source names in MLflow Experiments UI (#381, @andrewmchen)
- [UI] Added support to view ``.log`` and ``.tsv`` files from MLflow artifacts UI (#393, @Shenggan; #433, @whiletruelearn)
- [UI] Run names can now be edited from within the MLflow UI (#382, @smurching)
- [Serving] Added ``--host`` option to ``mlflow serve`` to allow listening on non-local addressess (#401, @hamroune)
- [Serving/SageMaker] SageMaker serving takes an AWS region argument (#366, @dbczumar)
- [Python] Added environment variables to support providing HTTP auth (username, password, token) when talking to a remote MLflow tracking server (#402, @aarondav)
- [Python] Added support to override S3 endpoint for S3 artifactory (#451, @hamroune)
- MLflow nightly Python wheel and JAR snapshots are now available and linked from https://github.com/mlflow/mlflow (#352, @aarondav)

Bug fixes and documentation updates:

- [Python] ``mlflow run`` now logs default parameters, in addition to explicitly provided ones (#392, @mparkhe)
- [Python] ``log_artifact`` in FileStore now requires a relative path as the artifact path (#439, @mparkhe)
- [Python] Fixed string representation of Python entities, so they now display both their type and serialized fields (#371, @smurching)
- [UI] Entry point name is now shown in MLflow UI (#345, @aarondav)
- [Models] Keras model export now includes Tensorflow graph explicitly to ensure the model can always be loaded at deployment time (#440, @tomasatdatabricks)
- [Python] Fixed issue where FileStore ignored provided Run Name (#358, @adrian555)
- [Python] Fixed an issue where any ``mlflow run`` failing printed an extraneous exception (#365, @smurching)
- [Python] uuid dependency removed (#351, @antonpaquin)
- [Python] Fixed issues with remote execution on Databricks (#357, #361, @smurching; #383, #387, @aarondav)
- [Docs] Added `comprehensive example <https://github.com/mlflow/mlflow/tree/master/examples/multistep_workflow>`_ of doing a multistep workflow, chaining MLflow runs together and reusing results (#338, @aarondav)
- [Docs] Added `comprehensive example <https://github.com/mlflow/mlflow/tree/master/examples/hyperparam>`_ of doing hyperparameter tuning (#368, @tomasatdatabricks)
- [Docs] Added code examples to ``mlflow.keras`` API (#341, @dmatrix)
- [Docs] Significant improvements to Python API documentation (#454, @stbof)
- [Docs] Examples folder refactored to improve readability. The examples now reside in ``examples/`` instead of ``example/``, too (#399, @mparkhe)
- Small bug fixes and doc updates (#328, #363, @ToonKBC; #336, #411, @aarondav; #284, @smurching; #377, @mparkhe; #389, gioa; #408, @aadamson; #397, @vfdev-5; #420, @adrian555; #459, #463, @stbof)


0.5.2 (2018-08-24)
------------------

MLflow 0.5.2 is a patch release on top of 0.5.1 containing only bug fixes and no breaking changes or features.

Bug fixes:

- Fix a bug with ECR client creation that caused ``mlflow.sagemaker.deploy()`` to fail when searching for a deployment Docker image (#366, @dbczumar)


0.5.1 (2018-08-23)
------------------

MLflow 0.5.1 is a patch release on top of 0.5.0 containing only bug fixes and no breaking changes or features.

Bug fixes:

- Fix ``with mlflow.start_run() as run`` to actually set ``run`` to the created Run (previously, it was None) (#322, @tomasatdatabricks)
- Fixes to DBFS artifactory to throw an exception if logging an artifact fails (#309) and to mimic FileStore's behavior of logging subdirectories (#347, @andrewmchen)
- Fix for Python 3.7 support with tarfiles (#329, @tomasatdatabricks)
- Fix spark.load_model not to delete the DFS tempdir (#335, @aarondav)
- MLflow UI now appropriately shows entrypoint if it's not main (#345, @aarondav)
- Make Python API forward-compatible with newer server versions of protos (#348, @aarondav)
- Improved API docs (#305, #284, @smurching)


0.5.0 (2018-08-17)
------------------

MLflow 0.5.0 offers some major improvements, including Keras and PyTorch first-class support as models, SFTP support as an artifactory, a new scatterplot visualization to compare runs, and a more complete Python SDK for experiment and run management.

Breaking changes:

- The Tracking API has been split into two pieces, a "basic logging" API and a "tracking service" API. The "basic logging" API deals with logging metrics, parameters, and artifacts to the currently-active active run, and is accessible in ``mlflow`` (e.g., ``mlflow.log_param``). The tracking service API allow managing experiments and runs (especially historical runs) and is available in ``mlflow.tracking``. The tracking service API will look analogous to the upcoming R and Java Tracking Service SDKs. Please be aware of the following breaking changes:

  - ``mlflow.tracking`` no longer exposes the basic logging API, only ``mlflow``. So, code that was written like ``from mlflow.tracking import log_param`` will have to be ``from mlflow import log_param`` (note that almost all examples were already doing this).
  - Access to the service API goes through the ``mlflow.tracking.get_service()`` function, which relies on the same tracking server set by either the environment variable ``MLFLOW_TRACKING_URI`` or by code with ``mlflow.tracking.set_tracking_uri()``. So code that used to look like ``mlflow.tracking.get_run()`` will now have to do ``mlflow.tracking.get_service().get_run()``. This does not apply to the basic logging API.
  - ``mlflow.ActiveRun`` has been converted into a lightweight wrapper around ``mlflow.entities.Run`` to enable the Python ``with`` syntax. This means that there are no longer any special methods on the object returned when calling ``mlflow.start_run()``. These can be converted to the service API.

  - The Python entities returned by the tracking service API are now accessible in ``mlflow.entities`` directly. Where previously you may have used ``mlflow.entities.experiment.Experiment``, you would now just use ``mlflow.entities.Experiment``. The previous version still exists, but is deprecated and may be hidden in a future version.
- REST API endpoint `/ajax-api/2.0/preview/mlflow/artifacts/get` has been moved to `$static_prefix/get-artifact`. This change is coversioned in the JavaScript, so should not be noticeable unless you were calling the REST API directly (#293, @andremchen)

Features:

- [Models] Keras integration: we now support logging Keras models directly in the log_model API, model format, and serving APIs (#280, @ToonKBC)
- [Models] PyTorch integration: we now support logging PyTorch models directly in the log_model API, model format, and serving APIs (#264, @vfdev-5)
- [UI] Scatterplot added to "Compare Runs" view to help compare runs using any two metrics as the axes (#268, @ToonKBC)
- [Artifacts] SFTP artifactory store added (#260, @ToonKBC)
- [Sagemaker] Users can specify a custom VPC when deploying SageMaker models (#304, @dbczumar)
- Pyfunc serialization now includes the Python version, and warns if the major version differs (can be suppressed by using ``load_pyfunc(suppress_warnings=True)``) (#230, @dbczumar)
- Pyfunc serve/predict will activate conda environment stored in MLModel. This can be disabled by adding ``--no-conda`` to ``mlflow pyfunc serve`` or ``mlflow pyfunc predict`` (#225, @0wu)
- Python SDK formalized in ``mlflow.tracking``. This includes adding SDK methods for ``get_run``, ``list_experiments``, ``get_experiment``, and ``set_terminated``. (#299, @aarondav)
- ``mlflow run`` can now be run against projects with no ``conda.yaml`` specified. By default, an empty conda environment will be created -- previously, it would just fail. You can still pass ``--no-conda`` to avoid entering a conda environment altogether (#218, @smurching)

Bug fixes:

- Fix numpy array serialization for int64 and other related types, allowing pyfunc to return such results (#240, @arinto)
- Fix DBFS artifactory calling ``log_artifacts`` with binary data (#295, @aarondav)
- Fix Run Command shown in UI to reproduce a run when the original run is targeted at a subdirectory of a Git repo (#294, @adrian555)
- Filter out ubiquitious dtype/ufunc warning messages (#317, @aarondav)
- Minor bug fixes and documentation updates (#261, @stbof; #279, @dmatrix; #313, @rbang1, #320, @yassineAlouini; #321, @tomasatdatabricks; #266, #282, #289, @smurching; #267, #265, @aarondav; #256, #290, @ToonKBC; #273, #263, @mateiz; #272, #319, @adrian555; #277, @aadamson; #283, #296, @andrewmchen)


0.4.2 (2018-08-07)
------------------

Breaking changes: None

Features:

- MLflow experiments REST API and ``mlflow experiments create`` now support providing ``--artifact-location`` (#232, @aarondav)
- [UI] Runs can now be sorted by columns, and added a Select All button (#227, @ToonKBC)
- Databricks File System (DBFS) artifactory support added (#226, @andrewmchen)
- databricks-cli version upgraded to >= 0.8.0 to support new DatabricksConfigProvider interface (#257, @aarondav)

Bug fixes:

- MLflow client sends REST API calls using snake_case instead of camelCase field names (#232, @aarondav)
- Minor bug fixes (#243, #242, @aarondav; #251, @javierluraschi; #245, @smurching; #252, @mateiz)


0.4.1 (2018-08-03)
------------------

Breaking changes: None

Features:

- [Projects] MLflow will use the conda installation directory given by the $MLFLOW_CONDA_HOME
  if specified (e.g. running conda commands by invoking "$MLFLOW_CONDA_HOME/bin/conda"), defaulting
  to running "conda" otherwise. (#231, @smurching)
- [UI] Show GitHub links in the UI for projects run from http(s):// GitHub URLs (#235, @smurching)

Bug fixes:

- Fix GCSArtifactRepository issue when calling list_artifacts on a path containing nested directories (#233, @jakeret)
- Fix Spark model support when saving/loading models to/from distributed filesystems (#180, @tomasatdatabricks)
- Add missing mlflow.version import to sagemaker module (#229, @dbczumar)
- Validate metric, parameter and run IDs in file store and Python client (#224, @mateiz)
- Validate that the tracking URI is a remote URI for Databricks project runs (#234, @smurching)
- Fix bug where we'd fetch git projects at SSH URIs into a local directory with the same name as
  the URI, instead of into a temporary directory (#236, @smurching)


0.4.0 (2018-08-01)
------------------

Breaking changes:

- [Projects] Removed the ``use_temp_cwd`` argument to ``mlflow.projects.run()``
  (``--new-dir`` flag in the ``mlflow run`` CLI). Runs of local projects now use the local project
  directory as their working directory. Git projects are still fetched into temporary directories
  (#215, @smurching)
- [Tracking] GCS artifact storage is now a pluggable dependency (no longer installed by default). 
  To enable GCS support, install ``google-cloud-storage`` on both the client and tracking server via pip.
  (#202, @smurching)
- [Tracking] Clients running MLflow 0.4.0 and above require a server running MLflow 0.4.0
  or above, due to a fix that ensures clients no longer double-serialize JSON into strings when
  sending data to the server (#200, @aarondav). However, the MLflow 0.4.0 server remains
  backwards-compatible with older clients (#216, @aarondav)


Features:

- [Examples] Add a more advanced tracking example: using MLflow with PyTorch and TensorBoard (#203)
- [Models] H2O model support (#170, @ToonKBC)
- [Projects] Support for running projects in subdirectories of Git repos (#153, @juntai-zheng)
- [SageMaker] Support for specifying a compute specification when deploying to SageMaker (#185, @dbczumar)
- [Server] Added --static-prefix option to serve UI from a specified prefix to MLflow UI and server (#116, @andrewmchen)
- [Tracking] Azure blob storage support for artifacts (#206, @mateiz)
- [Tracking] Add support for Databricks-backed RestStore (#200, @aarondav)
- [UI] Enable productionizing frontend by adding CSRF support (#199, @aarondav)
- [UI] Update metric and parameter filters to let users control column order (#186, @mateiz)

Bug fixes:

- Fixed incompatible file structure returned by GCSArtifactRepository (#173, @jakeret)
- Fixed metric values going out of order on x axis (#204, @mateiz)
- Fixed occasional hanging behavior when using the projects.run API (#193, @smurching)

- Miscellaneous bug and documentation fixes from @aarondav, @andrewmchen, @arinto, @jakeret, @mateiz, @smurching, @stbof


0.3.0 (2018-07-18)
------------------

Breaking changes:

- [MLflow Server] Renamed ``--artifact-root`` parameter to ``--default-artifact-root`` in ``mlflow server`` to better reflect its purpose (#165, @aarondav)

Features:

- Spark MLlib integration: we now support logging SparkML Models directly in the log_model API, model format, and serving APIs (#72, @tomasatdatabricks)
- Google Cloud Storage is now supported as an artifact storage root (#152, @bnekolny)
- Support asychronous/parallel execution of MLflow runs (#82, @smurching)
- [SageMaker] Support for deleting, updating applications deployed via SageMaker (#145, @dbczumar)
- [SageMaker] Pushing the MLflow SageMaker container now includes the MLflow version that it was published with (#124, @sueann)
- [SageMaker] Simplify parameters to SageMaker deploy by providing sane defaults (#126, @sueann)
- [UI] One-element metrics are now displayed as a bar char (#118, @cryptexis)

Bug fixes:

- Require gitpython>=2.1.0 (#98, @aarondav)
- Fixed TensorFlow model loading so that columns match the output names of the exported model (#94, @smurching)
- Fix SparkUDF when number of columns >= 10 (#97, @aarondav)
- Miscellaneous bug and documentation fixes from @emres, @dmatrix, @stbof, @gsganden, @dennyglee, @anabranch, @mikehuston, @andrewmchen, @juntai-zheng

0.2.1 (2018-06-28)
------------------

This is a patch release fixing some smaller issues after the 0.2.0 release.

- Switch protobuf implementation to C, fixing a bug related to tensorflow/mlflow import ordering (issues #33 and #77, PR #74, @andrewmchen)
- Enable running mlflow server without git binary installed (#90, @aarondav)
- Fix Spark UDF support when running on multi-node clusters (#92, @aarondav)

0.2.0 (2018-06-27)
------------------

- Added ``mlflow server`` to provide a remote tracking server. This is akin to ``mlflow ui`` with new options:

  - ``--host`` to allow binding to any ports (#27, @mdagost)
  - ``--artifact-root`` to allow storing artifacts at a remote location, S3 only right now (#78, @mateiz)
  - Server now runs behind gunicorn to allow concurrent requests to be made (#61, @mateiz)

- Tensorflow integration: we now support logging Tensorflow Models directly in the log_model API, model format, and serving APIs (#28, @juntai-zheng)
- Added ``experiments.list_experiments`` as part of experiments API (#37, @mparkhe)
- Improved support for unicode strings (#79, @smurching)
- Diabetes progression example dataset and training code (#56, @dennyglee)
- Miscellaneous bug and documentation fixes from @Jeffwan, @yupbank, @ndjido, @xueyumusic, @manugarri, @tomasatdatabricks, @stbof, @andyk, @andrewmchen, @jakeret, @0wu, @aarondav

0.1.0 (2018-06-05)
------------------

- Initial version of mlflow.
