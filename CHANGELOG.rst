Changelog
=========

1.0 (2019-06-03)
----------------
MLflow 1.0 includes many significant features and improvements. From this version, MLflow is no longer beta, and all APIs except those marked as experimental are intended to be stable until the next major version. As such, this release includes a number of breaking changes.

Major features, improvements, and breaking changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Support for recording, querying, and visualizing metrics along a new “step” axis (x coordinate), providing increased flexibility for examining model performance relative to training progress. For example, you can now record performance metrics as a function of the number of training iterations or epochs. MLflow 1.0’s enhanced metrics UI enables you to visualize the change in a metric’s value as a function of its step, augmenting MLflow’s existing UI for plotting a metric’s value as a function of wall-clock time. (#1202, #1237, @dbczumar; #1132, #1142, #1143, @smurching; #1211, #1225, @Zangr; #1372, @stbof)
- Search improvements. MLflow 1.0 includes additional support in both the API and UI for searching runs within a single experiment or a group of experiments. The search filter API supports a simplified version of the ``SQL WHERE`` clause. In addition to searching using run's metrics and params, the API has been enhanced to support a subset of run attributes as well as user and `system tags <https://mlflow.org/docs/latest/tracking.html#system-tags>`_. For details see `Search syntax <https://mlflow.org/docs/latest/search-syntax.html#syntax>`_ and `examples for programmatically searching runs <https://mlflow.org/docs/latest/search-syntax.html#programmatically-searching-runs>`_. (#1245, #1272, #1323, #1326, @mparkhe; #1052, @Zangr; #1363, @aarondav)
- Logging metrics in batches. MLflow 1.0 now has a ``runs/log-batch`` REST API endpoint for logging multiple metrics, params, and tags in a single API request. The endpoint useful for performant logging of multiple metrics at the end of a model training epoch (see `example <https://github.com/mlflow/mlflow/blob/bb8c7602dcb6a3a8786301fe6b98f01e8d3f288d/examples/hyperparam/search_hyperopt.py#L161>`_), or logging of many input model parameters at the start of training. You can call this batched-logging endpoint from Python (``mlflow.log_metrics``, ``mlflow.log_params``, ``mlflow.set_tags``), R (``mlflow_log_batch``), and Java (``MlflowClient.logBatch``). (#1214, @dbczumar; see 0.9.1 and 0.9.0 for other changes)
- Windows support for MLflow Tracking. The Tracking portion of the MLflow client is now supported on Windows. (#1171, @eedeleon, @tomasatdatabricks)
- HDFS support for artifacts. Hadoop artifact repository with Kerberos authorization support was added, so you can use HDFS to log and retrieve models and other artifacts. (#1011, @jaroslawk)
- CLI command to build Docker images for serving. Added an ``mlflow models build-docker`` CLI command for building a Docker image capable of serving an MLflow model. The model is served at port 8080 within the container by default. Note that this API is experimental and does not guarantee that the arguments nor format of the Docker container will remain the same. (#1329, @smurching, @tomasatdatabricks)
- New ``onnx`` model flavor for saving, loading, and evaluating ONNX models with MLflow. ONNX flavor APIs are available in the ``mlflow.onnx`` module. (#1127, @avflor, @dbczumar; #1388, @dbczumar)
- Major breaking changes:

  - Some of the breaking changes involve database schema changes in the SQLAlchemy tracking store. If your database instance's schema is not up-to-date, MLflow will issue an error at the start-up of ``mlflow server`` or ``mlflow ui``. To migrate an existing database to the newest schema, you can use the ``mlflow db upgrade`` CLI command. (#1155, #1371, @smurching; #1360, @aarondav)
  - [Installation] The MLflow Python package no longer depends on ``scikit-learn``, ``mleap``, or ``boto3``. If you want to use the ``scikit-learn`` support, the ``MLeap`` support, or ``s3`` artifact repository / ``sagemaker`` support, you will have to install these respective dependencies explicitly. (#1223, @aarondav)
  - [Artifacts] In the Models API, an artifact's location is now represented as a URI. See the `documentation <https://mlflow.org/docs/latest/tracking.html#artifact-locations>`_ for the list of accepted URIs. (#1190, #1254, @dbczumar; #1174, @dbczumar, @sueann; #1206, @tomasatdatabricks; #1253, @stbof)

    - The affected methods are:

      - Python: ``<model-type>.load_model``, ``azureml.build_image``, ``sagemaker.deploy``, ``sagemaker.run_local``, ``pyfunc._load_model_env``, ``pyfunc.load_pyfunc``, and ``pyfunc.spark_udf``
      - R: ``mlflow_load_model``, ``mlflow_rfunc_predict``, ``mlflow_rfunc_serve``
      - CLI: ``mlflow models serve``, ``mlflow models predict``, ``mlflow sagemaker``, ``mlflow azureml`` (with the new ``--model-uri`` option)

    - To allow referring to artifacts in the context of a run, MLflow introduces a new URI scheme of the form ``runs:/<run_id>/relative/path/to/artifact``. (#1169, #1175, @sueann)

  - [CLI] ``mlflow pyfunc`` and ``mlflow rfunc`` commands have been unified as ``mlflow models`` (#1257, @tomasatdatabricks; #1321, @dbczumar)
  - [CLI] ``mlflow artifacts download``, ``mlflow artifacts download-from-uri`` and ``mlflow download`` commands have been consolidated into ``mlflow artifacts download`` (#1233, @sueann)
  - [Runs] Expose ``RunData`` fields (``metrics``, ``params``, ``tags``) as dictionaries. Note that the ``mlflow.entities.RunData`` constructor still accepts lists of ``metric``/``param``/``tag`` entities. (#1078, @smurching)
  - [Runs] Rename ``run_uuid`` to ``run_id`` in Python, Java, and REST API. Where necessary, MLflow will continue to accept ``run_uuid`` until MLflow 1.1. (#1187, @aarondav)

Other breaking changes
~~~~~~~~~~~~~~~~~~~~~~

CLI:

- The ``--file-store`` option is deprecated in ``mlflow server`` and ``mlflow ui`` commands. (#1196, @smurching)
- The ``--host`` and ``--gunicorn-opts`` options are removed in the ``mlflow ui`` command. (#1267, @aarondav)
- Arguments to ``mlflow experiments`` subcommands, notably ``--experiment-name`` and ``--experiment-id`` are now options (#1235, @sueann)
- ``mlflow sagemaker list-flavors`` has been removed (#1233, @sueann)

Tracking:

- The ``user`` property of ``Run``s has been moved to tags (similarly, the ``run_name``, ``source_type``, ``source_name`` properties were moved to tags in 0.9.0). (#1230, @acroz; #1275, #1276, @aarondav)
- In R, the return values of experiment CRUD APIs have been updated to more closely match the REST API. In particular, ``mlflow_create_experiment`` now returns a string experiment ID instead of an experiment, and the other APIs return NULL. (#1246, @smurching)
- ``RunInfo.status``'s type is now string. (#1264, @mparkhe)
- Remove deprecated ``RunInfo`` properties from ``start_run``. (#1220, @aarondav)
- As deprecated in 0.9.1 and before, the ``RunInfo`` fields ``run_name``, ``source_name``, ``source_version``, ``source_type``, and ``entry_point_name`` and the ``SearchRuns`` field ``anded_expressions`` have been removed from the REST API and Python, Java, and R tracking client APIs. They are still available as tags, documented in the REST API documentation. (#1188, @aarondav)

Models and deployment:

- In Python, require arguments as keywords in ``log_model``, ``save_model`` and ``add_to_model`` methods in the ``tensorflow`` and ``mleap`` modules to avoid breaking changes in the future (#1226, @sueann)
- Remove the unsupported ``jars`` argument from ```spark.log_model`` in Python (#1222, @sueann)
- Introduce ``pyfunc.load_model`` to be consistent with other Models modules. ``pyfunc.load_pyfunc`` will be deprecated in the near future. (#1222, @sueann)
- Rename ``dst_path`` parameter in ``pyfunc.save_model`` to ``path`` (#1221, @aarondav)
- R flavors refactor (#1299, @kevinykuo)

  - ``mlflow_predict()`` has been added in favor of ``mlflow_predict_model()`` and ``mlflow_predict_flavor()`` which have been removed.
  - ``mlflow_save_model()`` is now a generic and ``mlflow_save_flavor()`` is no longer needed and has been removed.
  - ``mlflow_predict()`` takes ``...`` to pass to underlying predict methods.
  - ``mlflow_load_flavor()`` now has the signature ``function(flavor, model_path)`` and flavor authors should implement ``mlflow_load_flavor.mlflow_flavor_{FLAVORNAME}``. The flavor argument is inferred from the inputs of user-facing ``mlflow_load_model()`` and does not need to be explicitly provided by the user.

Projects:

- Remove and rename some ``projects.run`` parameters for generality and consistency. (#1222, @sueann)
- In R, the ``mlflow_run`` API for running MLflow projects has been modified to more closely reflect the Python ``mlflow.run`` API. In particular, the order of the ``uri`` and ``entry_point`` arguments has been reversed and the ``param_list`` argument has been renamed to ``parameters``. (#1265, @smurching)

R:

- Remove ``mlflow_snapshot`` and ``mlflow_restore_snapshot`` APIs. Also, the ``r_dependencies`` argument used to specify the path to a packrat r-dependencies.txt file has been removed from all APIs. (#1263, @smurching)
- The ``mlflow_cli`` and ``crate`` APIs are now private. (#1246, @smurching)

Environment variables:

- Prefix environment variables with "MLFLOW_" (#1268, @aarondav). Affected variables are: 

  - [Tracking] ``_MLFLOW_SERVER_FILE_STORE``, ``_MLFLOW_SERVER_ARTIFACT_ROOT``, ``_MLFLOW_STATIC_PREFIX``
  - [SageMaker] ``MLFLOW_SAGEMAKER_DEPLOY_IMG_URL``, ``MLFLOW_DEPLOYMENT_FLAVOR_NAME``
  - [Scoring] ``MLFLOW_SCORING_SERVER_MIN_THREADS``, ``MLFLOW_SCORING_SERVER_MAX_THREADS``

More features and improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- [Tracking] Non-default driver support for SQLAlchemy backends: ``db+driver`` is now a valid tracking backend URI scheme (#1297, @drewmcdonald; #1374, @mparkhe)
- [Tracking] Validate backend store URI before starting tracking server (#1218, @luke-zhu, @sueann)
- [Tracking] Add ``GetMetricHistory`` client API in Python and Java corresponding to the REST API. (#1178, @smurching)
- [Tracking] Add ``view_type`` argument to ``MlflowClient.list_experiments()`` in Python. (#1212, @smurching)
- [Tracking] Dictionary values provided to ``mlflow.log_params`` and ``mlflow.set_tags`` in Python can now be non-string types (e.g., numbers), and they are automatically converted to strings. (#1364, @aarondav)
- [Tracking] R API additions to be at parity with REST API and Python (#1122, @kevinykuo)
- [Tracking] Limit number of results returned from ``SearchRuns`` API and UI for faster load (#1125, @mparkhe; #1154, @andrewmchen)
- [Artifacts] To avoid having many copies of large model files in serving, ``ArtifactRepository.download_artifacts`` no longer copies local artifacts (#1307, @andrewmchen; #1383, @dbczumar)
- [Artifacts][Projects] Support GCS in download utilities. ``gs://bucket/path`` files are now supported by the ``mlflow artifacts download`` CLI command and as parameters of type ``path`` in MLProject files. (#1168, @drewmcdonald)
- [Models] All Python models exported by MLflow now declare ``mlflow`` as a dependency by default. In addition, we introduce a flag ``--install-mlflow`` users can pass to ``mlflow models serve`` and ``mlflow models predict`` methods to force installation of the latest version of MLflow into the model's environment. (#1308, @tomasatdatabricks)
- [Models] Update model flavors to lazily import dependencies in Python. Modules that define Model flavors now import extra dependencies such as ``tensorflow``, ``scikit-learn``, and ``pytorch`` inside individual _methods_, ensuring that these modules can be imported and explored even if the dependencies have not been installed on your system. Also, the ``DEFAULT_CONDA_ENVIRONMENT`` module variable has been replaced with a ``get_default_conda_env()`` function for each flavor.
- [Models] It is now possible to pass extra arguments to ``mlflow.keras.load_model`` that will be passed through to ``keras.load_model``. (#1330, @@yorickvP)
- [Serving] For better performance, switch to ``gunicorn`` for serving Python models. This does not change the user interface. (#1322, @tomasatdatabricks)
- [Deployment] For SageMaker, use the uniquely-generated model name as the S3 bucket prefix instead of requiring one. (#1183, @dbczumar)
- [REST API] Add support for API paths without the ``preview`` component. The ``preview`` paths will be deprecated in a future version of MLflow. (#1236, @mparkhe)

Bug fixes and documentation updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- [Tracking] Log metric timestamps in milliseconds by default (#1177, @smurching; #1333, @dbczumar)
- [Tracking] Fix bug when deserializing integer experiment ID for runs in ``SQLAlchemyStore`` (#1167, @smurching)
- [Tracking] Ensure unique constraint names in MLflow tracking database (#1292, @smurching)
- [Tracking] Fix base64 encoding for basic auth in R tracking client (#1126, @freefrag)
- [Tracking] Correctly handle ``file:`` URIs for the ``-—backend-store-uri`` option in ``mlflow server`` and ``mlflow ui`` CLI commands (#1171, @eedeleon, @tomasatdatabricks)
- [Artifacts] Update artifact repository download methods to return absolute paths (#1179, @dbczumar)
- [Artifacts] Make FileStore respect the default artifact location (#1332, @dbczumar)
- [Artifacts] Fix ``log_artifact`` failures due to existing directory on FTP server (#1327, @kafendt)
- [Artifacts] Fix GCS artifact logging of subdirectories (#1285, @jason-huling)
- [Projects] Fix bug not sharing ``SQLite`` database file with Docker container (#1347, @tomasatdatabricks; #1375, @aarondav)
- [Java] Mark ``sendPost`` and ``sendGet`` as experimental (#1186, @aarondav)
- [Python][CLI] Mark ``azureml.build_image`` as experimental (#1222, #1233 @sueann)
- [Docs] Document public MLflow environment variables (#1343, @aarondav)
- [Docs] Document MLflow system tags for runs (#1342, @aarondav)
- [Docs] Autogenerate CLI documentation to include subcommands and descriptions (#1231, @sueann)
- [Docs] Update run selection description in ``mlflow_get_run`` in R documentation (#1258, @dbczumar)
- [Examples] Update examples to reflect API changes (#1361, @tomasatdatabricks; #1367, @mparkhe)

Small bug fixes and doc updates (#1359, #1350, #1331, #1301, #1270, #1271, #1180, #1144, #1135, #1131, #1358, #1369, #1368, #1387, @aarondav; #1373, @akarloff; #1287, #1344, #1309, @stbof; #1312, @hchiuzhuo; #1348, #1349, #1294, #1227, #1384, @tomasatdatabricks; #1345, @withsmilo; #1316, @ancasarb; #1313, #1310, #1305, #1289, #1256, #1124, #1097, #1162, #1163, #1137, #1351, @smurching; #1319, #1244, #1224, #1195, #1194, #1328, @dbczumar; #1213, #1200, @Kublai-Jing; #1304, #1320, @andrewmchen; #1311, @Zangr; #1306, #1293, #1147, @mateiz; #1303, @gliptak; #1261, #1192, @eedeleon; #1273, #1259, @kevinykuo; #1277, #1247, #1243, #1182, #1376, @mparkhe; #1210, @vgod-dbx; #1199, @ashtuchkin; #1176, #1138, #1365, @sueann; #1157, @cclauss; #1156, @clemens-db; #1152, @pogil; #1146, @srowen; #875, #1251, @jimthompson5802)


0.9.1 (2019-04-21)
------------------
MLflow 0.9.1 is a patch release on top of 0.9.0 containing mostly bug fixes and internal improvements. We have also included a one breaking API change in preparation for additions in MLflow 1.0 and later. This release also includes significant improvements to the Search API.

Breaking changes:

- [Tracking] Generalized experiment_id to string (from a long) to be more permissive of different ID types in different backend stores. While breaking for the REST API, this change is backwards compatible for python and R clients. (#1067, #1034 @eedeleon)

More features and improvements:

- [Search][API] Moving search filters into a query string based syntax, with Java client, Python client, and UI support. This also improves quote, period, and special character handling in query strings and adds the ability to search on tags in filter string. (#1042, #1055, #1063, #1068, #1099, #1106 @mparkhe; #1025 @andrewmchen; #1060 @smurching)
- [Tracking] Limits and validations to batch-logging APIs in OSS server (#958 @smurching)
- [Tracking][Java] Java client API for batch-logging (#1081 @mparkhe)
- [Tracking] Improved consistency of handling multiple metric values per timestamp across tracking stores (#972, #999 @dbczumar)

Bug fixes and documentation updates:

- [Tracking][Python] Reintroduces the parent_run_id argument to MlflowClient.create_run. This API is planned for removal in MLflow 1.0 (#1137 @smurching)
- [Tracking][Python] Provide default implementations of AbstractStore log methods (#1051 @acroz)
- [R] (Released on CRAN as MLflow 0.9.0.1) Small bug fixes with R (#1123 @smurching; #1045, #1017, #1019, #1039, #1048, #1098,  #1101, #1107, #1108, #1119 @tomasatdatabricks)

Small bug fixes and doc updates (#1024, #1029 @bayethiernodiop; #1075 @avflor; #968, #1010, #1070, #1091, #1092 @smurching; #1004, #1085 @dbczumar; #1033, #1046 @sueann; #1053 @tomasatdatabricks; #987 @hanyucui; #935, #941 @jimthompson5802; #963 @amilbourne; #1016 @andrewmchen; #991 @jaroslawk; #1007 @mparkhe)


0.9.0.1 (2019-04-09)
--------------------
Bugfix release (PyPI only) with the following changes:

- Rebuilt MLflow JS assets to fix an issue where form input was broken in MLflow 0.9.0 (identified
  in #1056, #1113 by @shu-yusa, @timothyjlaurent)


0.9.0 (2019-03-13)
------------------

Major features:

- Support for running MLflow Projects in Docker containers. This allows you to include non-Python dependencies in their project environments and provides stronger isolation when running projects. See the `Projects documentation <https://mlflow.org/docs/latest/projects.html>`_ for more information. (#555, @marcusrehm; #819, @mparkhe; #970, @dbczumar)
- Database stores for the MLflow Tracking Server. Support for a scalable and performant backend store was one of the top community requests. This feature enables you to connect to local or remote SQLAlchemy-compatible databases (currently supported flavors include MySQL, PostgreSQL, SQLite, and MS SQL) and is compatible with file backed store. See the `Tracking Store documentation <https://mlflow.org/docs/latest/tracking.html#storage>`_ for more information. (#756, @AndersonReyes; #800, #844, #847, #848, #860, #868, #975, @mparkhe; #980, @dbczumar)
- Simplified custom Python model packaging. You can easily include custom preprocessing and postprocessing logic, as well as data dependencies in models with the ``python_function`` flavor using updated ``mlflow.pyfunc`` Python APIs. For more information, see the `Custom Python Models documentation <https://mlflow.org/docs/latest/models.html#custom-python-models>`_. (#791, #792, #793, #830, #910, @dbczumar)
- Plugin systems allowing third party libraries to extend MLflow functionality. The `proposal document <https://gist.github.com/zblz/9e337a55a7ba73314890be68370fa69a>`_ gives the full detail of the three main changes: 

  - You can register additional providers of tracking stores using the ``mlflow.tracking_store`` entrypoint. (#881, @zblz)
  - You can register additional providers of artifact repositories using the ``mlflow.artifact_repository`` entrypoint. (#882, @mociarain)
  - The logic generating run metadata from the run context (e.g. ``source_name``, ``source_version``) has been refactored into an extendable system of run context providers. Plugins can register additional providers using the ``mlflow.run_context_provider`` entrypoint, which add to or overwrite tags set by the base library. (#913, #926, #930, #978, @acroz)

- Support for HTTP authentication to the Tracking Server in the R client. Now you can connect to secure Tracking Servers using credentials set in environment variables, or provide custom plugins for setting the credentials. As an example, this release contains a Databricks plugin that can detect existing Databricks credentials to allow you to connect to the Databricks Tracking Server. (#938, #959, #992, @tomasatdatabricks)


Breaking changes:

- [Scoring] The ``pyfunc`` scoring server now expects requests with the ``application/json`` content type to contain json-serialized pandas dataframes in the split format, rather than the records format. See the `documentation on deployment <https://mlflow.org/docs/latest/models.html#deploy-a-python-function-model-as-a-local-rest-api-endpoint>`_ for more detail. (#960, @dbczumar) Also, when reading the pandas dataframes from JSON, the scoring server no longer automatically infers data types as it can result in unintentional conversion of data types (#916, @mparkhe).
- [API] Remove ``GetMetric`` & ``GetParam`` from the REST API as they are subsumed by ``GetRun``. (#879, @aarondav)


More features and improvements:

- [UI] Add a button for downloading artifacts (#967, @mateiz)
- [CLI] Add CLI commands for runs: now you can ``list``, ``delete``, ``restore``, and ``describe`` runs through the CLI (#720, @DorIndivo)
- [CLI] The ``run`` command now can take ``--experiment-name`` as an argument, as an alternative to the ``--experiment-id`` argument. You can also choose to set the ``_EXPERIMENT_NAME_ENV_VAR`` environment variable instead of passing in the value explicitly. (#889, #894, @mparkhe)
- [Examples] Add Image classification example with Keras. (#743, @tomasatdatabricks )
- [Artifacts] Add ``get_artifact_uri()`` and ``_download_artifact_from_uri`` convenience functions (#779)
- [Artifacts] Allow writing Spark models directly to the target artifact store when possible (#808, @smurching)
- [Models] PyTorch model persistence improvements to allow persisting definitions and dependencies outside the immediate scope:
  - Add a ``code_paths`` parameter to ``mlflow.pytorch.save_model`` and ``mlflow.pytorch.log_model`` to allow external module dependencies to be specified as paths to python files. (#842, @dbczumar)
  - Improve ``mlflow.pytorch.save_model`` to capture class definitions from notebooks and the ``__main__`` scope (#851, #861, @dbczumar)
- [Runs][R] Allow client to infer context info when creating new run in fluent API (#958, @tomasatdatabricks)
- [Runs][UI] Support Git Commit hyperlink for Gitlab and Bitbucket. Previously the clickable hyperlink was generated only for Github pages. (#901)
- [Search][API] Allow param value to have any content, not just alphanumeric characters, ``.``, and ``-`` (#788, @mparkhe)
- [Search][API] Support "filter" string in the ``SearchRuns`` API. Corresponding UI improvements are planned for the future (#905, @mparkhe)
- [Logging] Basic support for LogBatch. NOTE: The feature is currently experimental and the behavior is expected to change in the near future. (#950, #951, #955, #1001, @smurching)


Bug fixes and documentation updates:

- [Artifacts] Fix empty-file upload to DBFS in ``log_artifact`` and ``log_artifacts`` (#895, #818, @smurching)
- [Artifacts] S3 artifact store: fix path resolution error when artifact root is bucket root (#928, @dbczumar)
- [UI] Fix a bug with Databricks notebook URL links (#891, @smurching)
- [Export] Fix for missing run name in csv export (#864, @jimthompson5802)
- [Example] Correct missing tensorboardX module error in PyTorch example when running in MLflow Docker container (#809, @jimthompson5802)
- [Scoring][R] Fix local serving of rfunc models (#874, @kevinykuo)
- [Docs] Improve flavor-specific documentation in Models documentation (#909, @dbczumar)

Small bug fixes and doc updates (#822, #899, #787, #785, #780, #942, @hanyucui; #862, #904, #954, #806, #857, #845, @stbof; #907, #872, @smurching; #896, #858, #836, #859, #923, #939, #933, #931, #952, @dbczumar; #880, @zblz; #876, @acroz; #827, #812, #816, #829, @jimthompson5802; #837, #790, #897, #974, #900, @mparkhe; #831, #798, @aarondav; #814, @sueann; #824, #912, @mateiz; #922, #947, @tomasatdatabricks; #795, @KevYuen; #676, @mlaradji; #906, @4n4nd; #777, @tmielika; #804, @alkersan)


0.8.2 (2019-01-28)
------------------

MLflow 0.8.2 is a patch release on top of 0.8.1 containing only bug fixes and no breaking changes or features.

Bug fixes:

- [Python API] CloudPickle has been added to the set of MLflow library dependencies, fixing missing import errors when attempting to save models (#777, @tmielika)
- [Python API] Fixed a malformed logging call that prevented ``mlflow.sagemaker.push_image_to_ecr()`` invocations from succeeding (#784, @jackblandin)
- [Models] PyTorch models can now be saved with code dependencies, allowing model classes to be loaded successfully in new environments (#842, #836, @dbczumar)
- [Artifacts] Fixed a timeout when logging zero-length files to DBFS artifact stores (#818, @smurching)

Small docs updates (#845, @stbof; #840, @grahamhealy20; #839, @wilderrodrigues)


0.8.1 (2018-12-21)
------------------

MLflow 0.8.1 introduces several significant improvements:

- Improved UI responsiveness and load time, especially when displaying experiments containing hundreds to thousands of runs.
- Improved visualizations, including interactive scatter plots for MLflow run comparisons
- Expanded support for scoring Python models as Spark UDFs. For more information, see the `updated documentation for this feature <https://mlflow.org/docs/latest/models.html#export-a-python-function-model-as-an-apache-spark-udf>`_.
- By default, saved models will now include a Conda environment specifying all of the dependencies necessary for loading them in a new environment.

Features:

- [API/CLI] Support for running MLflow projects from ZIP files (#759, @jmorefieldexpe)
- [Python API] Support for passing model conda environments as dictionaries to ``save_model`` and ``log_model`` functions (#748, @dbczumar)
- [Models] Default Anaconda environments have been added to many Python model flavors. By default, models produced by `save_model` and `log_model` functions will include an environment that specifies all of the versioned dependencies necessary to load and serve the models. Previously, users had to specify these environments manually. (#705, #707, #708, #749, @dbczumar)
- [Scoring] Support for synchronous deployment of models to SageMaker (#717, @dbczumar)
- [Tracking] Include the Git repository URL as a tag when tracking an MLflow run within a Git repository (#741, @whiletruelearn, @mateiz)
- [UI] Improved runs UI performance by using a react-virtualized table to optimize row rendering (#765, #762, #745, @smurching)
- [UI] Significant performance improvements for rendering run metrics, tags, and parameter information (#764, #747, @smurching)
- [UI] Scatter plots, including run comparsion plots, are now interactive (#737, @mateiz)
- [UI] Extended CSRF support by allowing the MLflow UI server to specify a set of expected headers that clients should set when making AJAX requests (#733, @aarondav)

Bug fixes and documentation updates:

- [Python/Scoring] MLflow Python models that produce Pandas DataFrames can now be evaluated as Spark UDFs correctly. Spark UDF outputs containing multiple columns of primitive types are now supported (#719, @tomasatdatabricks)
- [Scoring] Fixed a serialization error that prevented models served with Azure ML from returning Pandas DataFrames (#754, @dbczumar)
- [Docs] New example demonstrating how the MLflow REST API can be used to create experiments and log run information (#750, kjahan)
- [Docs] R documentation has been updated for clarity and style consistency (#683, @stbof)
- [Docs] Added clarification about user setup requirements for executing remote MLflow runs on Databricks (#736, @andyk)

Small bug fixes and doc updates (#768, #715, @smurching; #728, dodysw; #730, mshr-h; #725, @kryptec; #769, #721, @dbczumar; #714, @stbof)


0.8.0 (2018-11-08)
-----------------

MLflow 0.8.0 introduces several major features:

- Dramatically improved UI for comparing experiment run results:

  - Metrics and parameters are by default grouped into a single column, to avoid an explosion of mostly-empty columns. Individual metrics and parameters can be moved into their own column to help compare across rows.
  - Runs that are "nested" inside other runs (e.g., as part of a hyperparameter search or multistep workflow) now show up grouped by their parent run, and can be expanded or collapsed altogether. Runs can be nested by calling ``mlflow.start_run`` or ``mlflow.run`` while already within a run.
  - Run names (as opposed to automatically generated run UUIDs) now show up instead of the run ID, making comparing runs in graphs easier.
  - The state of the run results table, including filters, sorting, and expanded rows, is persisted in browser local storage, making it easier to go back and forth between an individual run view and the table.

- Support for deploying models as Docker containers directly to Azure Machine Learning Service Workspace (as opposed to the previously-recommended solution of Azure ML Workbench).


Breaking changes:

- [CLI] ``mlflow sklearn serve`` has been removed in favor of ``mlflow pyfunc serve``, which takes the same arguments but works against any pyfunc model (#690, @dbczumar)


Features:

- [Scoring] pyfunc server and SageMaker now support the pandas "split" JSON format in addition to the "records" format. The split format allows the client to specify the order of columns, which is necessary for some model formats. We recommend switching client code over to use this new format (by sending the Content-Type header ``application/json; format=pandas-split``), as it will become the default JSON format in MLflow 0.9.0. (#690, @dbczumar)
- [UI] Add compact experiment view (#546, #620, #662, #665, @smurching)
- [UI] Add support for viewing & tracking nested runs in experiment view (#588, @andrewmchen; #618, #619, @aarondav)
- [UI] Persist experiments view filters and sorting in browser local storage (#687, @smurching)
- [UI] Show run name instead of run ID when present (#476, @smurching)
- [Scoring] Support for deploying Models directly to Azure Machine Learning Service Workspace (#631, @dbczumar)
- [Server/Python/Java] Add ``rename_experiment`` to Tracking API (#570, @aarondav)
- [Server] Add ``get_experiment_by_name`` to RestStore (#592, @dmarkhas)
- [Server] Allow passing gunicorn options when starting mlflow server (#626, @mparkhe)
- [Python] Cloudpickle support for sklearn serialization (#653, @dbczumar)
- [Artifacts] FTP artifactory store added (#287, @Shenggan)


Bug fixes and documentation updates:

- [Python] Update TensorFlow integration to match API provided by other flavors (#612, @dbczumar; #670, @mlaradji)
- [Python] Support for TensorFlow 1.12 (#692, @smurching)
- [R] Explicitly loading Keras module at predict time no longer required (#586, @kevinykuo)
- [R] pyfunc serve can correctly load models saved with the R Keras support (#634, @tomasatdatabricks)
- [R] Increase network timeout of calls to the RestStore from 1 second to 60 seconds (#704, @aarondav)
- [Server] Improve errors returned by RestStore (#582, @andrewmchen; #560, @smurching)
- [Server] Deleting the default experiment no longer causes it to be immediately recreated (#604, @andrewmchen; #641, @schipiga)
- [Server] Azure Blob Storage artifact repo supports Windows paths (#642, @marcusrehm)
- [Server] Improve behavior when environment and run files are corrupted (#632, #654, #661, @mparkhe)
- [UI] Improve error page when viewing nonexistent runs or views (#600, @andrewmchen; #560, @andrewmchen)
- [UI] UI no longer throws an error if all experiments are deleted (#605, @andrewmchen)
- [Docs] Include diagram of workflow for multistep example (#581, @dennyglee)
- [Docs] Add reference tags and R and Java APIs to tracking documentation (#514, @stbof)
- [Docs/R] Use CRAN installation (#686, @javierluraschi)

Small bug fixes and doc updates (#576, #594, @javierluraschi; #585, @kevinykuo; #593, #601, #611, #650, #669, #671, #679, @dbczumar; #607, @suzil; #583, #615, @andrewmchen; #622, #681, @aarondav; #625, @pogil; #589, @tomasatdatabricks; #529, #635, #684, @stbof; #657, @mvsusp; #682, @mateiz; #678, vfdev-5; #596, @yutannihilation; #663, @smurching)


0.7.0 (2018-10-01)
-----------------

MLflow 0.7.0 introduces several major features:

- An R client API (to be released on CRAN soon)
- Support for deleting runs (API + UI)
- UI support for adding notes to a run

The release also includes bugfixes and improvements across the Python and Java clients, tracking UI,
and documentation.

Breaking changes:

- [Python] The per-flavor implementation of load_pyfunc has been made private (#539, @tomasatdatabricks)
- [REST API, Java] logMetric now accepts a double metric value instead of a float (#566, @aarondav)

Features:

- [R] Support for R (#370, #471, @javierluraschi; #548 @kevinykuo)
- [UI] Add support for adding notes to Runs (#396, @aadamson)
- [Python] Python API, REST API, and UI support for deleting Runs (#418, #473, #526, #579 @andrewmchen)
- [Python] Set a tag containing the branch name when executing a branch of a Git project (#469, @adrian555)
- [Python] Add a set_experiment API to activate an experiment before starting runs (#462, @mparkhe)
- [Python] Add arguments for specifying a parent run to tracking & projects APIs (#547, @andrewmchen)
- [Java] Add Java set tag API (#495, @smurching)
- [Python] Support logging a conda environment with sklearn models (#489, @dbczumar)
- [Scoring] Support downloading MLflow scoring JAR from Maven during scoring container build (#507, @dbczumar)


Bug fixes:

- [Python] Print errors when the Databricks run fails to start (#412, @andrewmchen)
- [Python] Fix Spark ML PyFunc loader to work on Spark driver (#480, @tomasatdatabricks)
- [Python] Fix Spark ML load_pyfunc on distributed clusters (#490, @tomasatdatabricks)
- [Python] Fix error when downloading artifacts from a run's artifact root (#472, @dbczumar)
- [Python] Fix DBFS upload file-existence-checking logic during Databricks project execution (#510, @smurching)
- [Python] Support multi-line and unicode tags (#502, @mparkhe)
- [Python] Add missing DeleteExperiment, RestoreExperiment implementations in the Python REST API client (#551, @mparkhe)
- [Scoring] Convert Spark DataFrame schema to an MLeap schema prior to serialization (#540, @dbczumar)
- [UI] Fix bar chart always showing in metric view (#488, @smurching)


Small bug fixes and doc updates (#467 @drorata; #470, #497, #508, #518 @dbczumar;
#455, #466, #492, #504, #527 @aarondav; #481, #475, #484, #496, #515, #517, #498, #521, #522,
#573 @smurching; #477 @parkerzf; #494 @jainr; #501, #531, #532, #552 @mparkhe; #503, #520 @dmatrix;
#509, #532 @tomasatdatabricks; #484, #486 @stbof; #533, #534 @javierluraschi;
#542 @GCBallesteros; #511 @AdamBarnhard)


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
- [Models] Keras model export now includes TensorFlow graph explicitly to ensure the model can always be loaded at deployment time (#440, @tomasatdatabricks)
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

- TensorFlow integration: we now support logging TensorFlow Models directly in the log_model API, model format, and serving APIs (#28, @juntai-zheng)
- Added ``experiments.list_experiments`` as part of experiments API (#37, @mparkhe)
- Improved support for unicode strings (#79, @smurching)
- Diabetes progression example dataset and training code (#56, @dennyglee)
- Miscellaneous bug and documentation fixes from @Jeffwan, @yupbank, @ndjido, @xueyumusic, @manugarri, @tomasatdatabricks, @stbof, @andyk, @andrewmchen, @jakeret, @0wu, @aarondav

0.1.0 (2018-06-05)
------------------

- Initial version of mlflow.
