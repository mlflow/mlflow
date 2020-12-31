Changelog
=========
1.13.1 (2020-12-30)
-----------------
MLflow 1.13.1 is a patch release containing bug fixes and small changes:

- Fix bug causing Spark autologging to ignore configuration options specified by ``mlflow.autolog()`` (#3917, @dbczumar)
- Fix bugs causing metrics to be dropped during TensorFlow autologging (#3913, #3914, @dbczumar)
- Fix incorrect value of optimizer name parameter in autologging PyTorch Lightning (#3901, @harupy)
- Fix model registry database ``allow_null_for_run_id`` migration failure affecting MySQL databases (#3836, @t-henri)
- Fix failure in ``transition_model_version_stage`` when uncanonical stage name is passed (#3929, @harupy)
- Fix an undefined variable error causing AzureML model deployment to fail (#3922, @eedeleon)
- Reclassify scikit-learn as a pip dependency in MLflow Model conda environments (#3896, @harupy)
- Fix experiment view crash and artifact view inconsistency caused by artifact URIs with redundant slashes (#3928, @dbczumar)

1.13 (2020-12-22)
-----------------
MLflow 1.13 includes several major features and improvements:

Features:

New fluent APIs for logging in-memory objects as artifacts:

- Add ``mlflow.log_text`` which logs text as an artifact (#3678, @harupy)
- Add ``mlflow.log_dict`` which logs a dictionary as an artifact (#3685, @harupy)
- Add ``mlflow.log_figure`` which logs a figure object as an artifact (#3707, @harupy)
- Add ``mlflow.log_image`` which logs an image object as an artifact (#3728, @harupy)

UI updates / fixes (#3867, @smurching):

- Add model version link in compact experiment table view
- Add logged/registered model links in experiment runs page view
- Enhance artifact viewer for MLflow models
- Model registry UI settings are now persisted across browser sessions
- Add model version ``description`` field to model version table

Autologging enhancements:

- Improve robustness of autologging integrations to exceptions (#3682, #3815, dbczumar; #3860, @mohamad-arabi; #3854, #3855, #3861, @harupy)
- Add ``disable`` configuration option for autologging (#3682, #3815, dbczumar; #3838, @mohamad-arabi; #3854, #3855, #3861, @harupy)
- Add ``exclusive`` configuration option for autologging (#3851, @apurva-koti; #3869, @dbczumar)
- Add ``log_models`` configuration option for autologging (#3663, @mohamad-arabi)
- Set tags on autologged runs for easy identification (and add tags to start_run) (#3847, @dbczumar)

More features and improvements:

- Allow Keras models to be saved with ``SavedModel`` format (#3552, @skylarbpayne)
- Add support for ``statsmodels`` flavor (#3304, @olbapjose)
- Add support for nested-run in mlflow R client (#3765, @yitao-li)
- Deploying a model using ``mlflow.azureml.deploy`` now integrates better with the AzureML tracking/registry. (#3419, @trangevi)
- Update schema enforcement to handle integers with missing values (#3798, @tomasatdatabricks)

Bug fixes and documentation updates:

- When running an MLflow Project on Databricks, the version of MLflow installed on the Databricks cluster will now match the version used to run the Project (#3880, @FlorisHoogenboom)
- Fix bug where metrics are not logged for single-epoch ``tf.keras`` training sessions (#3853, @dbczumar)
- Reject boolean types when logging MLflow metrics (#3822, @HCoban)
- Fix alignment of Keras / ``tf.Keras`` metric history entries when ``initial_epoch`` is different from zero. (#3575, @garciparedes)
- Fix bugs in autologging integrations for newer versions of TensorFlow and Keras (#3735, @dbczumar)
- Drop global ``filterwwarnings`` module at import time (#3621, @jogo)
- Fix bug that caused preexisting Python loggers to be disabled when using MLflow with the SQLAlchemyStore (#3653, @arthury1n)
- Fix ``h5py`` library incompatibility for exported Keras models (#3667, @tomasatdatabricks)

Small changes, bug fixes and doc updates (#3887, #3882, #3845, #3833, #3830, #3828, #3826, #3825, #3800, #3809, #3807, #3786, #3794, #3731, #3776, #3760, #3771, #3754, #3750, #3749, #3747, #3736, #3701, #3699, #3698, #3658, #3675, @harupy; #3723, @mohamad-arabi; #3650, #3655, @shrinath-suresh; #3850, #3753, #3725, @dmatrix; ##3867, #3670, #3664, @smurching; #3681, @sueann; #3619, @andrewnitu; #3837, @javierluraschi; #3721, @szczeles; #3653, @arthury1n; #3883, #3874, #3870, #3877, #3878, #3815, #3859, #3844, #3703, @dbczumar; #3768, @wentinghu; #3784, @HCoban; #3643, #3649, @arjundc-db; #3864, @AveshCSingh, #3756, @yitao-li)

1.12.1 (2020-11-19)
-------------------
MLflow 1.12.1 is a patch release containing bug fixes and small changes:

- Fix ``run_link`` for cross-workspace model versions (#3681, @sueann)
- Remove hard dependency on matplotlib for sklearn autologging (#3703, @dbczumar)
- Do not disable existing loggers when initializing alembic (#3653, @arthury1n)

1.12.0 (2020-11-10)
-------------------
MLflow 1.12.0 includes several major features and improvements, in particular a number of improvements to autologging and MLflow's Pytorch integrations:

Features:
~~~~~~~~~

Autologging:

- Add universal ``mlflow.autolog`` which enables autologging for all supported integrations (#3561, #3590, @andrewnitu)
- Add ``mlflow.pytorch.autolog`` API for automatic logging of metrics, params, and models from Pytorch Lightning training (#3601, @shrinath-suresh, #3636, @karthik-77). This API is also enabled by ``mlflow.autolog``.
- Scikit-learn, XGBoost, and LightGBM autologging now support logging model signatures and input examples (#3386, #3403, #3449, @andrewnitu)
- ``mlflow.sklearn.autolog`` now supports logging metrics (e.g. accuracy) and plots (e.g. confusion matrix heat map) (#3423, #3327, @willzhan-db, @harupy)

PyTorch:

- ``mlflow.pytorch.log_model``, ``mlflow.pytorch.load_model`` now support logging/loading TorchScript models (#3557, @shrinath-suresh) 
- ``mlflow.pytorch.log_model`` supports passing ``requirements_file`` & ``extra_files`` arguments to log additional artifacts along with a model (#3436, @shrinath-suresh)


More features and improvements:

- Add ``mlflow.shap.log_explanation`` for logging model explanations generated by SHAP (#3513, @harupy)
- ``log_model`` and ``create_model_version`` now supports an ``await_creation_for`` argument (#3376, @andychow-db)
- Put preview paths before non-preview paths for backwards compatibility (#3648, @sueann)
- Clean up model registry endpoint and client method definitions (#3610, @sueann)
- MLflow deployments plugin now supports 'predict' CLI command (#3597, @shrinath-suresh)
- Support H2O for R (#3416, @yitao-li)
- Add ``MLFLOW_S3_IGNORE_TLS`` environment variable to enable skipping TLS verification of S3 endpoint (#3345, @dolfinus)

Bug fixes and documentation updates:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Ensure that results are synced across distributed processes if ddp enabled (no-op else) (#3651, @SeanNaren)
- Remove optimizer step override to ensure that all accelerator cases are covered by base module (#3635, @SeanNaren)
- Fix ``AttributeError`` in keras autologgging (#3611, @sephib)
- Scikit-learn autologging: Exclude feature extraction / selection estimator (#3600, @dbczumar)
- Scikit-learn autologging: Fix behavior when a child and its parent are both patched (#3582, @dbczumar)
- Fix a bug where ``lightgbm.Dataset(None)`` fails after running ``mlflow.lightgbm.autolog`` (#3594, @harupy)
- Fix a bug where ``xgboost.DMatrix(None)`` fails after running ``mlflow.xgboost.autolog`` (#3584, @harupy)
- Pass ``docker_args`` in non-synchronous mlflow project runs (#3563, @alfozan)
- Fix a bug of ``FTPArtifactRepository.log_artifacts`` with ``artifact_path`` keyword argument (issue #3388) (#3391, @kzm4269)
- Exclude preprocessing & imputation steps from scikit-learn autologging (#3491, @dbczumar)
- Fix duplicate stderr logging during artifact logging and project execution in the R client (#3145, @yitao-li)
- Don't call ``atexit.register(_flush_queue)`` in ``__main__`` scope of ``mlflow/tensorflow.py`` (#3410, @harupy)
- Fix for restarting terminated run not setting status correctly (#3329, @apurva-koti)
- Fix model version run_link URL for some Databricks regions (#3417, @sueann)
- Skip JSON validation when endpoint is not MLflow REST API (#3405, @harupy)
- Document ``mlflow-torchserve`` plugin (#3634, @karthik-77)
- Add ``mlflow-elasticsearchstore`` to the doc (#3462, @AxelVivien25)
- Add code snippets for fluent and MlflowClient APIs (#3385, #3437, #3489 #3573, @dmatrix)
- Document ``mlflow-yarn`` backend (#3373, @fhoering)
- Fix a breakage in loading Tensorflow and Keras models (#3667, @tomasatdatabricks)

Small bug fixes and doc updates (#3607, #3616, #3534, #3598, #3542, #3568, #3349, #3554, #3544, #3541, #3533, #3535, #3516, #3512, #3497, #3522, #3521, #3492, #3502, #3434, #3422, #3394, #3387, #3294, #3324, #3654, @harupy; #3451, @jgc128; #3638, #3632, #3608, #3452, #3399, @shrinath-suresh; #3495, #3459, #3662, #3668, #3670 @smurching; #3488, @edgan8; #3639, @karthik-77; #3589, #3444, #3276, @lorenzwalthert; #3538, #3506, #3509, #3507, #3510, #3508, @rahulporuri; #3504, @sbrugman; #3486, #3466, @apurva-koti; #3477, @juntai-zheng; #3617, #3609, #3605, #3603, #3560, @dbczumar; #3411, @danielvdende; #3377, @willzhan-db; #3420, #3404, @andrewnitu; #3591, @mateiz; #3465, @abawchen; #3543, @emptalk; #3302, @bramrodenburg; #3468, @ghisvail; #3496, @extrospective; #3549, #3501, #3435, @yitao-li; #3243, @OlivierBondu; #3439, @andrewnitu; #3651, #3635 @SeanNaren, #3470, @ankit-db)

1.11.0 (2020-08-31)
-------------------
MLflow 1.11.0 includes several major features and improvements:

Features:

- New ``mlflow.sklearn.autolog()`` API for automatic logging of metrics, params, and models from scikit-learn model training (#3287, @harupy; #3323, #3358 @dbczumar)
- Registered model & model version creation APIs now support specifying an initial ``description`` (#3271, @sueann)
- The R ``mlflow_log_model`` and ``mlflow_load_model`` APIs now support XGBoost models (#3085, @lorenzwalthert)
- New ``mlflow.list_run_infos`` fluent API for listing run metadata (#3183, @trangevi)
- Added section for visualizing and comparing model schemas to model version and model-version-comparison UIs (#3209, @zhidongqu-db)
- Enhanced support for using the model registry across Databricks workspaces: support for registering models to a Databricks workspace from outside the workspace (#3119, @sueann), tracking run-lineage of these models (#3128, #3164, @ankitmathur-db; #3187, @harupy), and calling ``mlflow.<flavor>.load_model`` against remote Databricks model registries (#3330, @sueann)
- UI support for setting/deleting registered model and model version tags (#3187, @harupy)
- UI support for archiving existing staging/production versions of a model when transitioning a new model version to staging/production (#3134, @harupy)

Bug fixes and documentation updates:

- Fixed parsing of MLflow project parameter values containing'=' (#3347, @dbczumar)
- Fixed a bug preventing listing of WASBS artifacts on the latest version of Azure Blob Storage (12.4.0) (#3348, @dbczumar)
- Fixed a bug where artifact locations become malformed when using an SFTP file store in Windows (#3168, @harupy)
- Fixed bug where ``list_artifacts`` returned incorrect results on GCS, preventing e.g. loading SparkML models from GCS (#3242, @santosh1994)
- Writing and reading artifacts via ``MlflowClient`` to a DBFS location in a Databricks tracking server specified through the ``tracking_uri`` parameter during the initialization of ``MlflowClient`` now works properly (#3220, @sueann)
- Fixed bug where ``FTPArtifactRepository`` returned artifact locations as absolute paths, rather than paths relative to the artifact repository root (#3210, @shaneing), and bug where calling `log_artifacts` against an FTP artifact location copied the logged directory itself into the FTP location, rather than the contents of the directory.
- Fixed bug where Databricks project execution failed due to passing of GET request params as part of the request body rather than as query parameters (#2947, @cdemonchy-pro)
- Fix bug where artifact viewer did not correctly render PDFs in MLflow 1.10 (#3172, @ankitmathur-db)
- Fixed parsing of ``order_by`` arguments to MLflow search APIs when ordering by fields whose names contain spaces (#3118, @jdlesage)
- Fixed bug where MLflow model schema enforcement raised exceptions when validating string columns using pandas >= 1.0 (#3130, @harupy)
- Fixed bug where ``mlflow.spark.log_model`` did not save model signature and input examples (#3151, @harupy)
- Fixed bug in runs UI where tags table did not reflect deletion of tags. (#3135, @ParseDark)
- Added example illustrating the use of RAPIDS with MLFlow (#3028, @drobison00)

Small bug fixes and doc updates (#3326, #3344, #3314, #3289, #3225, #3288, #3279, #3265, #3263, #3260, #3255, #3267, #3266, #3264, #3256, #3253, #3231, #3245, #3191, #3238, #3192, #3188, #3189, #3180, #3178, #3166, #3181, #3142, #3165, #2960, #3129, #3244, #3359 @harupy; #3236, #3141, @AveshCSingh; #3295, #3163, @arjundc-db; #3241, #3200, @zhidongqu-db; #3338, #3275, @sueann; #3020, @magnus-m; #3322, #3219, @dmatrix; #3341, #3179, #3355, #3360, #3363 @smurching; #3124, @jdlesage; #3232, #3146, @ankitmathur-db; #3140, @andreakress; #3174, #3133, @mlflow-automation; #3062, @cafeal; #3193, @tomasatdatabricks; 3115, @fhoering; #3328, @apurva-koti; #3046, @OlivierBondu; #3194, #3158, @dmatrix; #3250, @shivp950; #3259, @simonhessner; #3357 @dbczumar)

1.10.0 (2020-07-20)
-------------------
MLflow 1.10.0 includes several major features and improvements, in particular the release of
several new model registry Python client APIs.

Features:

- ``MlflowClient.transition_model_version_stage`` now supports an
  ``archive_existing_versions`` argument for archiving existing staging or production model
  versions when transitioning a new model version to staging or production (#3095, @harupy)
- Added ``set_registry_uri``, ``get_registry_uri`` APIs. Setting the model registry URI causes
  fluent APIs like ``mlflow.register_model`` to communicate with the model registry at the specified
  URI (#3072, @sueann)
- Added paginated ``MlflowClient.search_registered_models`` API (#2939, #3023, #3027 @ankitmathur-db; #2966, @mparkhe)
- Added syntax highlighting when viewing text files (YAML etc) in the MLflow runs UI (#3041, @harupy)
- Added REST API and Python client support for setting and deleting tags on model versions and registered models,
  via the ``MlflowClient.create_registered_model``,  ``MlflowClient.create_model_version``,
  ``MlflowClient.set_registered_model_tag``, ``MlflowClient.set_model_version_tag``,
  ``MlflowClient.delete_registered_model_tag``, and ``MlflowClient.delete_model_version_tag`` APIs (#3094, @zhidongqu-db)

Bug fixes and documentation updates:

- Removed usage of deprecated ``aws ecr get-login`` command in ``mlflow.sagemaker`` (#3036, @mrugeles)
- Fixed bug where artifacts could not be viewed and downloaded from the artifact UI when using
  Azure Blob Storage (#3014, @Trollgeir)
- Databricks credentials are now propagated to the project subprocess when running MLflow projects
  within a notebook (#3035, @smurching)
- Added docs explaining how to fetching an MLflow model from the model registry (#3000, @andychow-db)

Small bug fixes and doc updates (#3112, #3102, #3089, #3103, #3096, #3090, #3049, #3080, #3070, #3078, #3083, #3051, #3050, #2875, #2982, #2949, #3121 @harupy; #3082, @ankitmathur-db; #3084, #3019, @smurching)

1.9.1 (2020-06-25)
------------------
MLflow 1.9.1 is a patch release containing a number of bug-fixes and improvements:

Bug fixes and improvements:

* Fixes ``AttributeError`` when pickling an instance of the Python ``MlflowClient`` class (#2955, @Polyphenolx)
* Fixes bug that prevented updating model-version descriptions in the model registry UI (#2969, @AnastasiaKol) 
* Fixes bug where credentials were not properly propagated to artifact CLI commands when logging artifacts from Java to the DatabricksArtifactRepository (#3001, @dbczumar)
* Removes use of new Pandas API in new MLflow model-schema functionality, so that it can be used with older Pandas versions (#2988, @aarondav)

Small bug fixes and doc updates (#2998, @dbczumar; #2999, @arjundc-db)

1.9.0 (2020-06-19)
------------------
MLflow 1.9.0 includes numerous major features and improvements, and a breaking change to
experimental APIs:

Breaking Changes:

- The ``new_name`` argument to ``MlflowClient.update_registered_model``
  has been removed. Call ``MlflowClient.rename_registered_model`` instead. (#2946, @mparkhe)
- The ``stage`` argument to ``MlflowClient.update_model_version``
  has been removed. Call ``MlflowClient.transition_model_version_stage`` instead. (#2946, @mparkhe)

Features (MLflow Models and Flavors)

- ``log_model`` and ``save_model`` APIs now support saving model signatures (the model's input and output schema)
  and example input along with the model itself  (#2698, #2775, @tomasatdatabricks). Model signatures are used
  to reorder and validate input fields when scoring/serving models using the pyfunc flavor, ``mlflow models``
  CLI commands, or ``mlflow.pyfunc.spark_udf`` (#2920, @tomasatdatabricks and @aarondav)
- Introduce fastai model persistence and autologging APIs under ``mlflow.fastai`` (#2619, #2689 @antoniomdk)
- Add pluggable ``mlflow.deployments`` API and CLI for deploying models to custom serving tools, e.g. RedisAI
  (#2327, @hhsecond)
- Enables loading and scoring models whose conda environments include dependencies in conda-forge (#2797, @dbczumar)
- Add support for scoring ONNX-persisted models that return Python lists (#2742, @andychow-db)

Features (MLflow Projects)

- Add plugin interface for executing MLflow projects against custom backends (#2566, @jdlesage)
- Add ability to specify additional cluster-wide Python and Java libraries when executing
  MLflow projects remotely on Databricks (#2845, @pogil)
- Allow running MLflow projects against remote artifacts stored in any location with a corresponding
  ArtifactRepository implementation (Azure Blob Storage, GCS, etc) (#2774, @trangevi)
- Allow MLflow projects running on Kubernetes to specify a different tracking server to log to via the
  ``KUBE_MLFLOW_TRACKING_URI`` for passing a different tracking server to the kubernetes job (#2874, @catapulta)

Features (UI)

- Significant performance and scalability improvements to metric comparison and scatter plots in
  the UI (#2447, @mjlbach)
- The main MLflow experiment list UI now includes a link to the model registry UI (#2805, @zhidongqu-db),
- Enable viewing PDFs logged as artifacts from the runs UI  (#2859, @ankmathur96)
- UI accessibility improvements: better color contrast (#2872, @Zangr), add child roles to DOM elements (#2871, @Zangr)

Features (Tracking Client and Server)

- Adds ability to pass client certs as part of REST API requests when using the tracking or model
  registry APIs. (#2843, @PhilipMay)
- New community plugin: support for storing artifacts in Aliyun (Alibaba Cloud) (#2917, @SeaOfOcean)
- Infer and set content type and encoding of objects when logging models and artifacts to S3 (#2881, @hajapy)
- Adds support for logging artifacts to HDFS Federation ViewFs (#2782, @fhoering)
- Add healthcheck endpoint to the MLflow server at ``/health`` (#2725, @crflynn)
- Improves performance of default file-based tracking storage backend by using LibYAML (if installed)
  to read experiment and run metadata (#2707, @Higgcz)


Bug fixes and documentation updates:

- Several UI fixes: remove margins around icon buttons (#2827, @harupy),
  fix alignment issues in metric view (#2811, @zhidongqu-db), add handling of ``NaN``
  values in metrics plot (#2773, @dbczumar), truncate run ID in the run name when
  comparing multiple runs (#2508, @harupy)
- Database engine URLs are no longer logged when running ``mlflow db upgrade`` (#2849, @hajapy)
- Updates ``log_artifact``, ``log_model`` APIs to consistently use posix paths, rather than OS-dependent
  paths, when computing  artifact subpaths. (#2784, @mikeoconnor0308)
- Fix ``ValueError`` when scoring ``tf.keras`` 1.X models using ``mlflow.pyfunc.predict`` (#2762, @juntai-zheng)
- Fixes conda environment activation bug when running MLflow projects on Windows (#2731, @MynherVanKoek)
- ``mlflow.end_run`` will now clear the active run even if the run cannot be marked as
  terminated (e.g. because it's been deleted), (#2693, @ahmed-shariff)
- Add missing documentation for ``mlflow.spacy`` APIs (#2771, @harupy)


Small bug fixes and doc updates (#2919, @willzhan-db; #2940, #2942, #2941, #2943, #2927, #2929, #2926, #2914, #2928, #2913, #2852, #2876, #2808, #2810, #2442, #2780, #2758, #2732, #2734, #2431, #2733, #2716, @harupy; #2915, #2897, @jwgwalton; #2856, @jkthompson; #2962, @hhsecond; #2873, #2829, #2582, @dmatrix; #2908, #2865, #2880, #2866, #2833, #2785, #2723, @smurching; #2906, @dependabot[bot]; #2724, @aarondav; #2896, @ezeeetm; #2741, #2721, @mlflow-automation; #2864, @tallen94; #2726, @crflynn; #2710, #2951 @mparkhe; #2935, #2921, @ankitmathur-db; #2963, #2739, @dbczumar; #2853, @stat4jason; #2709, #2792, @juntai-zheng @juntai-zheng; #2749, @HiromuHota; #2957, #2911, #2718, @arjundc-db; #2885, @willzhan-db; #2803, #2761, @pogil; #2392, @jnmclarty; #2794, @Zethson; #2766, #2916 @shubham769)

1.8.0 (2020-04-16)
------------------
MLflow 1.8.0 includes several major features and improvements:

Features:

- Added ``mlflow.azureml.deploy`` API for deploying MLflow models to AzureML (#2375 @csteegz, #2711, @akshaya-a)
- Added support for case-sensitive LIKE and case-insensitive ILIKE queries (e.g. ``'params.framework LIKE '%sklearn%'``) with the SearchRuns API & UI when running against a SQLite backend (#2217, @t-henri; #2708, @mparkhe)
- Improved line smoothing in MLflow metrics UI using exponential moving averages (#2620, @Valentyn1997)
- Added ``mlflow.spacy`` module with support for logging and loading spaCy models (#2242, @arocketman)
- Parameter values that differ across runs are highlighted in run comparison UI (#2565, @gabrielbretschner)
- Added ability to compare source runs associated with model versions from the registered model UI  (#2537, @juntai-zheng)
- Added support for alphanumerical experiment IDs in the UI. (#2568, @jonas)
- Added support for passing arguments to ``docker run`` when running docker-based MLflow projects (#2608, @ksanjeevan)
- Added Windows support for ``mlflow sagemaker build-and-push-container`` CLI & API (#2500, @AndreyBulezyuk)
- Improved performance of reading experiment data from local filesystem when LibYAML is installed (#2707, @Higgcz)
- Added a healthcheck endpoint to the REST API server at ``/health`` that always returns a 200 response status code, to be used to verify health of the server (#2725, @crflynn)
- MLflow metrics UI plots now scale to rendering thousands of points using scattergl (#2447, @mjlbach)

Bug fixes:

- Fixed CLI summary message in ``mlflow azureml build_image`` CLI (#2712, @dbczumar)
- Updated ``examples/flower_classifier/score_images_rest.py`` with multiple bug fixes (#2647, @tfurmston)
- Fixed pip not found error while packaging models via ``mlflow models build-docker`` (#2699, @HiromuHota)
- Fixed bug in ``mlflow.tensorflow.autolog`` causing erroneous deletion of TensorBoard logging directory (#2670, @dbczumar)
- Fixed a bug that truncated the description of the ``mlflow gc`` subcommand in ``mlflow --help`` (#2679, @dbczumar)
- Fixed bug where ``mlflow models build-docker`` was failing due to incorrect Miniconda download URL (#2685, @michaeltinsley)
- Fixed a bug in S3 artifact logging functionality where ``MLFLOW_S3_ENDPOINT_URL`` was ignored (#2629, @poppash)
- Fixed a bug where Sqlite in-memory was not working as a tracking backend store by modifying DB upgrade logic (#2667, @dbczumar)
- Fixed a bug to allow numerical parameters with values >= 1000 in R ``mlflow::mlflow_run()`` API (#2665, @lorenzwalthert)
- Fixed a bug where AWS creds was not found in the Windows platform due path differences (#2634, @AndreyBulezyuk)
- Fixed a bug to add pip when necessary in ``_mlflow_conda_env`` (#2646, @tfurmston)
- Fixed error code to be more meaningful if input to model version is incorrect (#2625, @andychow-db)
- Fixed multiple bugs in model registry (#2638, @aarondav)
- Fixed support for conda env dicts with ``mlflow.pyfunc.log_model`` (#2618, @dbczumar)
- Fixed a bug where hiding the start time column in the UI would also hide run selection checkboxes (#2559, @harupy)

Documentation updates:

- Added links to source code to mlflow.org (#2627, @harupy)
- Documented fix for pandas-records payload (#2660, @SaiKiranBurle)
- Fixed documentation bug in TensorFlow ``load_model`` utility (#2666, @pogil)
- Added the missing Model Registry description and link on the first page (#2536, @dmatrix)
- Added documentation for expected datatype for step argument in ``log_metric`` to match REST API (#2654, @mparkhe)
- Added usage of the model registry to the ``log_model`` function in ``sklearn_elasticnet_wine/train.py`` example (#2609, @netanel246)

Small bug fixes and doc updates (#2594, @Trollgeir; #2703,#2709, @juntai-zheng; #2538, #2632, @keigohtr; #2656, #2553, @lorenzwalthert; #2622, @pingsutw; #2615, #2600, #2533, @mlflow-automation; #1391, @sueann; #2613, #2598, #2534, #2723, @smurching; #2652, #2710, @mparkhe; #2706, #2653, #2639, @tomasatdatabricks; #2611, @9dogs; #2700, #2705, @aarondav; #2675, #2540, @mengxr; #2686, @RensDimmendaal; #2694, #2695, #2532, @dbczumar; #2733, #2716, @harupy; #2726, @crflynn; #2582, #2687, @dmatrix)


1.7.2 (2020-03-20)
------------------------
MLflow 1.7.2 is a patch release containing a minor change:

- Pin alembic version to 1.4.1 or below to prevent pep517-related installation errors
  (#2612, @smurching)


1.7.1 (2020-03-17)
------------------------
MLflow 1.7.1 is a patch release containing bug fixes and small changes:

- Remove usage of Nonnull annotations and findbugs dependency in Java package (#2583, @mparkhe)
- Add version upper bound (<=1.3.13) to sqlalchemy dependency in Python package (#2587, @smurching)

Other bugfixes and doc updates (#2595, @mparkhe; #2567, @jdlesage)

1.7.0 (2020-03-02)
------------------
MLflow 1.7.0 includes several major features and improvements, and some notable breaking changes:

MLflow support for Python 2 is now deprecated and will be dropped in a future release. At that
point, existing Python 2 workflows that use MLflow will continue to work without modification, but
Python 2 users will no longer get access to the latest MLflow features and bugfixes. We recommend
that you upgrade to Python 3 - see  https://docs.python.org/3/howto/pyporting.html for a migration
guide.

Breaking changes to Model Registry REST APIs:

Model Registry REST APIs have been updated to be more consistent with the other MLflow APIs. With
this release Model Registry APIs are intended to be stable until the next major version.

- Python and Java client APIs for Model Registry have been updated to use the new REST APIs. When using an MLflow client with a server using updated REST endpoints, you won't need to change any code but will need to upgrade to a new client version. The client APIs contain deprecated arguments, which for this release are backward compatible, but will be dropped in future releases. (#2457, @tomasatdatabricks; #2502, @mparkhe).
- The Model Registry UI has been updated to use the new REST APIs (#2476 @aarondav; #2507, @mparkhe)


Other Features:

- Ability to click through to individual runs from metrics plot (#2295, @harupy)
- Added ``mlflow gc`` CLI for permanent deletion of runs (#2265, @t-henri)
- Metric plot state is now captured in page URLs for easier link sharing (#2393, #2408, #2498 @smurching; #2459, @harupy)
- Added experiment management to MLflow UI (create/rename/delete experiments) (#2348, @ggliem)
- Ability to search for experiments by name in the UI (#2324, @ggliem)
- MLflow UI page titles now reflect the content displayed on the page (#2420, @AveshCSingh)
- Added a new ``LogModel`` REST API endpoint for capturing model metadata, and call it from the Python and R clients (#2369, #2430, #2468 @tomasatdatabricks)
- Java Client API to download model artifacts from Model Registry (#2308, @andychow-db)

Bug fixes and documentation updates:

- Updated Model Registry documentation page with code snippets and examples (#2493, @dmatrix; #2517, @harupy)
- Better error message for Model Registry, when using incompatible backend server (#2456, @aarondav)
- matplotlib is no longer required to use XGBoost and LightGBM autologging (#2423, @harupy)
- Fixed bug where matplotlib figures were not closed in XGBoost and LightGBM autologging (#2386, @harupy)
- Fixed parameter reading logic to support param values with newlines in FileStore (#2376, @dbczumar)
- Improve readability of run table column selector nodes (#2388, @dbczumar)
- Validate experiment name supplied to ``UpdateExperiment`` REST API endpoint (#2357, @ggliem)
- Fixed broken MLflow DB README link in CLI docs (#2377, @dbczumar)
- Change copyright year across docs to 2020 (#2349, @ParseThis)

Small bug fixes and doc updates (#2378, #2449, #2402, #2397, #2391, #2387, #2523, #2527 @harupy; #2314, @juntai-zheng; #2404, @andychow-db; #2343, @pogil; #2366, #2370, #2364, #2356, @AveshCSingh; #2373, #2365, #2363, @smurching; #2358, @jcuquemelle; #2490, @RensDimmendaal; #2506, @dbczumar; #2234 @Zangr; #2359 @lbernickm; #2525, @mparkhe)

1.6.0 (2020-01-29)
-----------------------
MLflow 1.6.0 includes several new features, including a better runs table interface, a utility for easier parameter tuning, and automatic logging from XGBoost, LightGBM, and Spark. It also implements a long-awaited fix allowing @ symbols in database URLs. A complete list is below:

Features:

- Adds a new runs table column view based on `ag-grid` which adds functionality for nested runs, serverside sorting, column reordering, highlighting, and more. (#2251, @Zangr)
- Adds contour plot to the run comparsion page to better support parameter tuning (#2225, @harupy)
- If you use EarlyStopping with Keras autologging, MLflow now automatically captures the best model trained and the associated metrics (#2301, #2219, @juntai-zheng)
- Adds autologging functionality for LightGBM and XGBoost flavors to log feature importance, metrics per iteration, the trained model, and more. (#2275, #2238, @harupy) 
- Adds an experimental mlflow.spark.autolog() API for automatic logging of Spark datasource information to the current active run. (#2220, @smurching)
- Optimizes the file store to load less data from disk for each operation (#2339, @jonas)
- Upgrades from ubuntu:16.04 to ubuntu:18.04 when building a Docker image with `mlflow models build-docker` (#2256, @andychow-db)

Bug fixes and documentation updates:

- Fixes bug when running server against database URLs with @ symbols (#2289, @hershaw)
- Fixes model Docker image build on Windows (#2257, @jahas)
- Documents the SQL Server plugin (#2320, @avflor)
- Adds a help file for the R package (#2259, @lorenzwalthert)
- Adds an example of using the Search API to find the best performing model (#2313, @AveshCSingh)
- Documents how to write and use MLflow plugins (#2270, @smurching)

Small bug fixes and doc updates (#2293, #2328, #2244, @harupy; #2269, #2332, #2306, #2307, #2292, #2267, #2191, #2231, @juntai-zheng; #2325, @shubham769; #2291, @sueann; #2315, #2249, #2288, #2278, #2253, #2181, @smurching; #2342, @tomasatdatabricks; #2245, @dependabot[bot]; #2338, @jcuquemelle; #2285, @avflor; #2340, @pogil; #2237, #2226, #2243, #2272, #2286, @dbczumar; #2281, @renaudhager; #2246, @avaucher; #2258, @lorenzwalthert; #2261, @smith-kyle; 2352, @dbczumar)

1.5.0 (2019-12-19)
-----------------------
MLflow 1.5.0 includes several major features and improvements:

New Model Flavors and Flavor Updates:

- New support for a LightGBM flavor (#2136, @harupy)
- New support for a XGBoost flavor (#2124, @harupy)
- New support for a Gluon flavor and autologging (#1973, @cosmincatalin)
- Runs automatically created by ``mlflow.tensorflow.autolog()`` and ``mlflow.keras.autolog()`` (#2088) are now automatically ended after training and/or exporting your model. See the `docs <https://mlflow.org/docs/latest/tracking.html#automatic-logging-from-tensorflow-and-keras-experimental>`_ for more details (#2094, @juntai-zheng)

More features and improvements:

- When using the ``mlflow server`` CLI command, you can now expose metrics on ``/metrics`` for Prometheus via the optional --activate-parameter argument (#2097, @t-henri)
- The ``mlflow ui`` CLI command now has a ``--host``/``-h`` option to specify user-input IPs to bind to (#2176, @gandroz)
- MLflow now supports pulling Git submodules while using MLflow Projects (#2103, @badc0re)
- New ``mlflow models prepare-env`` command to do any preparation necessary to initialize an environment. This allows distinguishing configuration and user errors during predict/serve time (#2040, @aarondav)
- TensorFlow.Keras and Keras parameters are now logged by ``autolog()`` (#2119, @juntai-zheng)
- MLflow ``log_params()`` will recognize Spark ML params as keys and will now extract only the name attribute (#2064, @tomasatdatabricks)
- Exposes ``mlflow.tracking.is_tracking_uri_set()`` (#2026, @fhoering)
- The artifact image viewer now displays "Loading..." when it is loading an image (#1958, @harupy)
- The artifact image view now supports animated GIFs (#2070, @harupy)
- Adds ability to mount volumes and specify environment variables when using mlflow with docker (#1994, @nlml)
- Adds run context for detecting job information when using MLflow tracking APIs within Databricks Jobs. The following job types are supported: notebook jobs, Python Task jobs (#2205, @dbczumar)
- Performance improvement when searching for runs (#2030, #2059, @jcuquemelle; #2195, @rom1504)

Bug fixes and documentation updates:

- Fixed handling of empty directories in FS based artifact repositories (#1891, @tomasatdatabricks)
- Fixed ``mlflow.keras.save_model()`` usage with DBFS (#2216, @andychow-db)
- Fixed several build issues for the Docker image (#2107, @jimthompson5802)
- Fixed ``mlflow_list_artifacts()`` (R package) (#2200, @lorenzwalthert)
- Entrypoint commands of Kubernetes jobs are now shell-escaped (#2160, @zanitete)
- Fixed project run Conda path issue (#2147, @Zangr)
- Fixed spark model load from model repository (#2175, @tomasatdatabricks)
- Stripped "dev" suffix from PySpark versions (#2137, @dbczumar)
- Fixed note editor on the experiment page (#2054, @harupy)
- Fixed ``models serve``, ``models predict`` CLI commands against models:/ URIs (#2067, @smurching)
- Don't unconditionally format values as metrics in generic HtmlTableView component (#2068, @smurching)
- Fixed remote execution from Windows using posixpath (#1996, @aestene)
- Add XGBoost and LightGBM examples (#2186, @harupy)
- Add note about active run instantiation side effect in fluent APIs (#2197, @andychow-db)
- The tutorial page has been refactored to be be a 'Tutorials and Examples' page (#2182, @juntai-zheng)
- Doc enhancements for XGBoost and LightGBM flavors (#2170, @harupy)
- Add doc for XGBoost flavor (#2167, @harupy)
- Updated ``active_run()`` docs to clarify it cannot be used accessing current run data (#2138, @juntai-zheng)
- Document models:/ scheme for URI for load_model methods (#2128, @stbof)
- Added an example using Prophet via pyfunc (#2043, @dr3s)
- Added and updated some screenshots and explicit steps for the model registry (#2086, @stbof)

Small bug fixes and doc updates (#2142, #2121, #2105, #2069, #2083, #2061, #2022, #2036, #1972, #2034, #1998, #1959, @harupy; #2202, @t-henri; #2085, @stbof; #2098, @AdamBarnhard; #2180, #2109, #1977, #2039, #2062, @smurching; #2013, @aestene; #2146, @joelcthomas; #2161, #2120, #2100, #2095, #2088, #2076, #2057, @juntai-zheng; #2077, #2058, #2027, @sueann; #2149, @zanitete; #2204, #2188, @andychow-db; #2110, #2053, @jdlesage; #2003, #1953, #2004, @Djailla; #2074, @nlml; #2116, @Silas-Asamoah; #1104, @jimthompson5802; #2072, @cclauss; #2221, #2207, #2157, #2132, #2114, #2063, #2065, #2055, @dbczumar; #2033, @cthoyt; #2048, @philip-khor; #2002, @jspoorta; #2000, @christang; #2078, @dennyglee; #1986, @vguerra; #2020, @dependabot[bot])

1.4.0 (2019-10-30)
-----------------------
MLflow 1.4.0 includes several major features:

- Model Registry (Beta). Adds an experimental model registry feature, where you can manage, version, and keep lineage of your production models. (#1943, @mparkhe, @Zangr, @sueann, @dbczumar, @smurching, @gioa, @clemens-db, @pogil, @mateiz; #1988, #1989, #1995, #2021, @mparkhe; #1983, #1982, #1967, @dbczumar)
- TensorFlow updates 

  - MLflow Keras model saving, loading, and logging has been updated to be compatible with TensorFlow 2.0.  (#1927, @juntai-zheng)
  - Autologging for ``tf.estimator`` and ``tf.keras`` models has been updated to be compatible with TensorFlow 2.0. The same functionalities of autologging in TensorFlow 1.x are available in TensorFlow 2.0, namely when fitting ``tf.keras`` models and when exporting saved ``tf.estimator`` models. (#1910, @juntai-zheng)
  - Examples and READMEs for both TensorFlow 1.X and TensorFlow 2.0 have been added to ``mlflow/examples/tensorflow``. (#1946, @juntai-zheng)

More features and improvements:

- [API] Add functions ``get_run``, ``get_experiment``, ``get_experiment_by_name`` to the fluent API (#1923, @fhoering)
- [UI] Use Plotly as artifact image viewer, which allows zooming and panning (#1934, @harupy)
- [UI] Support deleting tags from the run details page (#1933, @harupy)
- [UI] Enable scrolling to zoom in metric and run comparison plots (#1929, @harupy)
- [Artifacts] Add support of viewfs URIs for HDFS federation for artifacts (#1947, @t-henri)
- [Models] Spark UDFs can now be called with struct input if the underlying spark implementation supports it. The data is passed as a pandas DataFrame with column names matching those in the struct. (#1882, @tomasatdatabricks)
- [Models] Spark models will now load faster from DFS by skipping unnecessary copies (#2008, @tomasatdatabricks)

Bug fixes and documentation updates:

- [Projects] Make detection of ``MLproject`` files case-insensitive (#1981, @smurching)
- [UI] Fix a bug where viewing metrics containing forward-slashes in the name would break the MLflow UI (#1968, @smurching)
- [CLI] ``models serve`` command now works in Windows (#1949, @rboyes)
- [Scoring] Fix a dependency installation bug in Java MLflow model scoring server (#1913, @smurching)

Small bug fixes and doc updates (#1932, #1935, @harupy; #1907, @marnixkoops; #1911, @HackyRoot; #1931, @jmcarp; #2007, @deniskovalenko; #1966, #1955, #1952, @Djailla; #1915, @sueann; #1978, #1894, @smurching; #1940, #1900, #1904, @mparkhe; #1914, @jerrygb; #1857, @mengxr; #2009, @dbczumar)


1.3 (2019-09-30)
------------------
MLflow 1.3.0 includes several major features and improvements:

Features:

- The Python client now supports logging & loading models using TensorFlow 2.0 (#1872, @juntai-zheng)
- Significant performance improvements when fetching runs and experiments in MLflow servers that use SQL database-backed storage (#1767, #1878, #1805 @dbczumar)
- New ``GetExperimentByName`` REST API endpoint, used in the Python client to speed up ``set_experiment`` and ``get_experiment_by_name`` (#1775, @smurching)
- New ``mlflow.delete_run``, ``mlflow.delete_experiment`` fluent APIs in the Python client(#1396, @MerelTheisenQB)
- New CLI command (``mlflow experiments csv``) to export runs of an experiment into a CSV (#1705, @jdlesage)
- Directories can now be logged as artifacts via ``mlflow.log_artifact`` in the Python fluent API (#1697, @apurva-koti)
- HTML and geojson artifacts are now rendered in the run UI (#1838, @sim-san; #1803, @spadarian)
- Keras autologging support for ``fit_generator`` Keras API (#1757, @charnger)
- MLflow models packaged as docker containers can be executed via Google Cloud Run (#1778, @ngallot)
- Artifact storage configurations are propagated to containers when executing docker-based MLflow projects locally (#1621, @nlaille)
- The Python, Java, R clients and UI now retry HTTP requests on 429 (Too Many Requests) errors (#1846, #1851, #1858, #1859 @tomasatdatabricks; #1847, @smurching)


Bug fixes and documentation updates:

- The R ``mlflow_list_artifact`` API no longer throws when listing artifacts for an empty run (#1862, @smurching)
- Fixed a bug preventing running the MLflow server against an MS SQL database (#1758, @sifanLV)
- MLmodel files (artifacts) now correctly display in the run UI (#1819, @ankitmathur-db)
- The Python ``mlflow.start_run`` API now throws when resuming a run whose experiment ID differs from the
  active experiment ID set via ``mlflow.set_experiment`` (#1820, @mcminnra).
- ``MlflowClient.log_metric`` now logs metric timestamps with millisecond (as opposed to second) resolution (#1804, @ustcscgyer)
- Fixed bugs when listing (#1800, @ahutterTA) and downloading (#1890, @jdlesage) artifacts stored in HDFS.
- Fixed a bug preventing Kubernetes Projects from pushing to private Docker repositories (#1788, @dbczumar)
- Fixed a bug preventing deploying Spark models to AzureML (#1769, @Ben-Epstein)
- Fixed experiment id resolution in projects (#1715, @drewmcdonald)
- Updated parallel coordinates plot to show all fields available in compared runs (#1753, @mateiz)
- Streamlined docs for getting started with hosted MLflow (#1834, #1785, #1860 @smurching)

Small bug fixes and doc updates (#1848, @pingsutw; #1868, @iver56; #1787, @apurvakoti; #1741, #1737, @apurva-koti; #1876, #1861, #1852, #1801, #1754, #1726, #1780, #1807 @smurching; #1859, #1858, #1851, @tomasatdatabricks; #1841, @ankitmathur-db; #1744, #1746, #1751, @mateiz; #1821, #1730, @dbczumar; #1727, cfmcgrady; #1716, @axsaucedo; #1714, @fhoering; #1405, @ancasarb; #1502, @jimthompson5802; #1720, jke-zq; #1871, @mehdi254; #1782, @stbof)


1.2 (2019-08-09)
----------------
MLflow 1.2 includes the following major features and improvements:

- Experiments now have editable tags and descriptions (#1630, #1632, #1678, @ankitmathur-db)
- Search latency has been significantly reduced in the SQLAlchemyStore (#1660, @t-henri)

**More features and improvements**

- Backend stores now support run tag values up to 5000 characters in length. Some store implementations may support longer tag values (#1687, @ankitmathur-db)
- Gunicorn options can now be configured for the ``mlflow models serve`` CLI with the ``GUNICORN_CMD_ARGS`` environment variable (#1557, @LarsDu)
- Jsonnet artifacts can now be previewed in the UI (#1683, @ankitmathur-db)
- Adds an optional ``python_version`` argument to ``mlflow_install`` for specifying the Python version (e.g. "3.5") to use within the conda environment created for installing the MLflow CLI. If ``python_version`` is unspecified, ``mlflow_install`` defaults to using Python 3.6. (#1722, @smurching)


**Bug fixes and documentation updates**

- [Tracking] The Autologging feature is now more resilient to tracking errors (#1690, @apurva-koti)
- [Tracking] The ``runs`` field in in the ``GetExperiment.Response`` proto has been deprecated & will be removed in MLflow 2.0. Please use the ``Search Runs`` API for fetching runs instead (#1647, @dbczumar)
- [Projects] Fixed a bug that prevented docker-based MLflow Projects from logging artifacts to the ``LocalArtifactRepository`` (#1450, @nlaille)
- [Projects] Running MLflow projects with the ``--no-conda`` flag in R no longer requires Anaconda to be installed (#1650, @spadarian)
- [Models/Scoring] Fixed a bug that prevented Spark UDFs from being loaded on Databricks (#1658, @smurching)
- [UI] AJAX requests made by the MLflow Server Frontend now specify correct MIME-Types (#1679, @ynotzort)
- [UI] Previews now render correctly for artifacts with uppercase file extensions (e.g., ``.JSON``, ``.YAML``) (#1664, @ankitmathur-db)
- [UI] Fixed a bug that caused search API errors to surface a Niagara Falls page (#1681, @dbczumar)
- [Installation] MLflow dependencies are now selected properly based on the target installation platform (#1643, @akshaya-a)
- [UI] Fixed a bug where the "load more" button in the experiment view did not appear on browsers in Windows (#1718, @Zangr)


Small bug fixes and doc updates (#1663, #1719, @dbczumar; #1693, @max-allen-db; #1695, #1659, @smurching; #1675, @jdlesage; #1699, @ankitmathur-db; #1696, @aarondav; #1710, #1700, #1656, @apurva-koti)


1.1 (2019-07-22)
----------------
MLflow 1.1 includes several major features and improvements: 

In MLflow Tracking: 

- Experimental support for autologging from Tensorflow and Keras. Using ``mlflow.tensorflow.autolog()`` will enable automatic logging of metrics and optimizer parameters from TensorFlow to MLflow. The feature will work with TensorFlow versions ``1.12 <= v < 2.0``. (#1520, #1601, @apurva-koti)
- Parallel coordinates plot in the MLflow compare run UI. Adds out of the box support for a parallel coordinates plot. The plot allows users to observe relationships between a n-dimensional set of parameters to metrics. It visualizes all runs as lines that are color-coded based on the value of a metric (e.g. accuracy), and shows what parameter values each run took on. (#1497, @Zangr)
- Pandas based search API. Adds the ability to return the results of a search as a pandas dataframe using the new ``mlflow.search_runs`` API. (#1483, #1548, @max-allen-db)
- Java fluent API. Adds a new set of APIs to create and log to MLflow runs. This API contrasts with the existing low level ``MlflowClient`` API which simply wraps the REST APIs. The new fluent API allows you to create and log runs similar to how you would using the Python fluent API. (#1508, @andrewmchen)
- Run tags improvements. Adds the ability to add and edit tags from the run view UI, delete tags from the API, and view tags in the experiment search view. (#1400, #1426, @Zangr; #1548, #1558, @ankitmathur-db)
- Search API improvements. Adds order by and pagination to the search API. Pagination allows you to read a large set of runs in small page sized chunks. This allows clients and backend implementations to handle an unbounded set of runs in a scalable manner. (#1444, @sueann; #1437, #1455, #1482, #1485, #1542, @aarondav; #1567, @max-allen-db; #1217, @mparkhe)
- Windows support for running the MLflow tracking server and UI. (#1080, @akshaya-a)

In MLflow Projects:

- Experimental support to run Docker based MLprojects in Kubernetes. Adds the first fully open source remote execution backend for MLflow projects. With this, you can leverage elastic compute resources managed by kubernetes for their ML training purposes. For example, you can run grid search over a set of hyperparameters by running several instances of an MLproject in parallel. (#1181, @marcusrehm, @tomasatdatabricks, @andrewmchen; #1566, @stbof, @dbczumar; #1574 @dbczumar)


**More features and improvements**

In MLflow Tracking: 

- Paginated “load more” and backend sorting for experiment search view UI. This change allows the UI to scalably display the sorted runs from large experiments. (#1564, @Zangr)
- Search results are encoded in the URL. This allows you to share searches through their URL and to deep link to them. (#1416, @apurva-koti)
- Ability to serve MLflow UI behind ``jupyter-server-proxy`` or outside of the root path ``/``. Previous to MLflow 1.1, the UI could only be hosted on `/` since the Javascript makes requests directly to ``/ajax-api/...``. With this patch, MLflow will make requests to ``ajax-api/...`` or a path relative to where the HTML is being served. (#1413, @xhochy)

In MLflow Models: 

- Update ``mlflow.spark.log_model()`` to accept descendants of pyspark.Model (#1519, @ankitmathur-db)
- Support for saving custom Keras models with ``custom_objects``. This field is semantically equivalent to custom_objects parameter of ``keras.models.load_model()`` function (#1525, @ankitmathur-db)
- New more performant split orient based input format for pyfunc scoring server (#1479, @lennon310)
- Ability to specify gunicorn server options for pyfunc scoring server built with `mlflow models build-docker`. #1428, @lennon310)

**Bug fixes and documentation updates**

- [Tracking] Fix database migration for MySQL. ``mlflow db upgrade`` should now work for MySQL backends. (#1404, @sueann)
- [Tracking] Make CLI ``mlflow server`` and ``mlflow ui`` commands to work with SQLAlchemy URIs that specify a database driver. (#1411, @sueann)
- [Tracking] Fix usability bugs related to FTP artifact repository. (#1398, @kafendt; #1421, @nlaille)
- [Tracking] Return appropriate HTTP status codes for MLflowException (#1434, @max-allen-db)
- [Tracking] Fix sorting by user ID in the experiment search view. (#1401, @andrewmchen)
- [Tracking] Allow calling log_metric with NaNs and infs. (#1573, @tomasatdatabricks)
- [Tracking] Fixes an infinite loop in downloading artifacts logged via dbfs and retrieved via S3. (#1605, @sueann)
- [Projects] Docker projects should preserve directory structure (#1436, @ahutterTA)
- [Projects] Fix conda activation for newer versions of conda. (#1576, @avinashraghuthu, @smurching)
- [Models] Allow you to log Tensorflow keras models from the ``tf.keras`` module. (#1546, @tomasatdatabricks)

Small bug fixes and doc updates (#1463, @mateiz; #1641, #1622, #1418, @sueann; #1607, #1568, #1536, #1478, #1406, #1408, @smurching; #1504, @LizaShak; #1490, @acroz; #1633, #1631, #1603, #1589, #1569, #1526, #1446, #1438, @apurva-koti; #1456, @Taur1ne; #1547, #1495, @aarondav; #1610, #1600, #1492, #1493, #1447, @tomasatdatabricks; #1430, @javierluraschi; #1424, @nathansuh; #1488, @henningsway; #1590, #1427, @Zangr; #1629, #1614, #1574, #1521, #1522, @dbczumar; #1577, #1514, @ankitmathur-db; #1588, #1566, @stbof; #1575, #1599, @max-allen-db; #1592, @abaveja313; #1606, @andrewmchen)


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
- New ``onnx`` model flavor for saving, loading, and evaluating ONNX models with MLflow. ONNX flavor APIs are available in the ``mlflow.onnx`` module. (#1127, @avflor, @dbczumar; #1388, #1389, @dbczumar)
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
- [Models] Update model flavors to lazily import dependencies in Python. Modules that define Model flavors now import extra dependencies such as ``tensorflow``, ``scikit-learn``, and ``pytorch`` inside individual _methods_, ensuring that these modules can be imported and explored even if the dependencies have not been installed on your system. Also, the ``DEFAULT_CONDA_ENVIRONMENT`` module variable has been replaced with a ``get_default_conda_env()`` function for each flavor.  (#1238, @dbczumar)
- [Models] It is now possible to pass extra arguments to ``mlflow.keras.load_model`` that will be passed through to ``keras.load_model``. (#1330, @yorickvP)
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
