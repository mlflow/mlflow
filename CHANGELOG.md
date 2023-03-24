# CHANGELOG

## 2.2.2 (2023-03-14)

MLflow 2.2.2 is a patch release containing the following bug fixes:

- [Model Registry] Allow `source` to be a local path within a run's artifact directory if a `run_id` is specified (#7993, @harupy)
- [Model Registry] Fix a bug where a windows UNC path is considered a local path (#7988, @WeichenXu123)
- [Model Registry] Disallow `name` to be a file path in  `FileStore.get_registered_model` (#7965, @harupy)

## 2.2.1 (2023-03-02)

MLflow 2.2.1 is a patch release containing the following bug fixes:

- [Model Registry] Fix a bug that caused too many results to be requested by default when calling ``MlflowClient.search_model_versions()`` (#7935, @dbczumar)
- [Model Registry] Patch for GHSA-xg73-94fp-g449 (#7908, @harupy)
- [Model Registry] Patch for GHSA-wp72-7hj9-5265 (#7965, @harupy)

## 2.2.0 (2023-02-28)

MLflow 2.2.0 includes several major features and improvements

Features:

- [Recipes] Add support for score calibration to the classification recipe (#7744, @sunishsheth2009)
- [Recipes] Add automatic label encoding to the classification recipe (#7711, @sunishsheth2009)
- [Recipes] Support custom data splitting logic in the classification and regression recipes (#7815, #7588, @sunishsheth2009)
- [Recipes] Introduce customizable MLflow Run name prefixes to the classification and regression recipes (#7746, @kamalesh0406; #7763, @sunishsheth2009)
- [UI] Add a new Chart View to the MLflow Experiment Page for model performance insights (#7864, @hubertzub-db, @apurva-koti, @prithvikannan, @ridhimag11, @sunishseth2009, @dbczumar)
- [UI] Modernize and improve parallel coordinates chart for model tuning (#7864, @hubertzub-db, @apurva-koti, @prithvikannan, @ridhimag11, @sunishseth2009, @dbczumar)
- [UI] Add typeahead suggestions to the MLflow Experiment Page search bar (#7864, @hubertzub-db, @apurva-koti, @prithvikannan, @ridhimag11, @sunishseth2009, @dbczumar)
- [UI] Improve performance of Experiments Sidebar for large numbers of experiments (#7804, @jmahlik)
- [Tracking] Introduce autologging support for native PyTorch models (#7627, @temporaer)
- [Tracking] Allow specifying ``model_format`` when autologging XGBoost models (#7781, @guyrosin)
- [Tracking] Add ``MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT`` environment variable to configure artifact operation timeouts (#7783, @wamartin-aml)
- [Artifacts] Include ``Content-Type`` response headers for artifacts downloaded from ``mlflow server`` (#7827, @bali0019)
- [Model Registry] Introduce the ``searchModelVersions()`` API to the Java client (#7880, @gabrielfu)
- [Model Registry] Introduce ``max_results``, ``order_by`` and ``page_token`` arguments to ``MlflowClient.search_model_versions()`` (#7623, @serena-ruan)
- [Models] Support logging large ONNX models by using external data (#7808, @dogeplusplus)
- [Models] Add support for logging Diviner models fit in Spark (#7800, @BenWilson2)
- [Models] Introduce ``MLFLOW_DEFAULT_PREDICTION_DEVICE`` environment variable to set the device for pyfunc model inference (#7922, @ankit-db)
- [Scoring] Publish official Docker images for the MLflow Model scoring server at github.com/mlflow/mlflow/pkgs (#7759, @dbczumar)

Bug fixes:

- [Recipes] Fix dataset format validation in the ingest step for custom dataset sources (#7638, @sunishsheth2009)
- [Recipes] Fix bug in identification of worst performing examples during training (#7658, @sunishsheth2009)
- [Recipes] Ensure consistent rendering of the recipe graph when ``inspect()`` is called (#7852, @sunishsheth2009)
- [Recipes] Correctly respect ``positive_class`` configuration in the transform step (#7626, @sunishsheth2009)
- [Recipes] Make logged metric names consistent with ``mlflow.evaluate()`` (#7613, @sunishsheth2009)
- [Recipes] Add ``run_id`` and ``artifact_path`` keys to logged MLmodel files (#7651, @sunishsheth2009)
- [UI] Fix bugs in UI validation of experiment names, model names, and tag keys (#7818, @subramaniam02)
- [Tracking] Resolve artifact locations to absolute paths when creating experiments (#7670, @bali0019)
- [Tracking] Exclude Delta checkpoints from Spark datasource autologging (#7902, @harupy)
- [Tracking] Consistently return an empty list from GetMetricHistory when a metric does not exist (#7589, @bali0019; #7659, @harupy)
- [Artifacts] Fix support for artifact operations on Windows paths in UNC format (#7750, @bali0019)
- [Artifacts] Fix bug in HDFS artifact listing (#7581, @pwnywiz)
- [Model Registry] Disallow creation of model versions with local filesystem sources in ``mlflow server`` (#7908, @harupy)
- [Model Registry] Fix handling of deleted model versions in FileStore (#7716, @harupy)
- [Model Registry] Correctly initialize Model Registry SQL tables independently of MLflow Tracking (#7704, @harupy)
- [Models] Correctly move PyTorch model outputs from GPUs to CPUs during inference with pyfunc (#7885, @ankit-db)
- [Build] Fix compatiblility issues with Python installations compiled using ``PYTHONOPTIMIZE=2`` (#7791, @dbczumar)
- [Build] Fix compatibility issues with the upcoming pandas 2.0 release (#7899, @harupy; #7910, @dbczumar)

Documentation updates:

- [Docs] Add an example of saving and loading Spark MLlib models with MLflow (#7706, @dipanjank)
- [Docs] Add usage examples for ``mlflow.lightgbm`` APIs (#7565, @canerturkseven)
- [Docs] Add an example of custom model flavor creation with ``sktime`` (#7624, @benjaminbluhm)
- [Docs] Clarify ``precision_recall_auc`` metric calculation in ``mlflow.evaluate()`` (#7701, @BenWilson2)
- [Docs] Remove outdated example links (#7587, @asloan7)

Small bug fixes and documentation updates:

#7866, #7751, #7724, #7699, #7697, #7666, @alekseyolg; #7896, #7861, #7858, #7862, #7872, #7859, #7863, #7767, #7766, #7765, #7741, @smurching; #7895, #7877, @viditjain99; #7898, @midhun1998; #7891, #7892, #7886, #7882, #7883, #7875, #7874, #7871, #7868, #7854, #7847, #7845, #7838, #7830, #7837, #7836, #7834, #7831, #7828, #7825, #7826, #7824, #7823, #7778, #7780, #7776, #7775, #7773, #7772, #7769, #7756, #7768, #7764, #7685, #7726, #7722, #7720, #7423, #7712, #7710, #7713, #7688, #7663, #7674, #7673, #7672, #7662, #7653, #7646, #7615, #7614, #7586, #7601, #7598, #7602, #7599, #7577, #7585, #7583, #7584, @harupy; #7865, #7803, #7753, #7719, @dipanjank; #7796, @serena-ruan; #7849, @turbotimon; #7822, #7600, @WeichenXu123; #7811, @guyrosin; #7812, #7788, #7787, #7748, #7730, #7616, #7593, @dbczumar; #7793, @Joel-hanson; #7792, #7694, #7643, @BenWilson2; #7771, #7657, #7644, @nsenno-dbr; #7738, @wkrt7; #7740, @Ark-kun; #7739, #7733, @bali0019; #7723, @andrehp; #7691, #7582, @agoyot; #7721, @Eseeldur; #7709, @srowen; #7693, @ry3s; #7649, @funkypenguin; #7665, @benjaminbluhm; #7668, @eltociear; #7550, @danielhstahl; #7920, @arjundc-db

## 2.1.0 (2022-12-21)

MLflow 2.1.0 includes several major features and improvements

Features:

- [Recipes] Introduce support for multi-class classification (#7458, @mshtelma)
- [Recipes] Extend the pyfunc representation of classification models to output scores in addition to labels (#7474, @sunishsheth2009)
- [UI] Add user ID and lifecycle stage quick search links to the Runs page (#7462, @jaeday)
- [Tracking] Paginate the GetMetricHistory API (#7523, #7415, @BenWilson2)
- [Tracking] Add Runs search aliases for Run name and start time that correspond to UI column names (#7492, @apurva-koti)
- [Tracking] Add a ``/version`` endpoint to ``mlflow server`` for querying the server's MLflow version (#7273, @joncarter1)
- [Model Registry] Add FileStore support for the Model Registry (#6605, @serena-ruan)
- [Model Registry] Introduce an ``mlflow.search_registered_models()`` fluent API (#7428, @TSienki)
- [Model Registry / Java] Add a ``getRegisteredModel()`` method to the Java client (#6602) (#7511, @drod331)
- [Model Registry / R] Add an ``mlflow_set_model_version_tag()`` method to the R client (#7401, @leeweijie)
- [Models] Introduce a ``metadata`` field to the MLmodel specification and ``log_model()`` methods (#7237, @jdonzallaz)
- [Models] Extend ``Model.load()`` to support loading MLmodel specifications from remote locations (#7517, @dbczumar)
- [Models] Pin the major version of MLflow in Models' ``requirements.txt`` and ``conda.yaml`` files (#7364, @BenWilson2)
- [Scoring] Extend ``mlflow.pyfunc.spark_udf()`` to support StructType results (#7527, @WeichenXu123)
- [Scoring] Extend TensorFlow and Keras Models to support multi-dimensional inputs with ``mlflow.pyfunc.spark_udf()``(#7531, #7291, @WeichenXu123)
- [Scoring] Support specifying deployment environment variables and tags when deploying models to SageMaker (#7433, @jhallard)

Bug fixes:

- [Recipes] Fix a bug that prevented use of custom ``early_stop`` functions during model tuning (#7538, @sunishsheth2009)
- [Recipes] Fix a bug in the logic used to create a Spark session during data ingestion (#7307, @WeichenXu123)
- [Tracking] Make the metric names produced by ``mlflow.autolog()`` consistent with ``mlflow.evaluate()`` (#7418, @wenfeiy-db)
- [Tracking] Fix an autologging bug that caused nested, redundant information to be logged for XGBoost and LightGBM models (#7404, @WeichenXu123)
- [Tracking] Correctly classify SQLAlchemy OperationalErrors as retryable HTTP errors (#7240, @barrywhart)
- [Artifacts] Correctly handle special characters in credentials when using FTP artifact storage (#7479, @HCTsai)
- [Models] Address an issue that prevented MLeap models from being saved on Windows (#6966, @dbczumar)
- [Scoring] Fix a permissions issue encountered when using NFS during model scoring with ``mlflow.pyfunc.spark_udf()`` (#7427, @WeichenXu123)

Documentation updates:

- [Docs] Add more examples to the Runs search documentation page (#7487, @apurva-koti)
- [Docs] Add documentation for Model flavors developed by the community (#7425, @mmerce)
- [Docs] Add an example for logging and scoring ONNX Models (#7398, @Rusteam)
- [Docs] Fix a typo in the model scoring REST API example for inputs with the ``dataframe_split`` format (#7540, @zhouyangyu)
- [Docs] Fix a typo in the model scoring REST API example for inputs with the ``dataframe_records`` format (#7361, @dbczumar)

Small bug fixes and documentation updates:

#7571, #7543, #7529, #7435, #7399, @WeichenXu123; #7568, @xiaoye-hua; #7549, #7557, #7509, #7498, #7499, #7485, #7486, #7484, #7391, #7388, #7390, #7381, #7366, #7348, #7346, #7334, #7340, #7323, @BenWilson2; #7561, #7562, #7560, #7553, #7546, #7539, #7544, #7542, #7541, #7533, #7507, #7470, #7469, #7467, #7466, #7464, #7453, #7449, #7450, #7440, #7430, #7436, #7429, #7426, #7410, #7406, #7409, #7407, #7405, #7396, #7393, #7395, #7384, #7376, #7379, #7375, #7354, #7353, #7351, #7352, #7350, #7345, #6493, #7343, #7344, @harupy; #7494, @dependabot[bot]; #7526, @tobycheese; #7489, @liangz1; #7534, @Jingnan-Jia; #7496, @danielhstahl; #7504, #7503, #7459, #7454, #7447, @tsugumi-sys; #7461, @wkrt7; #7451, #7414, #7372, #7289, @sunishsheth2009; #7441, @ikrizanic; #7432, @Pochingto; #7386, @jhallard; #7370, #7373, #7371, #7336, #7341, #7342, @dbczumar; #7335, @prithvikannan

## 2.0.1 (2022-11-14)

The 2.0.1 version of MLflow is a major milestone release that focuses on simplifying the management of end-to-end MLOps workflows, providing new feature-rich functionality, and expanding upon the production-ready MLOps capabilities offered by MLflow.
This release contains several important breaking changes from the 1.x API, additional major features and improvements.

Features:

- [Recipes] MLflow Pipelines is now MLflow Recipes - a framework that enables data scientists to quickly develop high-quality models and deploy them to production
- [Recipes] Add support for classification models to MLflow Recipes (#7082, @bbarnes52)
- [UI] Introduce support for pinning runs within the experiments UI (#7177, @harupy)
- [UI] Simplify the layout and provide customized displays of metrics, parameters, and tags within the experiments UI (#7177, @harupy)
- [UI] Simplify run filtering and ordering of runs within the experiments UI (#7177, @harupy)
- [Tracking] Update `mlflow.pyfunc.get_model_dependencies()` to download all referenced requirements files for specified models (#6733, @harupy)
- [Tracking] Add support for selecting the Keras model `save_format` used by `mlflow.tensorflow.autolog()` (#7123, @balvisio)
- [Models] Set `mlflow.evaluate()` status to stable as it is now a production-ready API
- [Models] Simplify APIs for specifying custom metrics and custom artifacts during model evaluation with `mlflow.evaluate()` (#7142, @harupy)
- [Models] Correctly infer the positive label for binary classification within `mlflow.evaluate()` (#7149, @dbczumar)
- [Models] Enable automated signature logging for `tensorflow` and `keras` models when `mlflow.tensorflow.autolog()` is enabled (#6678, @BenWilson2)
- [Models] Add support for native Keras and Tensorflow Core models within `mlflow.tensorflow` (#6530, @WeichenXu123)
- [Models] Add support for defining the `model_format` used by `mlflow.xgboost.save/log_model()` (#7068, @AvikantSrivastava)
- [Scoring] Overhaul the model scoring REST API to introduce format indicators for inputs and support multiple output fields (#6575, @tomasatdatabricks; #7254, @adriangonz)
- [Scoring] Add support for ragged arrays in model signatures (#7135, @trangevi)
- [Java] Add `getModelVersion` API to the java client (#6955, @wgottschalk)

Breaking Changes:

The following list of breaking changes are arranged by their order of significance within each category.

- [Core] Support for Python 3.7 has been dropped. MLflow now requires Python >=3.8
- [Recipes] `mlflow.pipelines` APIs have been replaced with `mlflow.recipes`
- [Tracking / Registry] Remove `/preview` routes for Tracking and Model Registry REST APIs (#6667, @harupy)
- [Tracking] Remove deprecated `list` APIs for experiments, models, and runs from Python, Java, R, and REST APIs (#6785, #6786, #6787, #6788, #6800, #6868, @dbczumar)
- [Tracking] Remove deprecated `runs` response field from `Get Experiment` REST API response (#6541, #6524 @dbczumar)
- [Tracking] Remove deprecated `MlflowClient.download_artifacts` API (#6537, @WeichenXu123)
- [Tracking] Change the behavior of environment variable handling for `MLFLOW_EXPERIMENT_NAME` such that the value is always used when creating an experiment (#6674, @BenWilson2)
- [Tracking] Update `mlflow server` to run in `--serve-artifacts` mode by default (#6502, @harupy)
- [Tracking] Update Experiment ID generation for the Filestore backend to enable threadsafe concurrency (#7070, @BenWilson2)
- [Tracking] Remove `dataset_name` and `on_data_{name | hash}` suffixes from `mlflow.evaluate()` metric keys (#7042, @harupy)
- [Models / Scoring / Projects] Change default environment manager to `virtualenv` instead of `conda` for model inference and project execution (#6459, #6489 @harupy)
- [Models] Move Keras model logging APIs to the `mlflow.tensorflow` flavor and drop support for TensorFlow Estimators (#6530, @WeichenXu123)
- [Models] Remove deprecated `mlflow.sklearn.eval_and_log_metrics()` API in favor of `mlflow.evaluate()` API (#6520, @dbczumar)
- [Models] Require `mlflow.evaluate()` model inputs to be specified as URIs (#6670, @harupy)
- [Models] Drop support for returning custom metrics and artifacts from the same function when using `mlflow.evaluate()`, in favor of `custom_artifacts` (#7142, @harupy)
- [Models] Extend `PyFuncModel` spec to support `conda` and `virtualenv` subfields (#6684, @harupy)
- [Scoring] Remove support for defining input formats using the `Content-Type` header (#6575, @tomasatdatabricks; #7254, @adriangonz)
- [Scoring] Replace the `--no-conda` CLI option argument for native serving with `--env-manager='local'` (#6501, @harupy)
- [Scoring] Remove public APIs for `mlflow.sagemaker.deploy()` and `mlflow.sagemaker.delete()` in favor of MLflow deployments APIs, such as `mlflow deployments -t sagemaker` (#6650, @dbczumar)
- [Scoring] Rename input argument `df` to `inputs` in `mlflow.deployments.predict()` method (#6681, @BenWilson2)
- [Projects] Replace the `use_conda` argument with the `env_manager` argument within the `run` CLI command for MLflow Projects (#6654, @harupy)
- [Projects] Modify the MLflow Projects docker image build options by renaming `--skip-image-build` to `--build-image` with a default of `False` (#7011, @harupy)
- [Integrations/Azure] Remove deprecated `mlflow.azureml` modules from MLflow in favor of the `azure-mlflow` deployment plugin (#6691, @BenWilson2)
- [R] Remove conda integration with the R client (#6638, @harupy)

Bug fixes:

- [Recipes] Fix rendering issue with profile cards polyfill (#7154, @hubertzub-db)
- [Tracking] Set the MLflow Run name correctly when specified as part of the `tags` argument to `mlflow.start_run()` (#7228, @Cokral)
- [Tracking] Fix an issue with conflicting MLflow Run name assignment if the `mlflow.runName` tag is set (#7138, @harupy)
- [Scoring] Fix incorrect payload constructor error in SageMaker deployment client `predict()` API (#7193, @dbczumar)
- [Scoring] Fix an issue where `DataCaptureConfig` information was not preserved when updating a Sagemaker deployment (#7281, @harupy)

Small bug fixes and documentation updates:

#7309, #7314, #7288, #7276, #7244, #7207, #7175, #7107, @sunishsheth2009; #7261, #7313, #7311, #7249, #7278, #7260, #7284, #7283, #7263, #7266, #7264, #7267, #7265, #7250, #7259, #7247, #7242, #7143, #7214, #7226, #7230, #7227, #7229, #7225, #7224, #7223, #7210, #7192, #7197, #7196, #7204, #7198, #7191, #7189, #7184, #7182, #7170, #7183, #7131, #7165, #7151, #7164, #7168, #7150, #7128, #7028, #7118, #7117, #7102, #7072, #7103, #7101, #7100, #7099, #7098, #7041, #7040, #6978, #6768, #6719, #6669, #6658, #6656, #6655, #6538, #6507, #6504 @harupy; #7310, #7308, #7300, #7290, #7239, #7220, #7127, #7091, #6713 @BenWilson2; #7332, #7299, #7271, #7209, #7180, #7179, #7158, #7147, #7114, @prithvikannan; #7275, #7245, #7134, #7059, @jinzhang21; #7306, #7298, #7287, #7272, #7258, #7236, @ayushthe1; #7279, @tk1012; #7219, @rddefauw; #7333, #7218, #7208, #7188, #7190, #7176, #7137, #7136, #7130, #7124, #7079, #7052, #6541 @dbczumar; #6640, @WeichenXu123; #7200, @hubertzub-db; #7121, @Gonmeso; #6988, @alonisser; #7141, @pdifranc; #7086, @jerrylian-db; #7286, @shogohida

## 1.30.0 (2022-10-19)

MLflow 1.30.0 includes several major features and improvements

Features:

- [Pipelines] Introduce hyperparameter tuning support to MLflow Pipelines (#6859, @prithvikannan)
- [Pipelines] Introduce support for prediction outlier comparison to training data set (#6991, @jinzhang21) 
- [Pipelines] Introduce support for recording all training parameters for reproducibility (#7026, #7094, @prithvikannan)
- [Pipelines] Add support for `Delta` tables as a datasource in the ingest step (#7010, @sunishsheth2009)
- [Pipelines] Add expanded support for data profiling up to 10,000 columns (#7035, @prithvikanna)
- [Pipelines] Add support for AutoML in MLflow Pipelines using FLAML (#6959, @mshtelma)
- [Pipelines] Add support for simplified transform step execution by allowing for unspecified configuration (#6909, @apurva-koti)
- [Pipelines] Introduce a data preview tab to the transform step card (#7033, @prithvikannan)
- [Tracking] Introduce `run_name` attribute for `create_run`, `get_run` and `update_run` APIs (#6782, #6798 @apurva-koti)
- [Tracking] Add support for searching by `creation_time` and `last_update_time` for the `search_experiments` API  (#6979, @harupy)
- [Tracking] Add support for search terms `run_id IN` and `run ID NOT IN` for the `search_runs` API (#6945, @harupy)
- [Tracking] Add support for searching by `user_id` and `end_time` for the `search_runs` API (#6881, #6880 @subramaniam02)
- [Tracking] Add support for searching by `run_name` and `run_id` for the `search_runs` API (#6899, @harupy; #6952, @alexacole)
- [Tracking] Add support for synchronizing run `name` attribute and `mlflow.runName` tag (#6971, @BenWilson2)
- [Tracking] Add support for signed tracking server requests using AWSSigv4 and AWS IAM (#7044, @pdifranc)
- [Tracking] Introduce the `update_run()` API for modifying the `status` and `name` attributes of existing runs (#7013, @gabrielfu)
- [Tracking] Add support for experiment deletion in the `mlflow gc` cli API (#6977, @shaikmoeed)
- [Models] Add support for environment restoration in the `evaluate()` API (#6728, @jerrylian-db)
- [Models] Remove restrictions on binary classification labels in the `evaluate()` API (#7077, @dbczumar)
- [Scoring] Add support for `BooleanType` to `mlflow.pyfunc.spark_udf()` (#6913, @BenWilson2)
- [SQLAlchemy] Add support for configurable `Pool` class options for `SqlAlchemyStore` (#6883, @mingyu89)

Bug fixes:

- [Pipelines] Enable Pipeline subprocess commands to create a new `SparkSession` if one does not exist (#6846, @prithvikannan)
- [Pipelines] Fix a rendering issue with `bool` column types in Step Card data profiles (#6907, @sunishsheth2009)
- [Pipelines] Add validation and an exception if required step files are missing (#7067, @mingyu89)
- [Pipelines] Change step configuration validation to only be performed during runtime execution of a step (#6967, @prithvikannan)
- [Tracking] Fix infinite recursion bug when inferring the model schema in `mlflow.pyspark.ml.autolog()` (#6831, @harupy)
- [UI] Remove the browser error notification when failing to fetch artifacts (#7001, @kevingreer)
- [Models] Allow `mlflow-skinny` package to serve as base requirement in `MLmodel` requirements (#6974, @BenWilson2)
- [Models] Fix an issue with code path resolution for loading SparkML models (#6968, @dbczumar)
- [Models] Fix an issue with dependency inference in logging SparkML models (#6912, @BenWilson2)
- [Models] Fix an issue involving potential duplicate downloads for SparkML models (#6903, @serena-ruan)
- [Models] Add missing `pos_label` to `sklearn.metrics.precision_recall_curve` in `mlflow.evaluate()` (#6854, @dbczumar)
- [SQLAlchemy] Fix a bug in `SqlAlchemyStore` where `set_tag()` updates the incorrect tags (#7027, @gabrielfu)

Documentation updates:

- [Models] Update details regarding the default `Keras` serialization format (#7022, @balvisio)

Small bug fixes and documentation updates:

#7093, #7095, #7092, #7064, #7049, #6921, #6920, #6940, #6926, #6923, #6862, @jerrylian-db; #6946, #6954, #6938, @mingyu89; #7047, #7087, #7056, #6936, #6925, #6892, #6860, #6828, @sunishsheth2009; #7061, #7058, #7098, #7071, #7073, #7057, #7038, #7029, #6918, #6993, #6944, #6976, #6960, #6933, #6943, #6941, #6900, #6901, #6898, #6890, #6888, #6886, #6887, #6885, #6884, #6849, #6835, #6834, @harupy; #7094, #7065, #7053, #7026, #7034, #7021, #7020, #6999, #6998, #6996, #6990, #6989, #6934, #6924, #6896, #6895, #6876, #6875, #6861, @prithvikannan; #7081, #7030, #7031, #6965, #6750, @bbarnes52; #7080, #7069, #7051, #7039, #7012, #7004, @dbczumar; #7054, @jinzhang21; #7055, #7037, #7036, #6949, #6951, @apurva-koti; #6815, @michaguenther; #6897, @chaturvedakash; #7025, #6981, #6950, #6948, #6937, #6829, #6830, @BenWilson2; #6982, @vadim; #6985, #6927, @kriscon-db; #6917, #6919, #6872, #6855, @WeichenXu123; #6980, @utkarsh867; #6973, #6935, @wentinghu; #6930, @mingyangge-db; #6956, @RohanBha1; #6916, @av-maslov; #6824, @shrinath-suresh; #6732, @oojo12; #6807, @ikrizanic; #7066, @subramaniam20jan; #7043, @AvikantSrivastava; #6879, @jspablo

## 1.29.0 (2022-09-16)

MLflow 1.29.0 includes several major features and improvements

Features:

- [Pipelines] Improve performance and fidelity of dataset profiling in the scikit-learn regression Pipeline (#6792, @sunishsheth2009) 
- [Pipelines] Add an `mlflow pipelines get-artifact` CLI for retrieving Pipeline artifacts (#6517, @prithvikannan)
- [Pipelines] Introduce an option for skipping dataset profiling to the scikit-learn regression Pipeline (#6456, @apurva-koti)
- [Pipelines / UI] Display an `mlflow pipelines` CLI command for reproducing a Pipeline run in the MLflow UI (#6376, @hubertzub-db)
- [Tracking] Automatically generate friendly names for Runs if not supplied by the user (#6736, @BenWilson2)
- [Tracking] Add `load_text()`, `load_image()` and `load_dict()` fluent APIs for convenient artifact loading (#6475, @subramaniam02)
- [Tracking] Add `creation_time` and `last_update_time` attributes to the Experiment class (#6756, @subramaniam02)
- [Tracking] Add official MLflow Tracking Server Dockerfiles to the MLflow repository (#6731, @oojo12)
- [Tracking] Add `searchExperiments` API to Java client and deprecate `listExperiments` (#6561, @dbczumar)
- [Tracking] Add `mlflow_search_experiments` API to R client and deprecate `mlflow_list_experiments` (#6576, @dbczumar)
- [UI] Make URLs clickable in the MLflow Tracking UI (#6526, @marijncv)
- [UI] Introduce support for csv data preview within the artifact viewer pane (#6567, @nnethery)
- [Model Registry / Models] Introduce `mlflow.models.add_libraries_to_model()` API for adding libraries to an MLflow Model (#6586, @arjundc-db)
- [Models] Add model validation support to `mlflow.evaluate()` (#6582, @jerrylian-db)
- [Models] Introduce `sample_weights` support to `mlflow.evaluate()` (#6806, @dbczumar)
- [Models] Add `pos_label` support to `mlflow.evaluate()` for identifying the positive class (#6696, @harupy)
- [Models] Make the metric name prefix and dataset info configurable in `mlflow.evaluate()` (#6593, @dbczumar)
- [Models] Add utility for validating the compatibility of a dataset with a model signature (#6494, @serena-ruan)
- [Models] Add `predict_proba()` support to the pyfunc representation of scikit-learn models (#6631, @skylarbpayne)
- [Models] Add support for Decimal type inference to MLflow Model schemas (#6600, @shitaoli-db)
- [Models] Add new CLI command for generating Dockerfiles for model serving (#6591, @anuarkaliyev23)
- [Scoring] Add `/health` endpoint to scoring server (#6574, @gabriel-milan)
- [Scoring] Support specifying a `variant_name` during Sagemaker deployment (#6486, @nfarley-soaren)
- [Scoring] Support specifying a `data_capture_config` during SageMaker deployment (#6423, @jonwiggins)

Bug fixes:

- [Tracking] Make Run and Experiment deletion and restoration idempotent (#6641, @dbczumar)
- [UI] Fix an alignment bug affecting the Experiments list in the MLflow UI  (#6569, @sunishsheth2009)
- [Models] Fix a regression in the directory path structure of logged Spark Models that occurred in MLflow 1.28.0 (#6683, @gwy1995)
- [Models] No longer reload the `__main__` module when loading model code (#6647, @Jooakim)
- [Artifacts] Fix an `mlflow server` compatibility issue with HDFS when running in `--serve-artifacts` mode (#6482, @shidianshifen)
- [Scoring] Fix an inference failure with 1-dimensional tensor inputs in TensorFlow and Keras (#6796, @LiamConnell)

Documentation updates:

- [Tracking] Mark the SearchExperiments API as stable (#6551, @dbczumar)
- [Tracking / Model Registry] Deprecate the ListExperiments, ListRegisteredModels, and `list_run_infos()` APIs (#6550, @dbczumar)
- [Scoring] Deprecate `mlflow.sagemaker.deploy()` in favor of `SageMakerDeploymentClient.create()` (#6651, @dbczumar)

Small bug fixes and documentation updates:

#6803, #6804, #6801, #6791, #6772, #6745, #6762, #6760, #6761, #6741, #6725, #6720, #6666, #6708, #6717, #6704, #6711, #6710, #6706, #6699, #6700, #6702, #6701, #6685, #6664, #6644, #6653, #6629, #6639, #6624, #6565, #6558, #6557, #6552, #6549, #6534, #6533, #6516, #6514, #6506, #6509, #6505, #6492, #6490, #6478, #6481, #6464, #6463, #6460, #6461, @harupy; #6810, #6809, #6727, #6648, @BenWilson2; #6808, #6766, #6729, @jerrylian-db; #6781, #6694, @marijncv; #6580, #6661, @bbarnes52; #6778, #6687, #6623, @shraddhafalane; #6662, #6737, #6612, #6595, @sunishsheth2009; #6777, @aviralsharma07; #6665, #6743, #6573, @liangz1; #6784, @apurva-koti; #6753, #6751, @mingyu89; #6690, #6455, #6484, @kriscon-db; #6465, #6689, @hubertzub-db; #6721, @WeichenXu123; #6722, #6718, #6668, #6663, #6621, #6547, #6508, #6474, #6452, @dbczumar; #6555, #6584, #6543, #6542, #6521, @dsgibbons; #6634, #6596, #6563, #6495, @prithvikannan; #6571, @smurching; #6630, #6483, @serena-ruan; #6642, @thinkall; #6614, #6597, @jinzhang21; #6457, @cnphil; #6570, #6559, @kumaryogesh17; #6560, #6540, @iamthen0ise; #6544, @Monkero; #6438, @ahlag; #3292, @dolfinus; #6637, @ninabacc-db; #6632, @arpitjasa-db

## 1.28.0 (2022-08-09)

MLflow 1.28.0 includes several major features and improvements:

Features:

- [Pipelines] Log the full Pipeline runtime configuration to MLflow Tracking during Pipeline execution (#6359, @jinzhang21)
- [Pipelines] Add ``pipeline.yaml`` configurations to specify the Model Registry backend used for model registration (#6284, @sunishsheth2009)
- [Pipelines] Support optionally skipping the ``transform`` step of the scikit-learn regression pipeline (#6362, @sunishsheth2009)
- [Pipelines] Add UI links to Runs and Models in Pipeline Step Cards on Databricks (#6294, @dbczumar)
- [Tracking] Introduce ``mlflow.search_experiments()`` API for searching experiments by name and by tags (#6333, @WeichenXu123; #6227, #6172, #6154, @harupy)
- [Tracking] Increase the maximum parameter value length supported by File and SQL backends to 500 characters (#6358, @johnyNJ)
- [Tracking] Introduce an ``--older-than`` flag to ``mlflow gc`` for removing runs based on deletion time (#6354, @Jason-CKY)
- [Tracking] Add ``MLFLOW_SQLALCHEMYSTORE_POOL_RECYCLE`` environment variable for recycling SQLAlchemy connections (#6344, @postrational)
- [UI] Display deeply nested runs in the Runs Table on the Experiment Page (#6065, @tospe)
- [UI] Add box plot visualization for metrics to the Compare Runs page (#6308, @ahlag)
- [UI] Display tags on the Compare Runs page (#6164, @CaioCavalcanti)
- [UI] Use scientific notation for axes when viewing metric plots in log scale (#6176, @RajezMariner)
- [UI] Add button to Metrics page for downloading metrics as CSV (#6048, @rafaelvp-db)
- [UI] Include NaN and +/- infinity values in plots on the Metrics page (#6422, @hubertzub-db)
- [Tracking / Model Registry] Introduce environment variables to control retry behavior and timeouts for REST API requests (#5745, @peterdhansen)
- [Tracking / Model Registry] Make ``MlflowClient`` importable as ``mlflow.MlflowClient`` (#6085, @subramaniam02)
- [Model Registry] Add support for searching registered models and model versions by tags (#6413, #6411, #6320, @WeichenXu123)
- [Model Registry] Add ``stage`` parameter to ``set_model_version_tag()`` (#6185, @subramaniam02)
- [Model Registry] Add ``--registry-store-uri`` flag to ``mlflow server`` for specifying the Model Registry backend URI (#6142, @Secbone)
- [Models] Improve performance of Spark Model logging on Databricks (#6282, @bbarnes52)
- [Models] Include Pandas Series names in inferred model schemas (#6361, @RynoXLI)
- [Scoring] Make ``model_uri`` optional in ``mlflow models build-docker`` to support building generic model serving images (#6302, @harupy)
- [R] Support logging of NA and NaN parameter values (#6263, @nathaneastwood)

Bug fixes and documentation updates:

- [Pipelines] Improve scikit-learn regression pipeline latency by limiting dataset profiling to the first 100 columns (#6297, @sunishsheth2009)
- [Pipelines] Use ``xdg-open`` instead of ``open`` for viewing Pipeline results on Linux systems (#6326, @strangiato)
- [Pipelines] Fix a bug that skipped Step Card rendering in Jupyter Notebooks (#6378, @apurva-koti)
- [Tracking] Use the 401 HTTP response code in authorization failure REST API responses, instead of 500 (#6106, @balvisio)
- [Tracking] Correctly classify artifacts as files and directories when using Azure Blob Storage (#6237, @nerdinand)
- [Tracking] Fix a bug in the File backend that caused run metadata to be lost in the event of a failed write (#6388, @dbczumar)
- [Tracking] Adjust ``mlflow.pyspark.ml.autolog()`` to only log model signatures for supported input / output data types (#6365, @harupy)
- [Tracking] Adjust ``mlflow.tensorflow.autolog()`` to log TensorFlow early stopping callback info when ``log_models=False`` is specified (#6170, @WeichenXu123)
- [Tracking] Fix signature and input example logging errors in ``mlflow.sklearn.autolog()`` for models containing transformers (#6230, @dbczumar)
- [Tracking] Fix a failure in ``mlflow gc`` that occurred when removing a run whose artifacts had been previously deleted (#6165, @dbczumar)
- [Tracking] Add missing ``sqlparse`` library to MLflow Skinny client, which is required for search support (#6174, @dbczumar)
- [Tracking / Model Registry] Fix an ``mlflow server`` bug that rejected parameters and tags with empty string values (#6179, @dbczumar)
- [Model Registry] Fix a failure preventing model version schemas from being downloaded with ``--serve-arifacts`` enabled (#6355, @abbas123456)
- [Scoring] Patch the Java Model Server to support MLflow Models logged on recent versions of the Databricks Runtime (#6337, @dbczumar)
- [Scoring] Verify that either the deployment name or endpoint is specified when invoking the ``mlflow deployments predict`` CLI (#6323, @dbczumar)
- [Scoring] Properly encode datetime columns when performing batch inference with ``mlflow.pyfunc.spark_udf()`` (#6244, @harupy)
- [Projects] Fix an issue where local directory paths were misclassified as Git URIs when running Projects (#6218, @ElefHead)
- [R] Fix metric logging behavior for +/- infinity values (#6271, @nathaneastwood)
- [Docs] Move Python API docs for ``MlflowClient`` from ``mlflow.tracking`` to ``mlflow.client`` (#6405, @dbczumar)
- [Docs] Document that MLflow Pipelines requires Make (#6216, @dbczumar)
- [Docs] Improve documentation for developing and testing MLflow JS changes in ``CONTRIBUTING.rst`` (#6330, @ahlag)

Small bug fixes and doc updates (#6322, #6321, #6213, @KarthikKothareddy; #6409, #6408, #6396, #6402, #6399, #6398, #6397, #6390, #6381, #6386, #6385, #6373, #6375, #6380, #6374, #6372, #6363, #6353, #6352, #6350, #6351, #6349, #6347, #6287, #6341, #6342, #6340, #6338, #6319, #6314, #6316, #6317, #6318, #6315, #6313, #6311, #6300, #6292, #6291, #6289, #6290, #6278, #6279, #6276, #6272, #6252, #6243, #6250, #6242, #6241, #6240, #6224, #6220, #6208, #6219, #6207, #6171, #6206, #6199, #6196, #6191, #6190, #6175, #6167, #6161, #6160, #6153, @harupy; #6193, @jwgwalton; #6304, #6239, #6234, #6229, @sunishsheth2009; #6258, @xanderwebs; #6106, @balvisio; #6303, @bbarnes52; #6117, @wenfeiy-db; #6389, #6214, @apurva-koti; #6412, #6420, #6277, #6266, #6260, #6148, @WeichenXu123; #6120, @ameya-parab; #6281, @nathaneastwood; #6426, #6415, #6417, #6418, #6257, #6182, #6157, @dbczumar; #6189, @shrinath-suresh; #6309, @SamirPS; #5897, @temporaer; #6251, @herrmann; #6198, @sniafas; #6368, #6158, @jinzhang21; #6236, @subramaniam02; #6036, @serena-ruan; #6430, @ninabacc-db)

## 1.27.0 (2022-06-27)

MLflow 1.27.0 includes several major features and improvements:

- [**Pipelines**] With MLflow 1.27.0, we are excited to announce the release of
[**MLflow Pipelines**](https://mlflow.org/docs/latest/pipelines.html), an opinionated framework for
structuring MLOps workflows that simplifies and standardizes machine learning application development
and productionization. MLflow Pipelines makes it easy for data scientists to follow best practices
for creating production-ready ML deliverables, allowing them to focus on developing excellent models.
MLflow Pipelines also enables ML engineers and DevOps teams to seamlessly deploy models to production
and incorporate them into applications. To get started with MLflow Pipelines, check out the docs at
https://mlflow.org/docs/latest/pipelines.html. (#6115)

- [UI] Introduce UI support for searching and comparing runs across multiple Experiments (#5971, @r3stl355)

More features:

- [Tracking] When using batch logging APIs, automatically split large sets of metrics, tags, and params into multiple requests (#6052, @nzw0301)
- [Tracking] When an Experiment is deleted, SQL-based backends also move the associate Runs to the "deleted" lifecycle stage (#6064, @AdityaIyengar27)
- [Tracking] Add support for logging single-element ``ndarray`` and tensor instances as metrics via the ``mlflow.log_metric()`` API (#5756, @ntakouris)
- [Models] Add support for ``CatBoostRanker`` models to the ``mlflow.catboost`` flavor (#6032, @danielgafni)
- [Models] Integrate SHAP's ``KernelExplainer`` with ``mlflow.evaluate()``, enabling model explanations on categorical data (#6044, #5920, @WeichenXu123)
- [Models] Extend ``mlflow.evaluate()`` to automatically log the ``score()`` outputs of scikit-learn models as metrics (#5935, #5903, @WeichenXu123)

Bug fixes and documentation updates:

- [UI] Fix broken model links in the Runs table on the MLflow Experiment Page (#6014, @hctpbl)
- [Tracking/Installation] Require ``sqlalchemy>=1.4.0`` upon MLflow installation, which is necessary for usage of SQL-based MLflow Tracking backends (#6024, @sniafas)
- [Tracking] Fix a regression that caused ``mlflow server`` to reject ``LogParam`` API requests containing empty string values (#6031, @harupy)
- [Tracking] Fix a failure in scikit-learn autologging that occurred when ``matplotlib`` was not installed on the host system (#5995, @fa9r)
- [Tracking] Fix a failure in TensorFlow autologging that occurred when training models on ``tf.data.Dataset`` inputs (#6061, @dbczumar)
- [Artifacts] Address artifact download failures from SFTP locations that occurred due to mismanaged concurrency (#5840, @rsundqvist)
- [Models] Fix a bug where MLflow Models did not restore bundled code properly if multiple models use the same code module name (#5926, @BFAnas)
- [Models] Address an issue where ``mlflow.sklearn.model()`` did not properly restore bundled model code (#6037, @WeichenXu123)
- [Models] Fix a bug in ``mlflow.evaluate()`` that caused input data objects to be mutated when evaluating certain scikit-learn models (#6141, @dbczumar)
- [Models] Fix a failure in ``mlflow.pyfunc.spark_udf`` that occurred when the UDF was invoked on an empty RDD partition (#6063, @WeichenXu123)
- [Models] Fix a failure in ``mlflow models build-docker`` that occurred when ``env-manager=local`` was specified (#6046, @bneijt)
- [Projects] Improve robustness of the git repository check that occurs prior to MLflow Project execution (#6000, @dkapur17)
- [Projects] Address a failure that arose when running a Project that does not have a ``master`` branch (#5889, @harupy)
- [Docs] Correct several typos throughout the MLflow docs (#5959, @ryanrussell)

Small bug fixes and doc updates (#6041, @drsantos89; #6138, #6137, #6132, @sunishsheth2009; #6144, #6124, #6125, #6123, #6057, #6060, #6050, #6038, #6029, #6030, #6025, #6018, #6019, #5962, #5974, #5972, #5957, #5947, #5907, #5938, #5906, #5932, #5919, #5914, #5888, #5890, #5886, #5873, #5865, #5843, @harupy; #6113, @comojin1994; #5930, @yashaswikakumanu; #5837, @shrinath-suresh; #6067, @deepyaman; #5997, @idlefella; #6021, @BenWilson2; #5984, @Sumanth077; #5929, @krunal16-c; #5879, @kugland; #5875, @ognis1205; #6006, @ryanrussell; #6140, @jinzhang21; #5983, @elk15; #6022, @apurva-koti; #5982, @EB-Joel; #5981, #5980, @punitkashyup; #6103, @ikrizanic; #5988, #5969, @SaumyaBhushan; #6020, #5991, @WeichenXu123; #5910, #5912, @Dark-Knight11; #6005, @Asinsa; #6023, @subramaniam02; #5999, @Regis-Caelum; #6007, @CaioCavalcanti; #5943, @kvaithin; #6017, #6002, @NeoKish; #6111, @T1b4lt; #5986, @seyyidibrahimgulec; #6053, @Zohair-coder; #6146, #6145, #6143, #6139, #6134, #6136, #6135, #6133, #6071, #6070, @dbczumar; #6026, @rotate2050)

## 1.26.1 (2022-05-27)

MLflow 1.26.1 is a patch release containing the following bug fixes:

- [Installation] Fix compatibility issue with ``protobuf >= 4.21.0`` (#5945, @harupy)
- [Models] Fix ``get_model_dependencies`` behavior for ``models:`` URIs containing artifact paths (#5921, @harupy)
- [Models] Revert a problematic change to ``artifacts`` persistence in ``mlflow.pyfunc.log_model()`` that was introduced in MLflow 1.25.0 (#5891, @kyle-jarvis)
- [Models] Close associated image files when ``EvaluationArtifact`` outputs from ``mlflow.evaluate()`` are garbage collected (#5900, @WeichenXu123)

Small bug fixes and updates (#5874, #5942, #5941, #5940, #5938, @harupy; #5893, @PrajwalBorkar; #5909, @yashaswikakumanu; #5937, @BenWilson2)

## 1.26.0 (2022-05-16)

MLflow 1.26.0 includes several major features and improvements:

Features:

- [CLI] Add endpoint naming and options configuration to the deployment CLI (#5731, @trangevi)
- [Build,Doc] Add development environment setup script for Linux and MacOS x86 Operating Systems (#5717, @BenWilson2)
- [Tracking] Update `mlflow.set_tracking_uri` to add support for paths defined as `pathlib.Path` in addition to existing `str` path declarations (#5824, @cacharle)
- [Scoring] Add custom timeout override option to the scoring server CLI to support high latency models (#5663, @sniafas)
- [UI] Add sticky header to experiment run list table to support column name visibility when scrolling beyond page fold (#5818, @hubertzub-db)
- [Artifacts] Add GCS support for MLflow garbage collection (#5811, @aditya-iyengar-rtl-de)
- [Evaluate] Add `pos_label` argument for `eval_and_log_metrics` API to support accurate binary classifier evaluation metrics (#5807, @yxiong)
- [UI] Add fields for latest, minimum and maximum metric values on metric display page (#5574, @adamreeve)
- [Models] Add support for `input_example` and `signature` logging for pyspark ml flavor when using autologging (#5719, @bali0019)
- [Models] Add `virtualenv` environment manager support for `mlflow models docker-build` CLI (#5728, @harupy)
- [Models] Add support for wildcard module matching in log_model_allowlist for PySpark models (#5723, @serena-ruan)
- [Projects] Add `virtualenv` environment manager support for MLflow projects (#5631, @harupy)
- [Models] Add `virtualenv` environment manager support for MLflow Models (#5380, @harupy)
- [Models] Add `virtualenv` environment manager support for `mlflow.pyfunc.spark_udf` (#5676, @WeichenXu123)
- [Models] Add support for `input_example` and `signature` logging for `tensorflow` flavor when using autologging (#5510, @bali0019)
- [Server-infra] Add JSON Schema Type Validation to enable raising 400 errors on malformed requests to REST API endpoints (#5458, @mrkaye97)
- [Scoring] Introduce abstract `endpoint` interface for mlflow deployments (#5378, @trangevi)
- [UI] Add `End Time` and `Duration` fields to run comparison page (#3378, @RealArpanBhattacharya)
- [Serving] Add schema validation support when parsing input csv data for model serving (#5531, @vvijay-bolt)

Bug fixes and documentation updates:

- [Models] Fix REPL ID propagation from datasource listener to publisher for Spark data sources (#5826, @dbczumar)
- [UI] Update `ag-grid` and implement `getRowId` to improve performance in the runs table visualization (#5725, @adamreeve)
- [Serving] Fix `tf-serving` parsing to support columnar-based formatting (#5825, @arjundc-db)
- [Artifacts] Update `log_artifact` to support models larger than 2GB in HDFS (#5812, @hitchhicker)
- [Models] Fix autologging to support `lightgbm` metric names with "@" symbols within their names (#5785, @mengchendd)
- [Models] Pyfunc: Fix code directory resolution of subdirectories (#5806, @dbczumar)
- [Server-Infra] Fix mlflow-R server starting failure on windows (#5767, @serena-ruan)
- [Docs] Add documentation for `virtualenv` environment manager support for MLflow projects (#5727, @harupy)
- [UI] Fix artifacts display sizing to support full width rendering in preview pane (#5606, @szczeles)
- [Models] Fix local hostname issues when loading spark model by binding driver address to localhost (#5753, @WeichenXu123)
- [Models] Fix autologging validation and batch_size calculations for `tensorflow` flavor (#5683, @MarkYHZhang)
- [Artifacts] Fix `SqlAlchemyStore.log_batch` implementation to make it log data in batches (#5460, @erensahin)

Small bug fixes and doc updates (#5858, #5859, #5853, #5854, #5845, #5829, #5842, #5834, #5795, #5777, #5794, #5766, #5778, #5765, #5763, #5768, #5769, #5760, #5727, #5748, #5726, #5721, #5711, #5710, #5708, #5703, #5702, #5696, #5695, #5669, #5670, #5668, #5661, #5638, @harupy; #5749, @arpitjasa-db; #5675, @Davidswinkels; #5803, #5797, @ahlag; #5743, @kzhang01; #5650, #5805, #5724, #5720, #5662, @BenWilson2; #5627, @cterrelljones; #5646, @kutal10; #5758, @davideli-db; #5810, @rahulporuri; #5816, #5764, @shrinath-suresh; #5869, #5715, #5737, #5752, #5677, #5636, @WeichenXu123; #5735, @subramaniam02; #5746, @akaigraham; #5734, #5685, @lucalves; #5761, @marcelatoffernet; #5707, @aashish-khub; #5808, @ketangangal; #5730, #5700, @shaikmoeed; #5775, @dbczumar; #5747, @zhixuanevelynwu)

## 1.25.1 (2022-04-13)

MLflow 1.25.1 is a patch release containing the following bug fixes:

- [Models] Fix a `pyfunc` artifact overwrite bug for when multiple artifacts are saved in sub-directories (#5657, @kyle-jarvis)
- [Scoring] Fix permissions issue for Spark workers accessing model artifacts from a temp directory created by the driver (#5684, @WeichenXu123)

## 1.25.0 (2022-04-11)

MLflow 1.25.0 includes several major features and improvements:

Features:

- [Tracking] Introduce a new fluent API `mlflow.last_active_run()` that provides the most recent fluent active run (#5584, @MarkYHZhang)
- [Tracking] Add `experiment_names` argument to the `mlflow.search_runs()` API to support searching runs by experiment names (#5564, @r3stl355)
- [Tracking] Add a `description` parameter to `mlflow.start_run()` (#5534, @dogeplusplus)
- [Tracking] Add `log_every_n_step` parameter to `mlflow.pytorch.autolog()` to control metric logging frequency (#5516, @adamreeve)
- [Tracking] Log `pyspark.ml.param.Params` values as MLflow parameters during PySpark autologging (#5481, @serena-ruan)
- [Tracking] Add support for  `pyspark.ml.Transformer`s to PySpark autologging (#5466, @serena-ruan)
- [Tracking] Add input example and signature autologging for Keras models (#5461, @bali0019)
- [Models] Introduce `mlflow.diviner` flavor for large-scale [time series forecasting](https://databricks-diviner.readthedocs.io/en/latest/?badge=latest) (#5553, @BenWilson2)
- [Models] Add `pyfunc.get_model_dependencies()` API to retrieve reproducible environment specifications for MLflow Models with the pyfunc flavor (#5503, @WeichenXu123)
- [Models] Add `code_paths` argument to all model flavors to support packaging custom module code with MLflow Models (#5448, @stevenchen-db)
- [Models] Support creating custom artifacts when evaluating models with `mlflow.evaluate()` (#5405, #5476 @MarkYHZhang)
- [Models] Add `mlflow_version` field to MLModel specification (#5515, #5576, @r3stl355)
- [Models] Add support for logging models to preexisting destination directories (#5572, @akshaya-a)
- [Scoring / Projects] Introduce `--env-manager` configuration for specifying environment restoration tools (e.g. `conda`) and deprecate `--no-conda` (#5567, @harupy)
- [Scoring] Support restoring model dependencies in `mlflow.pyfunc.spark_udf()` to ensure accurate predictions (#5487, #5561, @WeichenXu123)
- [Scoring] Add support for `numpy.ndarray` type inputs to the TensorFlow pyfunc `predict()` function (#5545, @WeichenXu123)
- [Scoring] Support deployment of MLflow Models to Sagemaker Serverless (#5610, @matthewmayo)
- [UI] Add MLflow version to header beneath logo (#5504, @adamreeve)
- [Artifacts] Introduce a `mlflow.artifacts.download_artifacts()` API mirroring the functionality of the `mlflow artifacts download` CLI (#5585, @dbczumar)
- [Artifacts] Introduce environment variables for controlling GCS artifact upload/download chunk size and timeouts (#5438, #5483, @mokrueger)

Bug fixes and documentation updates:

- [Tracking/SQLAlchemy] Create an index on `run_uuid` for PostgreSQL to improve query performance (#5446, @harupy)
- [Tracking] Remove client-side validation of metric, param, tag, and experiment fields (#5593, @BenWilson2)
- [Projects] Support setting the name of the MLflow Run when executing an MLflow Project (#5187, @bramrodenburg)
- [Scoring] Use pandas `split` orientation for DataFrame inputs to SageMaker deployment `predict()` API to preserve column ordering (#5522, @dbczumar)
- [Server-Infra] Fix runs search compatibility bugs with PostgreSQL, MySQL, and MSSQL (#5540, @harupy)
- [CLI] Fix a bug in the `mlflow-skinny` client that caused `mlflow --version` to fail (#5573, @BenWilson2)
- [Docs] Update guidance and examples for model deployment to AzureML to recommend using the `mlflow-azureml` package (#5491, @santiagxf)

Small bug fixes and doc updates (#5591, #5629, #5597, #5592, #5562, #5477, @BenWilson2; #5554, @juntai-zheng; #5570, @tahesse; #5605, @guelate; #5633, #5632, #5625, #5623, #5615, #5608, #5600, #5603, #5602, #5596, #5587, #5586, #5580, #5577, #5568, #5290, #5556, #5560, #5557, #5548, #5547, #5538, #5513, #5505, #5464, #5495, #5488, #5485, #5468, #5455, #5453, #5454, #5452, #5445, #5431, @harupy; #5640, @nchittela; #5520, #5422, @Ark-kun; #5639, #5604, @nishipy; #5543, #5532, #5447, #5435, @WeichenXu123; #5502, @singankit; #5500, @Sohamkayal4103; #5449, #5442, @apurva-koti; #5552, @vinijaiswal; #5511, @adamreeve; #5428, @jinzhang21; #5309, @sunishsheth2009; #5581, #5559, @Kr4is; #5626, #5618, #5529, @sisp; #5652, #5624, #5622, #5613, #5509, #5459, #5437, @dbczumar; #5616, @liangz1)

## 1.24.0 (2022-02-27)

MLflow 1.24.0 includes several major features and improvements:

Features:

- [Tracking] Support uploading, downloading, and listing artifacts through the MLflow server via `mlflow server --serve-artifacts` (#5320, @BenWilson2, @harupy)
- [Tracking] Add the `registered_model_name` argument to `mlflow.autolog()` for automatic model registration during autologging (#5395, @WeichenXu123)
- [UI] Improve and restructure the Compare Runs page. Additions include "show diff only" toggles and scrollable tables (#5306, @WeichenXu123)
- [Models] Introduce `mlflow.pmdarima` flavor for pmdarima models (#5373, @BenWilson2)
- [Models] When loading an MLflow Model, print a warning if a mismatch is detected between the current environment and the Model's dependencies (#5368, @WeichenXu123)
- [Models] Support computing custom scalar metrics during model evaluation with `mlflow.evaluate()` (#5389, @MarkYHZhang)
- [Scoring] Add support for deploying and evaluating SageMaker models via the [`MLflow Deployments API`](https://mlflow.org/docs/latest/models.html#deployment-to-custom-targets) (#4971, #5396, @jamestran201)

Bug fixes and documentation updates:

- [Tracking / UI] Fix artifact listing and download failures that occurred when operating the MLflow server in `--serve-artifacts` mode (#5409, @dbczumar)
- [Tracking] Support environment-variable-based authentication when making artifact requests to the MLflow server in `--serve-artifacts` mode (#5370, @TimNooren)
- [Tracking] Fix bugs in hostname and path resolution when making artifacts requests to the MLflow server in `--serve-artifacts` mode (#5384, #5385, @mert-kirpici)
- [Tracking] Fix an import error that occurred when `mlflow.log_figure()` was used without `matplotlib.figure` imported (#5406, @WeichenXu123)
- [Tracking] Correctly log XGBoost metrics containing the `@` symbol during autologging (#5403, @maxfriedrich)
- [Tracking] Fix a SQL Server database error that occurred during Runs search (#5382, @dianacarvalho1)
- [Tracking] When downloading artifacts from HDFS, store them in the user-specified destination directory (#5210, @DimaClaudiu)
- [Tracking / Model Registry] Improve performance of large artifact and model downloads (#5359, @mehtayogita)
- [Models] Fix fast.ai PyFunc inference behavior for models with 2D outputs (#5411, @santiagxf)
- [Models] Record Spark model information to the active run when `mlflow.spark.log_model()` is called (#5355, @szczeles)
- [Models] Restore onnxruntime execution providers when loading ONNX models with `mlflow.pyfunc.load_model()` (#5317, @ecm200)
- [Projects] Increase Docker image push timeout when using Projects with Docker (#5363, @zanitete)
- [Python] Fix a bug that prevented users from enabling DEBUG-level Python log outputs (#5362, @dbczumar)
- [Docs] Add a developer guide explaining how to build custom plugins for `mlflow.evaluate()` (#5333, @WeichenXu123)

Small bug fixes and doc updates (#5298, @wamartin-aml; #5399, #5321, #5313, #5307, #5305, #5268, #5284, @harupy; #5329, @Ark-kun; #5375, #5346, #5304, @dbczumar; #5401, #5366, #5345, @BenWilson2; #5326, #5315, @WeichenXu123; #5236, @singankit; #5302, @timvink; #5357, @maitre-matt; #5347, #5344, @mehtayogita; #5367, @apurva-koti; #5348, #5328, #5310, @liangz1; #5267, @sunishsheth2009)

## 1.23.1 (2022-01-27)

MLflow 1.23.1 is a patch release containing the following bug fixes:

- [Models] Fix a directory creation failure when loading PySpark ML models (#5299, @arjundc-db)
- [Model Registry] Revert to using case-insensitive validation logic for stage names in `models:/` URIs (#5312, @lichenran1234)
- [Projects] Fix a race condition during Project tar file creation (#5303, @dbczumar)

## 1.23.0 (2022-01-17)

MLflow 1.23.0 includes several major features and improvements:

Features:

- [Models] Introduce an `mlflow.evaluate()` API for evaluating MLflow Models, providing performance and explainability insights. For an overview, see https://mlflow.org/docs/latest/models.html#model-evaluation (#5069, #5092, #5256, @WeichenXu123)
- [Models] `log_model()` APIs now return information about the logged MLflow Model, including artifact location, flavors, and schema (#5230, @liangz1)
- [Models] Introduce an `mlflow.models.Model.load_input_example()` Python API for loading MLflow Model input examples (#5212, @maitre-matt)
- [Models] Add a UUID field to the MLflow Model specification. MLflow Models now have a unique identifier (#5149, #5167, @WeichenXu123)
- [Models] Support passing SciPy CSC and CSR matrices as MLflow Model input examples (#5016, @WeichenXu123)
- [Model Registry] Support specifying `latest` in model URI to get the latest version of a model regardless of the stage (#5027, @lichenran1234)
- [Tracking] Add support for LightGBM scikit-learn models to `mlflow.lightgbm.autolog()` (#5130, #5200, #5271 @jwyyy)
- [Tracking] Improve S3 artifact download speed by caching boto clients (#4695, @Samreay)
- [UI] Automatically update metric plots for in-progress runs (#5017, @cedkoffeto, @harupy)

Bug fixes and documentation updates:

- [Models] Fix a bug in MLflow Model schema enforcement where strings were incorrectly cast to Pandas objects (#5134, @stevenchen-db)
- [Models] Fix a bug where keyword arguments passed to `mlflow.pytorch.load_model()` were not applied for scripted models (#5163, @schmidt-jake)
- [Model Registry/R] Fix bug in R client `mlflow_create_model_version()` API that caused model `source` to be set incorrectly (#5185, @bramrodenburg)
- [Projects] Fix parsing behavior for Project URIs containing quotes (#5117, @dinaldoap)
- [Scoring] Use the correct 400-level error code for malformed MLflow Model Server requests (#5003, @abatomunkuev)
- [Tracking] Fix a bug where `mlflow.start_run()` modified user-supplied tags dictionary (#5191, @matheusMoreno)
- [UI] Fix a bug causing redundant scroll bars to be displayed on the Experiment Page (#5159, @sunishsheth2009)

Small bug fixes and doc updates (#5275, #5264, #5244, #5249, #5255, #5248, #5243, #5240, #5239, #5232, #5234, #5235, #5082, #5220, #5219, #5226, #5217, #5194, #5188, #5132, #5182, #5183, #5180, #5177, #5165, #5164, #5162, #5015, #5136, #5065, #5125, #5106, #5127, #5120, @harupy; #5045, @BenWilson2; #5156, @pbezglasny; #5202, @jwyyy; #3863, @JoshuaAnickat; #5205, @abhiramr; #4604, @OSobky; #4256, @einsmein; #5140, @AveshCSingh; #5273, #5186, #5176, @WeichenXu123; #5260, #5229, #5206, #5174, #5160, @liangz1)

## 1.22.0 (2021-11-29)

MLflow 1.22.0 includes several major features and improvements:

Features:

- [UI] Add a share button to the Experiment page (#4936, @marijncv)
- [UI] Improve readability of column sorting dropdown on Experiment page (#5022, @WeichenXu123; #5018, @NieuweNils, @coder-freestyle)
- [Tracking] Mark all autologging integrations as stable by removing `@experimental` decorators (#5028, @liangz1)
- [Tracking] Add optional `experiment_id` parameter to `mlflow.set_experiment()` (#5012, @dbczumar)
- [Tracking] Add support for XGBoost scikit-learn models to `mlflow.xgboost.autolog()` (#5078, @jwyyy)
- [Tracking] Improve statsmodels autologging performance by removing unnecessary metrics (#4942, @WeichenXu123)
- [Tracking] Update R client to tag nested runs with parent run ID (#4197, @yitao-li)
- [Models] Support saving and loading all XGBoost model types (#4954, @jwyyy)
- [Scoring] Support specifying AWS account and role when deploying models to SageMaker (#4923, @andresionek91)
- [Scoring] Support serving MLflow models with MLServer (#4963, @adriangonz)

Bug fixes and documentation updates:

- [UI] Fix bug causing Metric Plot page to crash when metric values are too large (#4947, @ianshan0915)
- [UI] Fix bug causing parallel coordinate curves to vanish (#5087, @harupy)
- [UI] Remove `Creator` field from Model Version page if user information is absent (#5089, @jinzhang21)
- [UI] Fix model loading instructions for non-pyfunc models in Artifact Viewer (#5006, @harupy)
- [Models] Fix a bug that added `mlflow` to `conda.yaml` even if a hashed version was already present (#5058, @maitre-matt)
- [Docs] Add Python documentation for metric, parameter, and tag key / value length limits (#4991, @westford14)
- [Examples] Update Python version used in Prophet example to fix installation errors (#5101, @BenWilson2)
- [Examples] Fix Kubernetes `resources` specification in MLflow Projects + Kubernetes example (#4948, @jianyuan)

Small bug fixes and doc updates (#5119, #5107, #5105, #5103, #5085, #5088, #5051, #5081, #5039, #5073, #5072, #5066, #5064, #5063, #5060, #4718, #5053, #5052, #5041, #5043, #5047, #5036, #5037, #5029, #5031, #5032, #5030, #5007, #5019, #5014, #5008, #4998, #4985, #4984, #4970, #4966, #4980, #4967, #4978, #4979, #4968, #4976, #4975, #4934, #4956, #4938, #4950, #4946, #4939, #4913, #4940, #4935, @harupy; #5095, #5070, #5002, #4958, #4945, @BenWilson2; #5099, @chaosddp; #5005, @you-n-g; #5042, #4952, @shrinath-suresh; #4962, #4995, @WeichenXu123; #5010, @lichenran1234; #5000, @wentinghu; #5111, @alexott; #5102, #5024, #5011, #4959, @dbczumar; #5075, #5044, #5026, #4997, #4964, #4989, @liangz1; #4999, @stevenchen-db)

## 1.21.0 (2021-10-23)

MLflow 1.21.0 includes several major features and improvements:

Features:

- [UI] Add a diff-only toggle to the runs table for filtering out columns with constant values (#4862, @marijncv)
- [UI] Add a duration column to the runs table (#4840, @marijncv)
- [UI] Display the default column sorting order in the runs table (#4847, @marijncv)
- [UI] Add `start_time` and `duration` information to exported runs CSV (#4851, @marijncv)
- [UI] Add lifecycle stage information to the run page (#4848, @marijncv)
- [UI] Collapse run page sections by default for space efficiency, limit artifact previews to 50MB (#4917, @dbczumar)
- [Tracking] Introduce autologging capabilities for PaddlePaddle model training (#4751, @jinminhao)
- [Tracking] Add an optional tags field to the CreateExperiment API (#4788, @dbczumar; #4795, @apurva-koti)
- [Tracking] Add support for deleting artifacts from SFTP stores via the `mlflow gc` CLI (#4670, @afaul)
- [Tracking] Support AzureDefaultCredential for authenticating with Azure artifact storage backends (#4002, @marijncv)
- [Models] Upgrade the fastai model flavor to support fastai V2 (`>=2.4.1`) (#4715, @jinzhang21)
- [Models] Introduce an `mlflow.prophet` model flavor for Prophet time series models (#4773, @BenWilson2)
- [Models] Introduce a CLI for publishing MLflow Models to the SageMaker Model Registry (#4669, @jinnig)
- [Models] Print a warning when inferred model dependencies are not available on PyPI (#4891, @dbczumar)
- [Models, Projects] Add `MLFLOW_CONDA_CREATE_ENV_CMD` for customizing Conda environment creation (#4746, @giacomov)

Bug fixes and documentation updates:

- [UI] Fix an issue where column selections made in the runs table were persisted across experiments (#4926, @sunishsheth2009)
- [UI] Fix an issue where the text `null` was displayed in the runs table column ordering dropdown (#4924, @harupy)
- [UI] Fix a bug causing the metric plot view to display NaN values upon click (#4858, @arpitjasa-db)
- [Tracking] Fix a model load failure for paths containing spaces or special characters on UNIX systems (#4890, @BenWilson2)
- [Tracking] Correct a migration issue that impacted usage of MLflow Tracking with SQL Server (#4880, @marijncv)
- [Tracking] Spark datasource autologging tags now respect the maximum allowable size for MLflow Tracking (#4809, @dbczumar)
- [Model Registry] Add previously-missing certificate sources for Model Registry REST API requests (#4731, @ericgosno91)
- [Model Registry] Throw an exception when users supply invalid Model Registry URIs for Databricks (#4877, @yunpark93)
- [Scoring] Fix a schema enforcement error that incorrectly cast date-like strings to datetime objects (#4902, @wentinghu)
- [Docs] Expand the documentation for the MLflow Skinny Client (#4113, @eedeleon)

Small bug fixes and doc updates (#4928, #4919, #4927, #4922, #4914, #4899, #4893, #4894, #4884, #4864, #4823, #4841, #4817, #4796, #4797, #4767, #4768, #4757, @harupy; #4863, #4838, @marijncv; #4834, @ksaur; #4772, @louisguitton; #4801, @twsl; #4929, #4887, #4856, #4843, #4789, #4780, @WeichenXu123; #4769, @Ark-kun; #4898, #4756, @apurva-koti; #4784, @lakshikaparihar; #4855, @ianshan0915; #4790, @eedeleon; #4931, #4857, #4846, 4777, #4748, @dbczumar)

## 1.20.2 (2021-09-03)

MLflow 1.20.2 is a patch release containing the following features and bug fixes:

Features:

- Enabled auto dependency inference in spark flavor in autologging (#4759, @harupy)

Bug fixes and documentation updates:

- Increased MLflow client HTTP request timeout from 10s to 120s (#4764, @jinzhang21)
- Fixed autologging compatibility bugs with TensorFlow and Keras version `2.6.0` (#4766, @dbczumar)

Small bug fixes and doc updates (#4770, @WeichenXu123)

## 1.20.1 (2021-08-26)

MLflow 1.20.1 is a patch release containing the following bug fixes:

- Avoid calling `importlib_metadata.packages_distributions` upon `mlflow.utils.requirements_utils` import (#4741, @dbczumar)
- Avoid depending on `importlib_metadata==4.7.0` (#4740, @dbczumar)

## 1.20.0 (2021-08-25)

MLflow 1.20.0 includes several major features and improvements:

Features:

- Autologging for scikit-learn now records post training metrics when scikit-learn evaluation APIs, such as `sklearn.metrics.mean_squared_error`, are called (#4491, #4628 #4638, @WeichenXu123)
- Autologging for PySpark ML now records post training metrics when model evaluation APIs, such as `Evaluator.evaluate()`, are called (#4686, @WeichenXu123)
- Add `pip_requirements` and `extra_pip_requirements` to `mlflow.*.log_model` and `mlflow.*.save_model` for directly specifying the pip requirements of the model to log / save (#4519, #4577, #4602, @harupy)
- Added `stdMetrics` entries to the training metrics recorded during PySpark CrossValidator autologging (#4672, @WeichenXu123)
- MLflow UI updates:
  1. Improved scalability of the parallel coordinates plot for run performance comparison,
  2. Added support for filtering runs based on their start time on the experiment page,
  3. Added a dropdown for runs table column sorting on the experiment page,
  4. Upgraded the AG Grid plugin, which is used for runs table loading on the experiment page, to version 25.0.0,
  5. Fixed a bug on the experiment page that caused the metrics section of the runs table to collapse when selecting columns from other table sections (#4712, @dbczumar)
- Added support for distributed execution to autologging for PyTorch Lightning (#4717, @dbczumar)
- Expanded R support for Model Registry functionality (#4527, @bramrodenburg)
- Added model scoring server support for defining custom prediction response wrappers (#4611, @Ark-kun)
- `mlflow.*.log_model` and `mlflow.*.save_model` now automatically infer the pip requirements of the model to log / save based on the current software environment (#4518, @harupy)
- Introduced support for running Sagemaker Batch Transform jobs with MLflow Models (#4410, #4589, @YQ-Wang)

Bug fixes and documentation updates:

- Deprecate `requirements_file` argument for `mlflow.*.save_model` and `mlflow.*.log_model` (#4620, @harupy)
- set nextPageToken to null (#4729, @harupy)
- Fix a bug in MLflow UI where the pagination token for run search is not refreshed when switching experiments (#4709, @harupy)
- Fix a bug in the model scoring server that rejected requests specifying a valid `Content-Type` header with the charset parameter (#4609, @Ark-kun)
- Fixed a bug that caused SQLAlchemy backends to exhaust DB connections. (#4663, @arpitjasa-db)
- Improve docker build procedures to raise exceptions if docker builds fail (#4610, @Ark-kun)
- Disable autologging for scikit-learn `cross_val_*` APIs, which are incompatible with autologging (#4590, @WeichenXu123)
- Deprecate MLflow Models support for fast.ai V1 (#4728, @dbczumar)
- Deprecate the old Azure ML deployment APIs `mlflow.azureml.cli.build_image` and `mlflow.azureml.build_image` (#4646, @trangevi)
- Deprecate MLflow Models support for TensorFlow < 2.0 and Keras < 2.3 (#4716, @harupy)

Small bug fixes and doc updates (#4730, #4722, #4725, #4723, #4703, #4710, #4679, #4694, #4707, #4708, #4706, #4705, #4625, #4701, #4700, #4662, #4699, #4682, #4691, #4684, #4683, #4675, #4666, #4648, #4653, #4651, #4641, #4649, #4627, #4637, #4632, #4634, #4621, #4619, #4622, #4460, #4608, #4605, #4599, #4600, #4581, #4583, #4565, #4575, #4564, #4580, #4572, #4570, #4574, #4576, #4568, #4559, #4537, #4542, @harupy; #4698, #4573, @Ark-kun; #4674, @kvmakes; #4555, @vagoston; #4644, @zhengjxu; #4690, #4588, @apurva-koti; #4545, #4631, #4734, @WeichenXu123; #4633, #4292, @shrinath-suresh; #4711, @jinzhang21; #4688, @murilommen; #4635, @ryan-duve; #4724, #4719, #4640, #4639, #4629, #4612, #4613, #4586, @dbczumar)

## 1.19.0 (2021-07-14)

MLflow 1.19.0 includes several major features and improvements:

Features:

- Add support for plotting per-class feature importance computed on linear boosters in XGBoost autologging (#4523, @dbczumar)
- Add `mlflow_create_registered_model` and `mlflow_delete_registered_model` for R to create/delete registered models.
- Add support for setting tags while resuming a run (#4497, @dbczumar)
- MLflow UI updates (#4490, @sunishsheth2009)

  - Add framework for internationalization support.
  - Move metric columns before parameter and tag columns in the runs table.
  - Change the display format of run start time to elapsed time (e.g. 3 minutes ago) from timestamp (e.g. 2021-07-14 14:02:10) in the runs table.

Bug fixes and documentation updates:

- Fix a bug causing MLflow UI to crash when sorting a column containing both `NaN` and empty values (#3409, @harupy)

Small bug fixes and doc updates (#4541, #4534, #4533, #4517, #4508, #4513, #4512, #4509, #4503, #4486, #4493, #4469, @harupy; #4458, @KasirajanA; #4501, @jimmyxu-db; #4521, #4515, @jerrylian-db; #4359, @shrinath-suresh; #4544, @WeichenXu123; #4549, @smurching; #4554, @derkomai; #4506, @tomasatdatabricks; #4551, #4516, #4494, @dbczumar; #4511, @keypointt)

## 1.18.0 (2021-06-18)

MLflow 1.18.0 includes several major features and improvements:

Features:

- Autologging performance improvements for XGBoost, LightGBM, and scikit-learn (#4416, #4473, @dbczumar)
- Add new PaddlePaddle flavor to MLflow Models (#4406, #4439, @jinminhao)
- Introduce paginated ListExperiments API (#3881, @wamartin-aml)
- Include Runtime version for MLflow Models logged on Databricks (#4421, @stevenchen-db)
- MLflow Models now log dependencies in pip requirements.txt format, in addition to existing conda format (#4409, #4422, @stevenchen-db)
- Add support for limiting the number child runs created by autologging for scikit-learn hyperparameter search models (#4382, @mohamad-arabi)
- Improve artifact upload / download performance on Databricks (#4260, @dbczumar)
- Migrate all model dependencies from conda to "pip" section (#4393, @WeichenXu123)

Bug fixes and documentation updates:

- Fix an MLflow UI bug that caused git source URIs to be rendered improperly (#4403, @takabayashi)
- Fix a bug that prevented reloading of MLflow Models based on the TensorFlow SavedModel format (#4223) (#4319, @saschaschramm)
- Fix a bug in the behavior of `KubernetesSubmittedRun.get_status()` for Kubernetes MLflow Project runs (#3962) (#4159, @jcasse)
- Fix a bug in TLS verification for MLflow artifact operations on S3 (#4047, @PeterSulcs)
- Fix a bug causing the MLflow server to crash after deletion of the default experiment (#4352, @asaf400)
- Fix a bug causing `mlflow models serve` to crash on Windows 10 (#4377, @simonvanbernem)
- Fix a crash in runs search when ordering by metric values against the MSSQL backend store (#2551) (#4238, @naor2013)
- Fix an autologging incompatibility issue with TensorFlow 2.5 (#4371, @dbczumar)
- Fix a bug in the `disable_for_unsupported_versions` autologging argument that caused library versions to be incorrectly compared (#4303, @WeichenXu123)

Small bug fixes and doc updates (#4405, @mohamad-arabi; #4455, #4461, #4459, #4464, #4453, #4444, #4449, #4301, #4424, #4418, #4417, #3759, #4398, #4389, #4386, #4385, #4384, #4380, #4373, #4378, #4372, #4369, #4348, #4364, #4363, #4349, #4350, #4174, #4285, #4341, @harupy; #4446, @kHarshit; #4471, @AveshCSingh; #4435, #4440, #4368, #4360, @WeichenXu123; #4431, @apurva-koti; #4428, @stevenchen-db; #4467, #4402, #4261, @dbczumar)

## 1.17.0 (2021-05-07)

MLflow 1.17.0 includes several major features and improvements:

Features:

- Add support for hyperparameter-tuning models to `mlflow.pyspark.ml.autolog()` (#4270, @WeichenXu123)

Bug fixes and documentation updates:

- Fix PyTorch Lightning callback definition for compatibility with PyTorch Lightning 1.3.0 (#4333, @dbczumar)
- Fix a bug in scikit-learn autologging that omitted artifacts for unsupervised models (#4325, @dbczumar)
- Support logging `datetime.date` objects as part of model input examples (#4313, @vperiyasamy)
- Implement HTTP request retries in the MLflow Java client for 500-level responses (#4311, @dbczumar)
- Include a community code of conduct (#4310, @dennyglee)

Small bug fixes and doc updates (#4276, #4263, @WeichenXu123; #4289, #4302, #3599, #4287, #4284, #4265, #4266, #4275, #4268, @harupy; #4335, #4297, @dbczumar; #4324, #4320, @tleyden)

## 1.16.0 (2021-04-22)

MLflow 1.16.0 includes several major features and improvements:

Features:

- Add `mlflow.pyspark.ml.autolog()` API for autologging of `pyspark.ml` estimators (#4228, @WeichenXu123)
- Add `mlflow.catboost.log_model`, `mlflow.catboost.save_model`, `mlflow.catboost.load_model` APIs for CatBoost model persistence (#2417, @harupy)
- Enable `mlflow.pyfunc.spark_udf` to use column names from model signature by default (#4236, @Loquats)
- Add `datetime` data type for model signatures (#4241, @vperiyasamy)
- Add `mlflow.sklearn.eval_and_log_metrics` API that computes and logs metrics for the given scikit-learn model and labeled dataset. (#4218, @alkispoly-db)

Bug fixes and documentation updates:

- Fix a database migration error for PostgreSQL (#4211, @dolfinus)
- Fix autologging silent mode bugs (#4231, @dbczumar)

Small bug fixes and doc updates (#4255, #4252, #4254, #4253, #4242, #4247, #4243, #4237, #4233, @harupy; #4225, @dmatrix; #4206, @mlflow-automation; #4207, @shrinath-suresh; #4264, @WeichenXu123; #3884, #3866, #3885, @ankan94; #4274, #4216, @dbczumar)

## 1.15.0 (2021-03-26)

MLflow 1.15.0 includes several features, bug fixes and improvements. Notably, it includes a number of improvements to MLflow autologging:

Features:

- Add `silent=False` option to all autologging APIs, to allow suppressing MLflow warnings and logging statements during autologging setup and training (#4173, @dbczumar)
- Add `disable_for_unsupported_versions=False` option to all autologging APIs, to disable autologging for versions of ML frameworks that have not been explicitly tested against the current version of the MLflow client (#4119, @WeichenXu123)

Bug fixes:

- Autologged runs are now terminated when execution is interrupted via SIGINT (#4200, @dbczumar)
- The R `mlflow_get_experiment` API now returns the same tag structure as `mlflow_list_experiments` and `mlflow_get_run` (#4017, @lorenzwalthert)
- Fix bug where `mlflow.tensorflow.autolog` would previously mutate the user-specified callbacks list when fitting `tf.keras` models (#4195, @dbczumar)
- Fix bug where SQL-backed MLflow tracking server initialization failed when using the MLflow skinny client (#4161, @eedeleon)
- Model version creation (e.g. via `mlflow.register_model`) now fails if the model version status is not READY (#4114, @ankit-db)

Small bug fixes and doc updates (#4191, #4149, #4162, #4157, #4155, #4144, #4141, #4138, #4136, #4133, #3964, #4130, #4118, @harupy; #4152, @mlflow-automation; #4139, @WeichenXu123; #4193, @smurching; #4029, @architkulkarni; #4134, @xhochy; #4116, @wenleix; #4160, @wentinghu; #4203, #4184, #4167, @dbczumar)

## 1.14.1 (2021-03-01)

MLflow 1.14.1 is a patch release containing the following bug fix:

- Fix issues in handling flexible numpy datatypes in TensorSpec (#4147, @arjundc-db)

## 1.14.0 (2021-02-18)

MLflow 1.14.0 includes several major features and improvements:

- MLflow's model inference APIs (`mlflow.pyfunc.predict`), built-in model serving tools (`mlflow models serve`), and model signatures now support tensor inputs. In particular, MLflow now provides built-in support for scoring PyTorch, TensorFlow, Keras, ONNX, and Gluon models with tensor inputs. For more information, see https://mlflow.org/docs/latest/models.html#deploy-mlflow-models (#3808, #3894, #4084, #4068 @wentinghu; #4041 @tomasatdatabricks, #4099, @arjundc-db)
- Add new `mlflow.shap.log_explainer`, `mlflow.shap.load_explainer` APIs for logging and loading `shap.Explainer` instances (#3989, @vivekchettiar)
- The MLflow Python client is now available with a reduced dependency set via the `mlflow-skinny` PyPI package (#4049, @eedeleon)
- Add new `RequestHeaderProvider` plugin interface for passing custom request headers with REST API requests made by the MLflow Python client (#4042, @jimmyxu-db)
- `mlflow.keras.log_model` now saves models in the TensorFlow SavedModel format by default instead of the older Keras H5 format (#4043, @harupy)
- `mlflow_log_model` now supports logging MLeap models in R (#3819, @yitao-li)
- Add `mlflow.pytorch.log_state_dict`, `mlflow.pytorch.load_state_dict` for logging and loading PyTorch state dicts (#3705, @shrinath-suresh)
- `mlflow gc` can now garbage-collect artifacts stored in S3 (#3958, @sklingel)

Bug fixes and documentation updates:

- Enable autologging for TensorFlow estimators that extend `tensorflow.compat.v1.estimator.Estimator` (#4097, @mohamad-arabi)
- Fix for universal autolog configs overriding integration-specific configs (#4093, @dbczumar)
- Allow `mlflow.models.infer_signature` to handle dataframes containing `pandas.api.extensions.ExtensionDtype` (#4069, @caleboverman)
- Fix bug where `mlflow_restore_run` doesn't propagate the `client` parameter to `mlflow_get_run` (#4003, @yitao-li)
- Fix bug where scoring on served model fails when request data contains a string that looks like URL and pandas version is later than 1.1.0 (#3921, @Secbone)
- Fix bug causing `mlflow_list_experiments` to fail listing experiments with tags (#3942, @lorenzwalthert)
- Fix bug where metrics plots are computed from incorrect target values in scikit-learn autologging (#3993, @mtrencseni)
- Remove redundant / verbose Python event logging message in autologging (#3978, @dbczumar)
- Fix bug where `mlflow_load_model` doesn't load metadata associated to MLflow model flavor in R (#3872, @yitao-li)
- Fix `mlflow.spark.log_model`, `mlflow.spark.load_model` APIs on passthrough-enabled environments against ACL'd artifact locations (#3443, @smurching)

Small bug fixes and doc updates (#4102, #4101, #4096, #4091, #4067, #4059, #4016, #4054, #4052, #4051, #4038, #3992, #3990, #3981, #3949, #3948, #3937, #3834, #3906, #3774, #3916, #3907, #3938, #3929, #3900, #3902, #3899, #3901, #3891, #3889, @harupy; #4014, #4001, @dmatrix; #4028, #3957, @dbczumar; #3816, @lorenzwalthert; #3939, @pauldj54; #3740, @jkthompson; #4070, #3946, @jimmyxu-db; #3836, @t-henri; #3982, @neo-anderson; #3972, #3687, #3922, @eedeleon; #4044, @WeichenXu123; #4063, @yitao-li; #3976, @whiteh; #4110, @tomasatdatabricks; #4050, @apurva-koti; #4100, #4084, @wentinghu; #3947, @vperiyasamy; #4021, @trangevi; #3773, @ankan94; #4090, @jinzhang21; #3918, @danielfrg)

## 1.13.1 (2020-12-30)

MLflow 1.13.1 is a patch release containing bug fixes and small changes:

- Fix bug causing Spark autologging to ignore configuration options specified by `mlflow.autolog()` (#3917, @dbczumar)
- Fix bugs causing metrics to be dropped during TensorFlow autologging (#3913, #3914, @dbczumar)
- Fix incorrect value of optimizer name parameter in autologging PyTorch Lightning (#3901, @harupy)
- Fix model registry database `allow_null_for_run_id` migration failure affecting MySQL databases (#3836, @t-henri)
- Fix failure in `transition_model_version_stage` when uncanonical stage name is passed (#3929, @harupy)
- Fix an undefined variable error causing AzureML model deployment to fail (#3922, @eedeleon)
- Reclassify scikit-learn as a pip dependency in MLflow Model conda environments (#3896, @harupy)
- Fix experiment view crash and artifact view inconsistency caused by artifact URIs with redundant slashes (#3928, @dbczumar)

## 1.13 (2020-12-22)

MLflow 1.13 includes several major features and improvements:

Features:

New fluent APIs for logging in-memory objects as artifacts:

- Add `mlflow.log_text` which logs text as an artifact (#3678, @harupy)
- Add `mlflow.log_dict` which logs a dictionary as an artifact (#3685, @harupy)
- Add `mlflow.log_figure` which logs a figure object as an artifact (#3707, @harupy)
- Add `mlflow.log_image` which logs an image object as an artifact (#3728, @harupy)

UI updates / fixes (#3867, @smurching):

- Add model version link in compact experiment table view
- Add logged/registered model links in experiment runs page view
- Enhance artifact viewer for MLflow models
- Model registry UI settings are now persisted across browser sessions
- Add model version `description` field to model version table

Autologging enhancements:

- Improve robustness of autologging integrations to exceptions (#3682, #3815, dbczumar; #3860, @mohamad-arabi; #3854, #3855, #3861, @harupy)
- Add `disable` configuration option for autologging (#3682, #3815, dbczumar; #3838, @mohamad-arabi; #3854, #3855, #3861, @harupy)
- Add `exclusive` configuration option for autologging (#3851, @apurva-koti; #3869, @dbczumar)
- Add `log_models` configuration option for autologging (#3663, @mohamad-arabi)
- Set tags on autologged runs for easy identification (and add tags to start_run) (#3847, @dbczumar)

More features and improvements:

- Allow Keras models to be saved with `SavedModel` format (#3552, @skylarbpayne)
- Add support for `statsmodels` flavor (#3304, @olbapjose)
- Add support for nested-run in mlflow R client (#3765, @yitao-li)
- Deploying a model using `mlflow.azureml.deploy` now integrates better with the AzureML tracking/registry. (#3419, @trangevi)
- Update schema enforcement to handle integers with missing values (#3798, @tomasatdatabricks)

Bug fixes and documentation updates:

- When running an MLflow Project on Databricks, the version of MLflow installed on the Databricks cluster will now match the version used to run the Project (#3880, @FlorisHoogenboom)
- Fix bug where metrics are not logged for single-epoch `tf.keras` training sessions (#3853, @dbczumar)
- Reject boolean types when logging MLflow metrics (#3822, @HCoban)
- Fix alignment of Keras / `tf.Keras` metric history entries when `initial_epoch` is different from zero. (#3575, @garciparedes)
- Fix bugs in autologging integrations for newer versions of TensorFlow and Keras (#3735, @dbczumar)
- Drop global `filterwwarnings` module at import time (#3621, @jogo)
- Fix bug that caused preexisting Python loggers to be disabled when using MLflow with the SQLAlchemyStore (#3653, @arthury1n)
- Fix `h5py` library incompatibility for exported Keras models (#3667, @tomasatdatabricks)

Small changes, bug fixes and doc updates (#3887, #3882, #3845, #3833, #3830, #3828, #3826, #3825, #3800, #3809, #3807, #3786, #3794, #3731, #3776, #3760, #3771, #3754, #3750, #3749, #3747, #3736, #3701, #3699, #3698, #3658, #3675, @harupy; #3723, @mohamad-arabi; #3650, #3655, @shrinath-suresh; #3850, #3753, #3725, @dmatrix; ##3867, #3670, #3664, @smurching; #3681, @sueann; #3619, @andrewnitu; #3837, @javierluraschi; #3721, @szczeles; #3653, @arthury1n; #3883, #3874, #3870, #3877, #3878, #3815, #3859, #3844, #3703, @dbczumar; #3768, @wentinghu; #3784, @HCoban; #3643, #3649, @arjundc-db; #3864, @AveshCSingh, #3756, @yitao-li)

## 1.12.1 (2020-11-19)

MLflow 1.12.1 is a patch release containing bug fixes and small changes:

- Fix `run_link` for cross-workspace model versions (#3681, @sueann)
- Remove hard dependency on matplotlib for sklearn autologging (#3703, @dbczumar)
- Do not disable existing loggers when initializing alembic (#3653, @arthury1n)

## 1.12.0 (2020-11-10)

MLflow 1.12.0 includes several major features and improvements, in particular a number of improvements to autologging and MLflow's Pytorch integrations:

Features:

Autologging:

- Add universal `mlflow.autolog` which enables autologging for all supported integrations (#3561, #3590, @andrewnitu)
- Add `mlflow.pytorch.autolog` API for automatic logging of metrics, params, and models from Pytorch Lightning training (#3601, @shrinath-suresh, #3636, @karthik-77). This API is also enabled by `mlflow.autolog`.
- Scikit-learn, XGBoost, and LightGBM autologging now support logging model signatures and input examples (#3386, #3403, #3449, @andrewnitu)
- `mlflow.sklearn.autolog` now supports logging metrics (e.g. accuracy) and plots (e.g. confusion matrix heat map) (#3423, #3327, @willzhan-db, @harupy)

PyTorch:

- `mlflow.pytorch.log_model`, `mlflow.pytorch.load_model` now support logging/loading TorchScript models (#3557, @shrinath-suresh)
- `mlflow.pytorch.log_model` supports passing `requirements_file` & `extra_files` arguments to log additional artifacts along with a model (#3436, @shrinath-suresh)

More features and improvements:

- Add `mlflow.shap.log_explanation` for logging model explanations generated by SHAP (#3513, @harupy)
- `log_model` and `create_model_version` now supports an `await_creation_for` argument (#3376, @andychow-db)
- Put preview paths before non-preview paths for backwards compatibility (#3648, @sueann)
- Clean up model registry endpoint and client method definitions (#3610, @sueann)
- MLflow deployments plugin now supports 'predict' CLI command (#3597, @shrinath-suresh)
- Support H2O for R (#3416, @yitao-li)
- Add `MLFLOW_S3_IGNORE_TLS` environment variable to enable skipping TLS verification of S3 endpoint (#3345, @dolfinus)

Bug fixes and documentation updates:

- Ensure that results are synced across distributed processes if ddp enabled (no-op else) (#3651, @SeanNaren)
- Remove optimizer step override to ensure that all accelerator cases are covered by base module (#3635, @SeanNaren)
- Fix `AttributeError` in keras autologgging (#3611, @sephib)
- Scikit-learn autologging: Exclude feature extraction / selection estimator (#3600, @dbczumar)
- Scikit-learn autologging: Fix behavior when a child and its parent are both patched (#3582, @dbczumar)
- Fix a bug where `lightgbm.Dataset(None)` fails after running `mlflow.lightgbm.autolog` (#3594, @harupy)
- Fix a bug where `xgboost.DMatrix(None)` fails after running `mlflow.xgboost.autolog` (#3584, @harupy)
- Pass `docker_args` in non-synchronous mlflow project runs (#3563, @alfozan)
- Fix a bug of `FTPArtifactRepository.log_artifacts` with `artifact_path` keyword argument (issue #3388) (#3391, @kzm4269)
- Exclude preprocessing & imputation steps from scikit-learn autologging (#3491, @dbczumar)
- Fix duplicate stderr logging during artifact logging and project execution in the R client (#3145, @yitao-li)
- Don't call `atexit.register(_flush_queue)` in `__main__` scope of `mlflow/tensorflow.py` (#3410, @harupy)
- Fix for restarting terminated run not setting status correctly (#3329, @apurva-koti)
- Fix model version run_link URL for some Databricks regions (#3417, @sueann)
- Skip JSON validation when endpoint is not MLflow REST API (#3405, @harupy)
- Document `mlflow-torchserve` plugin (#3634, @karthik-77)
- Add `mlflow-elasticsearchstore` to the doc (#3462, @AxelVivien25)
- Add code snippets for fluent and MlflowClient APIs (#3385, #3437, #3489 #3573, @dmatrix)
- Document `mlflow-yarn` backend (#3373, @fhoering)
- Fix a breakage in loading Tensorflow and Keras models (#3667, @tomasatdatabricks)

Small bug fixes and doc updates (#3607, #3616, #3534, #3598, #3542, #3568, #3349, #3554, #3544, #3541, #3533, #3535, #3516, #3512, #3497, #3522, #3521, #3492, #3502, #3434, #3422, #3394, #3387, #3294, #3324, #3654, @harupy; #3451, @jgc128; #3638, #3632, #3608, #3452, #3399, @shrinath-suresh; #3495, #3459, #3662, #3668, #3670 @smurching; #3488, @edgan8; #3639, @karthik-77; #3589, #3444, #3276, @lorenzwalthert; #3538, #3506, #3509, #3507, #3510, #3508, @rahulporuri; #3504, @sbrugman; #3486, #3466, @apurva-koti; #3477, @juntai-zheng; #3617, #3609, #3605, #3603, #3560, @dbczumar; #3411, @danielvdende; #3377, @willzhan-db; #3420, #3404, @andrewnitu; #3591, @mateiz; #3465, @abawchen; #3543, @emptalk; #3302, @bramrodenburg; #3468, @ghisvail; #3496, @extrospective; #3549, #3501, #3435, @yitao-li; #3243, @OlivierBondu; #3439, @andrewnitu; #3651, #3635 @SeanNaren, #3470, @ankit-db)

## 1.11.0 (2020-08-31)

MLflow 1.11.0 includes several major features and improvements:

Features:

- New `mlflow.sklearn.autolog()` API for automatic logging of metrics, params, and models from scikit-learn model training (#3287, @harupy; #3323, #3358 @dbczumar)
- Registered model & model version creation APIs now support specifying an initial `description` (#3271, @sueann)
- The R `mlflow_log_model` and `mlflow_load_model` APIs now support XGBoost models (#3085, @lorenzwalthert)
- New `mlflow.list_run_infos` fluent API for listing run metadata (#3183, @trangevi)
- Added section for visualizing and comparing model schemas to model version and model-version-comparison UIs (#3209, @zhidongqu-db)
- Enhanced support for using the model registry across Databricks workspaces: support for registering models to a Databricks workspace from outside the workspace (#3119, @sueann), tracking run-lineage of these models (#3128, #3164, @ankitmathur-db; #3187, @harupy), and calling `mlflow.<flavor>.load_model` against remote Databricks model registries (#3330, @sueann)
- UI support for setting/deleting registered model and model version tags (#3187, @harupy)
- UI support for archiving existing staging/production versions of a model when transitioning a new model version to staging/production (#3134, @harupy)

Bug fixes and documentation updates:

- Fixed parsing of MLflow project parameter values containing'=' (#3347, @dbczumar)
- Fixed a bug preventing listing of WASBS artifacts on the latest version of Azure Blob Storage (12.4.0) (#3348, @dbczumar)
- Fixed a bug where artifact locations become malformed when using an SFTP file store in Windows (#3168, @harupy)
- Fixed bug where `list_artifacts` returned incorrect results on GCS, preventing e.g. loading SparkML models from GCS (#3242, @santosh1994)
- Writing and reading artifacts via `MlflowClient` to a DBFS location in a Databricks tracking server specified through the `tracking_uri` parameter during the initialization of `MlflowClient` now works properly (#3220, @sueann)
- Fixed bug where `FTPArtifactRepository` returned artifact locations as absolute paths, rather than paths relative to the artifact repository root (#3210, @shaneing), and bug where calling `log_artifacts` against an FTP artifact location copied the logged directory itself into the FTP location, rather than the contents of the directory.
- Fixed bug where Databricks project execution failed due to passing of GET request params as part of the request body rather than as query parameters (#2947, @cdemonchy-pro)
- Fix bug where artifact viewer did not correctly render PDFs in MLflow 1.10 (#3172, @ankitmathur-db)
- Fixed parsing of `order_by` arguments to MLflow search APIs when ordering by fields whose names contain spaces (#3118, @jdlesage)
- Fixed bug where MLflow model schema enforcement raised exceptions when validating string columns using pandas >= 1.0 (#3130, @harupy)
- Fixed bug where `mlflow.spark.log_model` did not save model signature and input examples (#3151, @harupy)
- Fixed bug in runs UI where tags table did not reflect deletion of tags. (#3135, @ParseDark)
- Added example illustrating the use of RAPIDS with MLFlow (#3028, @drobison00)

Small bug fixes and doc updates (#3326, #3344, #3314, #3289, #3225, #3288, #3279, #3265, #3263, #3260, #3255, #3267, #3266, #3264, #3256, #3253, #3231, #3245, #3191, #3238, #3192, #3188, #3189, #3180, #3178, #3166, #3181, #3142, #3165, #2960, #3129, #3244, #3359 @harupy; #3236, #3141, @AveshCSingh; #3295, #3163, @arjundc-db; #3241, #3200, @zhidongqu-db; #3338, #3275, @sueann; #3020, @magnus-m; #3322, #3219, @dmatrix; #3341, #3179, #3355, #3360, #3363 @smurching; #3124, @jdlesage; #3232, #3146, @ankitmathur-db; #3140, @andreakress; #3174, #3133, @mlflow-automation; #3062, @cafeal; #3193, @tomasatdatabricks; 3115, @fhoering; #3328, @apurva-koti; #3046, @OlivierBondu; #3194, #3158, @dmatrix; #3250, @shivp950; #3259, @simonhessner; #3357 @dbczumar)

## 1.10.0 (2020-07-20)

MLflow 1.10.0 includes several major features and improvements, in particular the release of several new model registry Python client APIs.

Features:

- `MlflowClient.transition_model_version_stage` now supports an
  `archive_existing_versions` argument for archiving existing staging or production model
  versions when transitioning a new model version to staging or production (#3095, @harupy)
- Added `set_registry_uri`, `get_registry_uri` APIs. Setting the model registry URI causes
  fluent APIs like `mlflow.register_model` to communicate with the model registry at the specified
  URI (#3072, @sueann)
- Added paginated `MlflowClient.search_registered_models` API (#2939, #3023, #3027 @ankitmathur-db; #2966, @mparkhe)
- Added syntax highlighting when viewing text files (YAML etc) in the MLflow runs UI (#3041, @harupy)
- Added REST API and Python client support for setting and deleting tags on model versions and registered models,
  via the `MlflowClient.create_registered_model`, `MlflowClient.create_model_version`,
  `MlflowClient.set_registered_model_tag`, `MlflowClient.set_model_version_tag`,
  `MlflowClient.delete_registered_model_tag`, and `MlflowClient.delete_model_version_tag` APIs (#3094, @zhidongqu-db)

Bug fixes and documentation updates:

- Removed usage of deprecated `aws ecr get-login` command in `mlflow.sagemaker` (#3036, @mrugeles)
- Fixed bug where artifacts could not be viewed and downloaded from the artifact UI when using
  Azure Blob Storage (#3014, @Trollgeir)
- Databricks credentials are now propagated to the project subprocess when running MLflow projects
  within a notebook (#3035, @smurching)
- Added docs explaining how to fetching an MLflow model from the model registry (#3000, @andychow-db)

Small bug fixes and doc updates (#3112, #3102, #3089, #3103, #3096, #3090, #3049, #3080, #3070, #3078, #3083, #3051, #3050, #2875, #2982, #2949, #3121 @harupy; #3082, @ankitmathur-db; #3084, #3019, @smurching)

## 1.9.1 (2020-06-25)

MLflow 1.9.1 is a patch release containing a number of bug-fixes and improvements:

Bug fixes and improvements:

- Fixes `AttributeError` when pickling an instance of the Python `MlflowClient` class (#2955, @Polyphenolx)
- Fixes bug that prevented updating model-version descriptions in the model registry UI (#2969, @AnastasiaKol)
- Fixes bug where credentials were not properly propagated to artifact CLI commands when logging artifacts from Java to the DatabricksArtifactRepository (#3001, @dbczumar)
- Removes use of new Pandas API in new MLflow model-schema functionality, so that it can be used with older Pandas versions (#2988, @aarondav)

Small bug fixes and doc updates (#2998, @dbczumar; #2999, @arjundc-db)

## 1.9.0 (2020-06-19)

MLflow 1.9.0 includes numerous major features and improvements, and a breaking change to
experimental APIs:

Breaking Changes:

- The `new_name` argument to `MlflowClient.update_registered_model`
  has been removed. Call `MlflowClient.rename_registered_model` instead. (#2946, @mparkhe)
- The `stage` argument to `MlflowClient.update_model_version`
  has been removed. Call `MlflowClient.transition_model_version_stage` instead. (#2946, @mparkhe)

Features (MLflow Models and Flavors)

- `log_model` and `save_model` APIs now support saving model signatures (the model's input and output schema)
  and example input along with the model itself (#2698, #2775, @tomasatdatabricks). Model signatures are used
  to reorder and validate input fields when scoring/serving models using the pyfunc flavor, `mlflow models`
  CLI commands, or `mlflow.pyfunc.spark_udf` (#2920, @tomasatdatabricks and @aarondav)
- Introduce fastai model persistence and autologging APIs under `mlflow.fastai` (#2619, #2689 @antoniomdk)
- Add pluggable `mlflow.deployments` API and CLI for deploying models to custom serving tools, e.g. RedisAI
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
  `KUBE_MLFLOW_TRACKING_URI` for passing a different tracking server to the kubernetes job (#2874, @catapulta)

Features (UI)

- Significant performance and scalability improvements to metric comparison and scatter plots in
  the UI (#2447, @mjlbach)
- The main MLflow experiment list UI now includes a link to the model registry UI (#2805, @zhidongqu-db),
- Enable viewing PDFs logged as artifacts from the runs UI (#2859, @ankmathur96)
- UI accessibility improvements: better color contrast (#2872, @Zangr), add child roles to DOM elements (#2871, @Zangr)

Features (Tracking Client and Server)

- Adds ability to pass client certs as part of REST API requests when using the tracking or model
  registry APIs. (#2843, @PhilipMay)
- New community plugin: support for storing artifacts in Aliyun (Alibaba Cloud) (#2917, @SeaOfOcean)
- Infer and set content type and encoding of objects when logging models and artifacts to S3 (#2881, @hajapy)
- Adds support for logging artifacts to HDFS Federation ViewFs (#2782, @fhoering)
- Add healthcheck endpoint to the MLflow server at `/health` (#2725, @crflynn)
- Improves performance of default file-based tracking storage backend by using LibYAML (if installed)
  to read experiment and run metadata (#2707, @Higgcz)

Bug fixes and documentation updates:

- Several UI fixes: remove margins around icon buttons (#2827, @harupy),
  fix alignment issues in metric view (#2811, @zhidongqu-db), add handling of `NaN`
  values in metrics plot (#2773, @dbczumar), truncate run ID in the run name when
  comparing multiple runs (#2508, @harupy)
- Database engine URLs are no longer logged when running `mlflow db upgrade` (#2849, @hajapy)
- Updates `log_artifact`, `log_model` APIs to consistently use posix paths, rather than OS-dependent
  paths, when computing artifact subpaths. (#2784, @mikeoconnor0308)
- Fix `ValueError` when scoring `tf.keras` 1.X models using `mlflow.pyfunc.predict` (#2762, @juntai-zheng)
- Fixes conda environment activation bug when running MLflow projects on Windows (#2731, @MynherVanKoek)
- `mlflow.end_run` will now clear the active run even if the run cannot be marked as
  terminated (e.g. because it's been deleted), (#2693, @ahmed-shariff)
- Add missing documentation for `mlflow.spacy` APIs (#2771, @harupy)

Small bug fixes and doc updates (#2919, @willzhan-db; #2940, #2942, #2941, #2943, #2927, #2929, #2926, #2914, #2928, #2913, #2852, #2876, #2808, #2810, #2442, #2780, #2758, #2732, #2734, #2431, #2733, #2716, @harupy; #2915, #2897, @jwgwalton; #2856, @jkthompson; #2962, @hhsecond; #2873, #2829, #2582, @dmatrix; #2908, #2865, #2880, #2866, #2833, #2785, #2723, @smurching; #2906, @dependabot[bot]; #2724, @aarondav; #2896, @ezeeetm; #2741, #2721, @mlflow-automation; #2864, @tallen94; #2726, @crflynn; #2710, #2951 @mparkhe; #2935, #2921, @ankitmathur-db; #2963, #2739, @dbczumar; #2853, @stat4jason; #2709, #2792, @juntai-zheng @juntai-zheng; #2749, @HiromuHota; #2957, #2911, #2718, @arjundc-db; #2885, @willzhan-db; #2803, #2761, @pogil; #2392, @jnmclarty; #2794, @Zethson; #2766, #2916 @shubham769)

## 1.8.0 (2020-04-16)

MLflow 1.8.0 includes several major features and improvements:

Features:

- Added `mlflow.azureml.deploy` API for deploying MLflow models to AzureML (#2375 @csteegz, #2711, @akshaya-a)
- Added support for case-sensitive LIKE and case-insensitive ILIKE queries (e.g. `'params.framework LIKE '%sklearn%'`) with the SearchRuns API & UI when running against a SQLite backend (#2217, @t-henri; #2708, @mparkhe)
- Improved line smoothing in MLflow metrics UI using exponential moving averages (#2620, @Valentyn1997)
- Added `mlflow.spacy` module with support for logging and loading spaCy models (#2242, @arocketman)
- Parameter values that differ across runs are highlighted in run comparison UI (#2565, @gabrielbretschner)
- Added ability to compare source runs associated with model versions from the registered model UI (#2537, @juntai-zheng)
- Added support for alphanumerical experiment IDs in the UI. (#2568, @jonas)
- Added support for passing arguments to `docker run` when running docker-based MLflow projects (#2608, @ksanjeevan)
- Added Windows support for `mlflow sagemaker build-and-push-container` CLI & API (#2500, @AndreyBulezyuk)
- Improved performance of reading experiment data from local filesystem when LibYAML is installed (#2707, @Higgcz)
- Added a healthcheck endpoint to the REST API server at `/health` that always returns a 200 response status code, to be used to verify health of the server (#2725, @crflynn)
- MLflow metrics UI plots now scale to rendering thousands of points using scattergl (#2447, @mjlbach)

Bug fixes:

- Fixed CLI summary message in `mlflow azureml build_image` CLI (#2712, @dbczumar)
- Updated `examples/flower_classifier/score_images_rest.py` with multiple bug fixes (#2647, @tfurmston)
- Fixed pip not found error while packaging models via `mlflow models build-docker` (#2699, @HiromuHota)
- Fixed bug in `mlflow.tensorflow.autolog` causing erroneous deletion of TensorBoard logging directory (#2670, @dbczumar)
- Fixed a bug that truncated the description of the `mlflow gc` subcommand in `mlflow --help` (#2679, @dbczumar)
- Fixed bug where `mlflow models build-docker` was failing due to incorrect Miniconda download URL (#2685, @michaeltinsley)
- Fixed a bug in S3 artifact logging functionality where `MLFLOW_S3_ENDPOINT_URL` was ignored (#2629, @poppash)
- Fixed a bug where Sqlite in-memory was not working as a tracking backend store by modifying DB upgrade logic (#2667, @dbczumar)
- Fixed a bug to allow numerical parameters with values >= 1000 in R `mlflow::mlflow_run()` API (#2665, @lorenzwalthert)
- Fixed a bug where AWS creds was not found in the Windows platform due path differences (#2634, @AndreyBulezyuk)
- Fixed a bug to add pip when necessary in `_mlflow_conda_env` (#2646, @tfurmston)
- Fixed error code to be more meaningful if input to model version is incorrect (#2625, @andychow-db)
- Fixed multiple bugs in model registry (#2638, @aarondav)
- Fixed support for conda env dicts with `mlflow.pyfunc.log_model` (#2618, @dbczumar)
- Fixed a bug where hiding the start time column in the UI would also hide run selection checkboxes (#2559, @harupy)

Documentation updates:

- Added links to source code to mlflow.org (#2627, @harupy)
- Documented fix for pandas-records payload (#2660, @SaiKiranBurle)
- Fixed documentation bug in TensorFlow `load_model` utility (#2666, @pogil)
- Added the missing Model Registry description and link on the first page (#2536, @dmatrix)
- Added documentation for expected datatype for step argument in `log_metric` to match REST API (#2654, @mparkhe)
- Added usage of the model registry to the `log_model` function in `sklearn_elasticnet_wine/train.py` example (#2609, @netanel246)

Small bug fixes and doc updates (#2594, @Trollgeir; #2703,#2709, @juntai-zheng; #2538, #2632, @keigohtr; #2656, #2553, @lorenzwalthert; #2622, @pingsutw; #2615, #2600, #2533, @mlflow-automation; #1391, @sueann; #2613, #2598, #2534, #2723, @smurching; #2652, #2710, @mparkhe; #2706, #2653, #2639, @tomasatdatabricks; #2611, @9dogs; #2700, #2705, @aarondav; #2675, #2540, @mengxr; #2686, @RensDimmendaal; #2694, #2695, #2532, @dbczumar; #2733, #2716, @harupy; #2726, @crflynn; #2582, #2687, @dmatrix)

## 1.7.2 (2020-03-20)

MLflow 1.7.2 is a patch release containing a minor change:

- Pin alembic version to 1.4.1 or below to prevent pep517-related installation errors
  (#2612, @smurching)

## 1.7.1 (2020-03-17)

MLflow 1.7.1 is a patch release containing bug fixes and small changes:

- Remove usage of Nonnull annotations and findbugs dependency in Java package (#2583, @mparkhe)
- Add version upper bound (<=1.3.13) to sqlalchemy dependency in Python package (#2587, @smurching)

Other bugfixes and doc updates (#2595, @mparkhe; #2567, @jdlesage)

## 1.7.0 (2020-03-02)

MLflow 1.7.0 includes several major features and improvements, and some notable breaking changes:

MLflow support for Python 2 is now deprecated and will be dropped in a future release. At that
point, existing Python 2 workflows that use MLflow will continue to work without modification, but
Python 2 users will no longer get access to the latest MLflow features and bugfixes. We recommend
that you upgrade to Python 3 - see https://docs.python.org/3/howto/pyporting.html for a migration
guide.

Breaking changes to Model Registry REST APIs:

Model Registry REST APIs have been updated to be more consistent with the other MLflow APIs. With
this release Model Registry APIs are intended to be stable until the next major version.

- Python and Java client APIs for Model Registry have been updated to use the new REST APIs. When using an MLflow client with a server using updated REST endpoints, you won't need to change any code but will need to upgrade to a new client version. The client APIs contain deprecated arguments, which for this release are backward compatible, but will be dropped in future releases. (#2457, @tomasatdatabricks; #2502, @mparkhe).
- The Model Registry UI has been updated to use the new REST APIs (#2476 @aarondav; #2507, @mparkhe)

Other Features:

- Ability to click through to individual runs from metrics plot (#2295, @harupy)
- Added `mlflow gc` CLI for permanent deletion of runs (#2265, @t-henri)
- Metric plot state is now captured in page URLs for easier link sharing (#2393, #2408, #2498 @smurching; #2459, @harupy)
- Added experiment management to MLflow UI (create/rename/delete experiments) (#2348, @ggliem)
- Ability to search for experiments by name in the UI (#2324, @ggliem)
- MLflow UI page titles now reflect the content displayed on the page (#2420, @AveshCSingh)
- Added a new `LogModel` REST API endpoint for capturing model metadata, and call it from the Python and R clients (#2369, #2430, #2468 @tomasatdatabricks)
- Java Client API to download model artifacts from Model Registry (#2308, @andychow-db)

Bug fixes and documentation updates:

- Updated Model Registry documentation page with code snippets and examples (#2493, @dmatrix; #2517, @harupy)
- Better error message for Model Registry, when using incompatible backend server (#2456, @aarondav)
- matplotlib is no longer required to use XGBoost and LightGBM autologging (#2423, @harupy)
- Fixed bug where matplotlib figures were not closed in XGBoost and LightGBM autologging (#2386, @harupy)
- Fixed parameter reading logic to support param values with newlines in FileStore (#2376, @dbczumar)
- Improve readability of run table column selector nodes (#2388, @dbczumar)
- Validate experiment name supplied to `UpdateExperiment` REST API endpoint (#2357, @ggliem)
- Fixed broken MLflow DB README link in CLI docs (#2377, @dbczumar)
- Change copyright year across docs to 2020 (#2349, @ParseThis)

Small bug fixes and doc updates (#2378, #2449, #2402, #2397, #2391, #2387, #2523, #2527 @harupy; #2314, @juntai-zheng; #2404, @andychow-db; #2343, @pogil; #2366, #2370, #2364, #2356, @AveshCSingh; #2373, #2365, #2363, @smurching; #2358, @jcuquemelle; #2490, @RensDimmendaal; #2506, @dbczumar; #2234 @Zangr; #2359 @lbernickm; #2525, @mparkhe)

## 1.6.0 (2020-01-29)

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

## 1.5.0 (2019-12-19)

MLflow 1.5.0 includes several major features and improvements:

New Model Flavors and Flavor Updates:

- New support for a LightGBM flavor (#2136, @harupy)
- New support for a XGBoost flavor (#2124, @harupy)
- New support for a Gluon flavor and autologging (#1973, @cosmincatalin)
- Runs automatically created by `mlflow.tensorflow.autolog()` and `mlflow.keras.autolog()` (#2088) are now automatically ended after training and/or exporting your model. See the [`docs`](https://mlflow.org/docs/latest/tracking.html#automatic-logging-from-tensorflow-and-keras-experimental) for more details (#2094, @juntai-zheng)

More features and improvements:

- When using the `mlflow server` CLI command, you can now expose metrics on `/metrics` for Prometheus via the optional --activate-parameter argument (#2097, @t-henri)
- The `mlflow ui` CLI command now has a `--host`/`-h` option to specify user-input IPs to bind to (#2176, @gandroz)
- MLflow now supports pulling Git submodules while using MLflow Projects (#2103, @badc0re)
- New `mlflow models prepare-env` command to do any preparation necessary to initialize an environment. This allows distinguishing configuration and user errors during predict/serve time (#2040, @aarondav)
- TensorFlow.Keras and Keras parameters are now logged by `autolog()` (#2119, @juntai-zheng)
- MLflow `log_params()` will recognize Spark ML params as keys and will now extract only the name attribute (#2064, @tomasatdatabricks)
- Exposes `mlflow.tracking.is_tracking_uri_set()` (#2026, @fhoering)
- The artifact image viewer now displays "Loading..." when it is loading an image (#1958, @harupy)
- The artifact image view now supports animated GIFs (#2070, @harupy)
- Adds ability to mount volumes and specify environment variables when using mlflow with docker (#1994, @nlml)
- Adds run context for detecting job information when using MLflow tracking APIs within Databricks Jobs. The following job types are supported: notebook jobs, Python Task jobs (#2205, @dbczumar)
- Performance improvement when searching for runs (#2030, #2059, @jcuquemelle; #2195, @rom1504)

Bug fixes and documentation updates:

- Fixed handling of empty directories in FS based artifact repositories (#1891, @tomasatdatabricks)
- Fixed `mlflow.keras.save_model()` usage with DBFS (#2216, @andychow-db)
- Fixed several build issues for the Docker image (#2107, @jimthompson5802)
- Fixed `mlflow_list_artifacts()` (R package) (#2200, @lorenzwalthert)
- Entrypoint commands of Kubernetes jobs are now shell-escaped (#2160, @zanitete)
- Fixed project run Conda path issue (#2147, @Zangr)
- Fixed spark model load from model repository (#2175, @tomasatdatabricks)
- Stripped "dev" suffix from PySpark versions (#2137, @dbczumar)
- Fixed note editor on the experiment page (#2054, @harupy)
- Fixed `models serve`, `models predict` CLI commands against models:/ URIs (#2067, @smurching)
- Don't unconditionally format values as metrics in generic HtmlTableView component (#2068, @smurching)
- Fixed remote execution from Windows using posixpath (#1996, @aestene)
- Add XGBoost and LightGBM examples (#2186, @harupy)
- Add note about active run instantiation side effect in fluent APIs (#2197, @andychow-db)
- The tutorial page has been refactored to be be a 'Tutorials and Examples' page (#2182, @juntai-zheng)
- Doc enhancements for XGBoost and LightGBM flavors (#2170, @harupy)
- Add doc for XGBoost flavor (#2167, @harupy)
- Updated `active_run()` docs to clarify it cannot be used accessing current run data (#2138, @juntai-zheng)
- Document models:/ scheme for URI for load_model methods (#2128, @stbof)
- Added an example using Prophet via pyfunc (#2043, @dr3s)
- Added and updated some screenshots and explicit steps for the model registry (#2086, @stbof)

Small bug fixes and doc updates (#2142, #2121, #2105, #2069, #2083, #2061, #2022, #2036, #1972, #2034, #1998, #1959, @harupy; #2202, @t-henri; #2085, @stbof; #2098, @AdamBarnhard; #2180, #2109, #1977, #2039, #2062, @smurching; #2013, @aestene; #2146, @joelcthomas; #2161, #2120, #2100, #2095, #2088, #2076, #2057, @juntai-zheng; #2077, #2058, #2027, @sueann; #2149, @zanitete; #2204, #2188, @andychow-db; #2110, #2053, @jdlesage; #2003, #1953, #2004, @Djailla; #2074, @nlml; #2116, @Silas-Asamoah; #1104, @jimthompson5802; #2072, @cclauss; #2221, #2207, #2157, #2132, #2114, #2063, #2065, #2055, @dbczumar; #2033, @cthoyt; #2048, @philip-khor; #2002, @jspoorta; #2000, @christang; #2078, @dennyglee; #1986, @vguerra; #2020, @dependabot[bot])

## 1.4.0 (2019-10-30)

MLflow 1.4.0 includes several major features:

- Model Registry (Beta). Adds an experimental model registry feature, where you can manage, version, and keep lineage of your production models. (#1943, @mparkhe, @Zangr, @sueann, @dbczumar, @smurching, @gioa, @clemens-db, @pogil, @mateiz; #1988, #1989, #1995, #2021, @mparkhe; #1983, #1982, #1967, @dbczumar)
- TensorFlow updates

  - MLflow Keras model saving, loading, and logging has been updated to be compatible with TensorFlow 2.0. (#1927, @juntai-zheng)
  - Autologging for `tf.estimator` and `tf.keras` models has been updated to be compatible with TensorFlow 2.0. The same functionalities of autologging in TensorFlow 1.x are available in TensorFlow 2.0, namely when fitting `tf.keras` models and when exporting saved `tf.estimator` models. (#1910, @juntai-zheng)
  - Examples and READMEs for both TensorFlow 1.X and TensorFlow 2.0 have been added to `mlflow/examples/tensorflow`. (#1946, @juntai-zheng)

More features and improvements:

- [API] Add functions `get_run`, `get_experiment`, `get_experiment_by_name` to the fluent API (#1923, @fhoering)
- [UI] Use Plotly as artifact image viewer, which allows zooming and panning (#1934, @harupy)
- [UI] Support deleting tags from the run details page (#1933, @harupy)
- [UI] Enable scrolling to zoom in metric and run comparison plots (#1929, @harupy)
- [Artifacts] Add support of viewfs URIs for HDFS federation for artifacts (#1947, @t-henri)
- [Models] Spark UDFs can now be called with struct input if the underlying spark implementation supports it. The data is passed as a pandas DataFrame with column names matching those in the struct. (#1882, @tomasatdatabricks)
- [Models] Spark models will now load faster from DFS by skipping unnecessary copies (#2008, @tomasatdatabricks)

Bug fixes and documentation updates:

- [Projects] Make detection of `MLproject` files case-insensitive (#1981, @smurching)
- [UI] Fix a bug where viewing metrics containing forward-slashes in the name would break the MLflow UI (#1968, @smurching)
- [CLI] `models serve` command now works in Windows (#1949, @rboyes)
- [Scoring] Fix a dependency installation bug in Java MLflow model scoring server (#1913, @smurching)

Small bug fixes and doc updates (#1932, #1935, @harupy; #1907, @marnixkoops; #1911, @HackyRoot; #1931, @jmcarp; #2007, @deniskovalenko; #1966, #1955, #1952, @Djailla; #1915, @sueann; #1978, #1894, @smurching; #1940, #1900, #1904, @mparkhe; #1914, @jerrygb; #1857, @mengxr; #2009, @dbczumar)

## 1.3 (2019-09-30)

MLflow 1.3.0 includes several major features and improvements:

Features:

- The Python client now supports logging & loading models using TensorFlow 2.0 (#1872, @juntai-zheng)
- Significant performance improvements when fetching runs and experiments in MLflow servers that use SQL database-backed storage (#1767, #1878, #1805 @dbczumar)
- New `GetExperimentByName` REST API endpoint, used in the Python client to speed up `set_experiment` and `get_experiment_by_name` (#1775, @smurching)
- New `mlflow.delete_run`, `mlflow.delete_experiment` fluent APIs in the Python client(#1396, @MerelTheisenQB)
- New CLI command (`mlflow experiments csv`) to export runs of an experiment into a CSV (#1705, @jdlesage)
- Directories can now be logged as artifacts via `mlflow.log_artifact` in the Python fluent API (#1697, @apurva-koti)
- HTML and geojson artifacts are now rendered in the run UI (#1838, @sim-san; #1803, @spadarian)
- Keras autologging support for `fit_generator` Keras API (#1757, @charnger)
- MLflow models packaged as docker containers can be executed via Google Cloud Run (#1778, @ngallot)
- Artifact storage configurations are propagated to containers when executing docker-based MLflow projects locally (#1621, @nlaille)
- The Python, Java, R clients and UI now retry HTTP requests on 429 (Too Many Requests) errors (#1846, #1851, #1858, #1859 @tomasatdatabricks; #1847, @smurching)

Bug fixes and documentation updates:

- The R `mlflow_list_artifact` API no longer throws when listing artifacts for an empty run (#1862, @smurching)
- Fixed a bug preventing running the MLflow server against an MS SQL database (#1758, @sifanLV)
- MLmodel files (artifacts) now correctly display in the run UI (#1819, @ankitmathur-db)
- The Python `mlflow.start_run` API now throws when resuming a run whose experiment ID differs from the
  active experiment ID set via `mlflow.set_experiment` (#1820, @mcminnra).
- `MlflowClient.log_metric` now logs metric timestamps with millisecond (as opposed to second) resolution (#1804, @ustcscgyer)
- Fixed bugs when listing (#1800, @ahutterTA) and downloading (#1890, @jdlesage) artifacts stored in HDFS.
- Fixed a bug preventing Kubernetes Projects from pushing to private Docker repositories (#1788, @dbczumar)
- Fixed a bug preventing deploying Spark models to AzureML (#1769, @Ben-Epstein)
- Fixed experiment id resolution in projects (#1715, @drewmcdonald)
- Updated parallel coordinates plot to show all fields available in compared runs (#1753, @mateiz)
- Streamlined docs for getting started with hosted MLflow (#1834, #1785, #1860 @smurching)

Small bug fixes and doc updates (#1848, @pingsutw; #1868, @iver56; #1787, @apurvakoti; #1741, #1737, @apurva-koti; #1876, #1861, #1852, #1801, #1754, #1726, #1780, #1807 @smurching; #1859, #1858, #1851, @tomasatdatabricks; #1841, @ankitmathur-db; #1744, #1746, #1751, @mateiz; #1821, #1730, @dbczumar; #1727, cfmcgrady; #1716, @axsaucedo; #1714, @fhoering; #1405, @ancasarb; #1502, @jimthompson5802; #1720, jke-zq; #1871, @mehdi254; #1782, @stbof)

## 1.2 (2019-08-09)

MLflow 1.2 includes the following major features and improvements:

- Experiments now have editable tags and descriptions (#1630, #1632, #1678, @ankitmathur-db)
- Search latency has been significantly reduced in the SQLAlchemyStore (#1660, @t-henri)

**More features and improvements**

- Backend stores now support run tag values up to 5000 characters in length. Some store implementations may support longer tag values (#1687, @ankitmathur-db)
- Gunicorn options can now be configured for the `mlflow models serve` CLI with the `GUNICORN_CMD_ARGS` environment variable (#1557, @LarsDu)
- Jsonnet artifacts can now be previewed in the UI (#1683, @ankitmathur-db)
- Adds an optional `python_version` argument to `mlflow_install` for specifying the Python version (e.g. "3.5") to use within the conda environment created for installing the MLflow CLI. If `python_version` is unspecified, `mlflow_install` defaults to using Python 3.6. (#1722, @smurching)

**Bug fixes and documentation updates**

- [Tracking] The Autologging feature is now more resilient to tracking errors (#1690, @apurva-koti)
- [Tracking] The `runs` field in in the `GetExperiment.Response` proto has been deprecated & will be removed in MLflow 2.0. Please use the `Search Runs` API for fetching runs instead (#1647, @dbczumar)
- [Projects] Fixed a bug that prevented docker-based MLflow Projects from logging artifacts to the `LocalArtifactRepository` (#1450, @nlaille)
- [Projects] Running MLflow projects with the `--no-conda` flag in R no longer requires Anaconda to be installed (#1650, @spadarian)
- [Models/Scoring] Fixed a bug that prevented Spark UDFs from being loaded on Databricks (#1658, @smurching)
- [UI] AJAX requests made by the MLflow Server Frontend now specify correct MIME-Types (#1679, @ynotzort)
- [UI] Previews now render correctly for artifacts with uppercase file extensions (e.g., `.JSON`, `.YAML`) (#1664, @ankitmathur-db)
- [UI] Fixed a bug that caused search API errors to surface a Niagara Falls page (#1681, @dbczumar)
- [Installation] MLflow dependencies are now selected properly based on the target installation platform (#1643, @akshaya-a)
- [UI] Fixed a bug where the "load more" button in the experiment view did not appear on browsers in Windows (#1718, @Zangr)

Small bug fixes and doc updates (#1663, #1719, @dbczumar; #1693, @max-allen-db; #1695, #1659, @smurching; #1675, @jdlesage; #1699, @ankitmathur-db; #1696, @aarondav; #1710, #1700, #1656, @apurva-koti)

## 1.1 (2019-07-22)

MLflow 1.1 includes several major features and improvements:

In MLflow Tracking:

- Experimental support for autologging from Tensorflow and Keras. Using `mlflow.tensorflow.autolog()` will enable automatic logging of metrics and optimizer parameters from TensorFlow to MLflow. The feature will work with TensorFlow versions `1.12 <= v < 2.0`. (#1520, #1601, @apurva-koti)
- Parallel coordinates plot in the MLflow compare run UI. Adds out of the box support for a parallel coordinates plot. The plot allows users to observe relationships between a n-dimensional set of parameters to metrics. It visualizes all runs as lines that are color-coded based on the value of a metric (e.g. accuracy), and shows what parameter values each run took on. (#1497, @Zangr)
- Pandas based search API. Adds the ability to return the results of a search as a pandas dataframe using the new `mlflow.search_runs` API. (#1483, #1548, @max-allen-db)
- Java fluent API. Adds a new set of APIs to create and log to MLflow runs. This API contrasts with the existing low level `MlflowClient` API which simply wraps the REST APIs. The new fluent API allows you to create and log runs similar to how you would using the Python fluent API. (#1508, @andrewmchen)
- Run tags improvements. Adds the ability to add and edit tags from the run view UI, delete tags from the API, and view tags in the experiment search view. (#1400, #1426, @Zangr; #1548, #1558, @ankitmathur-db)
- Search API improvements. Adds order by and pagination to the search API. Pagination allows you to read a large set of runs in small page sized chunks. This allows clients and backend implementations to handle an unbounded set of runs in a scalable manner. (#1444, @sueann; #1437, #1455, #1482, #1485, #1542, @aarondav; #1567, @max-allen-db; #1217, @mparkhe)
- Windows support for running the MLflow tracking server and UI. (#1080, @akshaya-a)

In MLflow Projects:

- Experimental support to run Docker based MLprojects in Kubernetes. Adds the first fully open source remote execution backend for MLflow projects. With this, you can leverage elastic compute resources managed by kubernetes for their ML training purposes. For example, you can run grid search over a set of hyperparameters by running several instances of an MLproject in parallel. (#1181, @marcusrehm, @tomasatdatabricks, @andrewmchen; #1566, @stbof, @dbczumar; #1574 @dbczumar)

**More features and improvements**

In MLflow Tracking:

- Paginated “load more” and backend sorting for experiment search view UI. This change allows the UI to scalably display the sorted runs from large experiments. (#1564, @Zangr)
- Search results are encoded in the URL. This allows you to share searches through their URL and to deep link to them. (#1416, @apurva-koti)
- Ability to serve MLflow UI behind `jupyter-server-proxy` or outside of the root path `/`. Previous to MLflow 1.1, the UI could only be hosted on `/` since the Javascript makes requests directly to `/ajax-api/...`. With this patch, MLflow will make requests to `ajax-api/...` or a path relative to where the HTML is being served. (#1413, @xhochy)

In MLflow Models:

- Update `mlflow.spark.log_model()` to accept descendants of pyspark.Model (#1519, @ankitmathur-db)
- Support for saving custom Keras models with `custom_objects`. This field is semantically equivalent to custom_objects parameter of `keras.models.load_model()` function (#1525, @ankitmathur-db)
- New more performant split orient based input format for pyfunc scoring server (#1479, @lennon310)
- Ability to specify gunicorn server options for pyfunc scoring server built with `mlflow models build-docker`. #1428, @lennon310)

**Bug fixes and documentation updates**

- [Tracking] Fix database migration for MySQL. `mlflow db upgrade` should now work for MySQL backends. (#1404, @sueann)
- [Tracking] Make CLI `mlflow server` and `mlflow ui` commands to work with SQLAlchemy URIs that specify a database driver. (#1411, @sueann)
- [Tracking] Fix usability bugs related to FTP artifact repository. (#1398, @kafendt; #1421, @nlaille)
- [Tracking] Return appropriate HTTP status codes for MLflowException (#1434, @max-allen-db)
- [Tracking] Fix sorting by user ID in the experiment search view. (#1401, @andrewmchen)
- [Tracking] Allow calling log_metric with NaNs and infs. (#1573, @tomasatdatabricks)
- [Tracking] Fixes an infinite loop in downloading artifacts logged via dbfs and retrieved via S3. (#1605, @sueann)
- [Projects] Docker projects should preserve directory structure (#1436, @ahutterTA)
- [Projects] Fix conda activation for newer versions of conda. (#1576, @avinashraghuthu, @smurching)
- [Models] Allow you to log Tensorflow keras models from the `tf.keras` module. (#1546, @tomasatdatabricks)

Small bug fixes and doc updates (#1463, @mateiz; #1641, #1622, #1418, @sueann; #1607, #1568, #1536, #1478, #1406, #1408, @smurching; #1504, @LizaShak; #1490, @acroz; #1633, #1631, #1603, #1589, #1569, #1526, #1446, #1438, @apurva-koti; #1456, @Taur1ne; #1547, #1495, @aarondav; #1610, #1600, #1492, #1493, #1447, @tomasatdatabricks; #1430, @javierluraschi; #1424, @nathansuh; #1488, @henningsway; #1590, #1427, @Zangr; #1629, #1614, #1574, #1521, #1522, @dbczumar; #1577, #1514, @ankitmathur-db; #1588, #1566, @stbof; #1575, #1599, @max-allen-db; #1592, @abaveja313; #1606, @andrewmchen)

## 1.0 (2019-06-03)

MLflow 1.0 includes many significant features and improvements. From this version, MLflow is no longer beta, and all APIs except those marked as experimental are intended to be stable until the next major version. As such, this release includes a number of breaking changes.

Major features, improvements, and breaking changes

- Support for recording, querying, and visualizing metrics along a new “step” axis (x coordinate), providing increased flexibility for examining model performance relative to training progress. For example, you can now record performance metrics as a function of the number of training iterations or epochs. MLflow 1.0’s enhanced metrics UI enables you to visualize the change in a metric’s value as a function of its step, augmenting MLflow’s existing UI for plotting a metric’s value as a function of wall-clock time. (#1202, #1237, @dbczumar; #1132, #1142, #1143, @smurching; #1211, #1225, @Zangr; #1372, @stbof)
- Search improvements. MLflow 1.0 includes additional support in both the API and UI for searching runs within a single experiment or a group of experiments. The search filter API supports a simplified version of the `SQL WHERE` clause. In addition to searching using run's metrics and params, the API has been enhanced to support a subset of run attributes as well as user and [system tags](https://mlflow.org/docs/latest/tracking.html#system-tags). For details see [Search syntax](https://mlflow.org/docs/latest/search-syntax.html#syntax) and [examples for programmatically searching runs](https://mlflow.org/docs/latest/search-syntax.html#programmatically-searching-runs). (#1245, #1272, #1323, #1326, @mparkhe; #1052, @Zangr; #1363, @aarondav)
- Logging metrics in batches. MLflow 1.0 now has a `runs/log-batch` REST API endpoint for logging multiple metrics, params, and tags in a single API request. The endpoint useful for performant logging of multiple metrics at the end of a model training epoch (see [example](https://github.com/mlflow/mlflow/blob/bb8c7602dcb6a3a8786301fe6b98f01e8d3f288d/examples/hyperparam/search_hyperopt.py#L161)), or logging of many input model parameters at the start of training. You can call this batched-logging endpoint from Python (`mlflow.log_metrics`, `mlflow.log_params`, `mlflow.set_tags`), R (`mlflow_log_batch`), and Java (`MlflowClient.logBatch`). (#1214, @dbczumar; see 0.9.1 and 0.9.0 for other changes)
- Windows support for MLflow Tracking. The Tracking portion of the MLflow client is now supported on Windows. (#1171, @eedeleon, @tomasatdatabricks)
- HDFS support for artifacts. Hadoop artifact repository with Kerberos authorization support was added, so you can use HDFS to log and retrieve models and other artifacts. (#1011, @jaroslawk)
- CLI command to build Docker images for serving. Added an `mlflow models build-docker` CLI command for building a Docker image capable of serving an MLflow model. The model is served at port 8080 within the container by default. Note that this API is experimental and does not guarantee that the arguments nor format of the Docker container will remain the same. (#1329, @smurching, @tomasatdatabricks)
- New `onnx` model flavor for saving, loading, and evaluating ONNX models with MLflow. ONNX flavor APIs are available in the `mlflow.onnx` module. (#1127, @avflor, @dbczumar; #1388, #1389, @dbczumar)
- Major breaking changes:

  - Some of the breaking changes involve database schema changes in the SQLAlchemy tracking store. If your database instance's schema is not up-to-date, MLflow will issue an error at the start-up of `mlflow server` or `mlflow ui`. To migrate an existing database to the newest schema, you can use the `mlflow db upgrade` CLI command. (#1155, #1371, @smurching; #1360, @aarondav)
  - [Installation] The MLflow Python package no longer depends on `scikit-learn`, `mleap`, or `boto3`. If you want to use the `scikit-learn` support, the `MLeap` support, or `s3` artifact repository / `sagemaker` support, you will have to install these respective dependencies explicitly. (#1223, @aarondav)
  - [Artifacts] In the Models API, an artifact's location is now represented as a URI. See the [documentation](https://mlflow.org/docs/latest/tracking.html#artifact-locations) for the list of accepted URIs. (#1190, #1254, @dbczumar; #1174, @dbczumar, @sueann; #1206, @tomasatdatabricks; #1253, @stbof)

    - The affected methods are:

      - Python: `<model-type>.load_model`, `azureml.build_image`, `sagemaker.deploy`, `sagemaker.run_local`, `pyfunc._load_model_env`, `pyfunc.load_pyfunc`, and `pyfunc.spark_udf`
      - R: `mlflow_load_model`, `mlflow_rfunc_predict`, `mlflow_rfunc_serve`
      - CLI: `mlflow models serve`, `mlflow models predict`, `mlflow sagemaker`, `mlflow azureml` (with the new `--model-uri` option)

    - To allow referring to artifacts in the context of a run, MLflow introduces a new URI scheme of the form `runs:/<run_id>/relative/path/to/artifact`. (#1169, #1175, @sueann)

  - [CLI] `mlflow pyfunc` and `mlflow rfunc` commands have been unified as `mlflow models` (#1257, @tomasatdatabricks; #1321, @dbczumar)
  - [CLI] `mlflow artifacts download`, `mlflow artifacts download-from-uri` and `mlflow download` commands have been consolidated into `mlflow artifacts download` (#1233, @sueann)
  - [Runs] Expose `RunData` fields (`metrics`, `params`, `tags`) as dictionaries. Note that the `mlflow.entities.RunData` constructor still accepts lists of `metric`/`param`/`tag` entities. (#1078, @smurching)
  - [Runs] Rename `run_uuid` to `run_id` in Python, Java, and REST API. Where necessary, MLflow will continue to accept `run_uuid` until MLflow 1.1. (#1187, @aarondav)

Other breaking changes

CLI:

- The `--file-store` option is deprecated in `mlflow server` and `mlflow ui` commands. (#1196, @smurching)
- The `--host` and `--gunicorn-opts` options are removed in the `mlflow ui` command. (#1267, @aarondav)
- Arguments to `mlflow experiments` subcommands, notably `--experiment-name` and `--experiment-id` are now options (#1235, @sueann)
- `mlflow sagemaker list-flavors` has been removed (#1233, @sueann)

Tracking:

- The `user` property of `Run`s has been moved to tags (similarly, the `run_name`, `source_type`, `source_name` properties were moved to tags in 0.9.0). (#1230, @acroz; #1275, #1276, @aarondav)
- In R, the return values of experiment CRUD APIs have been updated to more closely match the REST API. In particular, `mlflow_create_experiment` now returns a string experiment ID instead of an experiment, and the other APIs return NULL. (#1246, @smurching)
- `RunInfo.status`'s type is now string. (#1264, @mparkhe)
- Remove deprecated `RunInfo` properties from `start_run`. (#1220, @aarondav)
- As deprecated in 0.9.1 and before, the `RunInfo` fields `run_name`, `source_name`, `source_version`, `source_type`, and `entry_point_name` and the `SearchRuns` field `anded_expressions` have been removed from the REST API and Python, Java, and R tracking client APIs. They are still available as tags, documented in the REST API documentation. (#1188, @aarondav)

Models and deployment:

- In Python, require arguments as keywords in `log_model`, `save_model` and `add_to_model` methods in the `tensorflow` and `mleap` modules to avoid breaking changes in the future (#1226, @sueann)
- Remove the unsupported `jars` argument from ``spark.log_model` in Python (#1222, @sueann)
- Introduce `pyfunc.load_model` to be consistent with other Models modules. `pyfunc.load_pyfunc` will be deprecated in the near future. (#1222, @sueann)
- Rename `dst_path` parameter in `pyfunc.save_model` to `path` (#1221, @aarondav)
- R flavors refactor (#1299, @kevinykuo)

  - `mlflow_predict()` has been added in favor of `mlflow_predict_model()` and `mlflow_predict_flavor()` which have been removed.
  - `mlflow_save_model()` is now a generic and `mlflow_save_flavor()` is no longer needed and has been removed.
  - `mlflow_predict()` takes `...` to pass to underlying predict methods.
  - `mlflow_load_flavor()` now has the signature `function(flavor, model_path)` and flavor authors should implement `mlflow_load_flavor.mlflow_flavor_{FLAVORNAME}`. The flavor argument is inferred from the inputs of user-facing `mlflow_load_model()` and does not need to be explicitly provided by the user.

Projects:

- Remove and rename some `projects.run` parameters for generality and consistency. (#1222, @sueann)
- In R, the `mlflow_run` API for running MLflow projects has been modified to more closely reflect the Python `mlflow.run` API. In particular, the order of the `uri` and `entry_point` arguments has been reversed and the `param_list` argument has been renamed to `parameters`. (#1265, @smurching)

R:

- Remove `mlflow_snapshot` and `mlflow_restore_snapshot` APIs. Also, the `r_dependencies` argument used to specify the path to a packrat r-dependencies.txt file has been removed from all APIs. (#1263, @smurching)
- The `mlflow_cli` and `crate` APIs are now private. (#1246, @smurching)

Environment variables:

- Prefix environment variables with "MLFLOW\_" (#1268, @aarondav). Affected variables are:

  - [Tracking] `_MLFLOW_SERVER_FILE_STORE`, `_MLFLOW_SERVER_ARTIFACT_ROOT`, `_MLFLOW_STATIC_PREFIX`
  - [SageMaker] `MLFLOW_SAGEMAKER_DEPLOY_IMG_URL`, `MLFLOW_DEPLOYMENT_FLAVOR_NAME`
  - [Scoring] `MLFLOW_SCORING_SERVER_MIN_THREADS`, `MLFLOW_SCORING_SERVER_MAX_THREADS`

More features and improvements

- [Tracking] Non-default driver support for SQLAlchemy backends: `db+driver` is now a valid tracking backend URI scheme (#1297, @drewmcdonald; #1374, @mparkhe)
- [Tracking] Validate backend store URI before starting tracking server (#1218, @luke-zhu, @sueann)
- [Tracking] Add `GetMetricHistory` client API in Python and Java corresponding to the REST API. (#1178, @smurching)
- [Tracking] Add `view_type` argument to `MlflowClient.list_experiments()` in Python. (#1212, @smurching)
- [Tracking] Dictionary values provided to `mlflow.log_params` and `mlflow.set_tags` in Python can now be non-string types (e.g., numbers), and they are automatically converted to strings. (#1364, @aarondav)
- [Tracking] R API additions to be at parity with REST API and Python (#1122, @kevinykuo)
- [Tracking] Limit number of results returned from `SearchRuns` API and UI for faster load (#1125, @mparkhe; #1154, @andrewmchen)
- [Artifacts] To avoid having many copies of large model files in serving, `ArtifactRepository.download_artifacts` no longer copies local artifacts (#1307, @andrewmchen; #1383, @dbczumar)
- [Artifacts/Projects] Support GCS in download utilities. `gs://bucket/path` files are now supported by the `mlflow artifacts download` CLI command and as parameters of type `path` in MLProject files. (#1168, @drewmcdonald)
- [Models] All Python models exported by MLflow now declare `mlflow` as a dependency by default. In addition, we introduce a flag `--install-mlflow` users can pass to `mlflow models serve` and `mlflow models predict` methods to force installation of the latest version of MLflow into the model's environment. (#1308, @tomasatdatabricks)
- [Models] Update model flavors to lazily import dependencies in Python. Modules that define Model flavors now import extra dependencies such as `tensorflow`, `scikit-learn`, and `pytorch` inside individual _methods_, ensuring that these modules can be imported and explored even if the dependencies have not been installed on your system. Also, the `DEFAULT_CONDA_ENVIRONMENT` module variable has been replaced with a `get_default_conda_env()` function for each flavor. (#1238, @dbczumar)
- [Models] It is now possible to pass extra arguments to `mlflow.keras.load_model` that will be passed through to `keras.load_model`. (#1330, @yorickvP)
- [Serving] For better performance, switch to `gunicorn` for serving Python models. This does not change the user interface. (#1322, @tomasatdatabricks)
- [Deployment] For SageMaker, use the uniquely-generated model name as the S3 bucket prefix instead of requiring one. (#1183, @dbczumar)
- [REST API] Add support for API paths without the `preview` component. The `preview` paths will be deprecated in a future version of MLflow. (#1236, @mparkhe)

Bug fixes and documentation updates

- [Tracking] Log metric timestamps in milliseconds by default (#1177, @smurching; #1333, @dbczumar)
- [Tracking] Fix bug when deserializing integer experiment ID for runs in `SQLAlchemyStore` (#1167, @smurching)
- [Tracking] Ensure unique constraint names in MLflow tracking database (#1292, @smurching)
- [Tracking] Fix base64 encoding for basic auth in R tracking client (#1126, @freefrag)
- [Tracking] Correctly handle `file:` URIs for the `-—backend-store-uri` option in `mlflow server` and `mlflow ui` CLI commands (#1171, @eedeleon, @tomasatdatabricks)
- [Artifacts] Update artifact repository download methods to return absolute paths (#1179, @dbczumar)
- [Artifacts] Make FileStore respect the default artifact location (#1332, @dbczumar)
- [Artifacts] Fix `log_artifact` failures due to existing directory on FTP server (#1327, @kafendt)
- [Artifacts] Fix GCS artifact logging of subdirectories (#1285, @jason-huling)
- [Projects] Fix bug not sharing `SQLite` database file with Docker container (#1347, @tomasatdatabricks; #1375, @aarondav)
- [Java] Mark `sendPost` and `sendGet` as experimental (#1186, @aarondav)
- [Python/CLI] Mark `azureml.build_image` as experimental (#1222, #1233 @sueann)
- [Docs] Document public MLflow environment variables (#1343, @aarondav)
- [Docs] Document MLflow system tags for runs (#1342, @aarondav)
- [Docs] Autogenerate CLI documentation to include subcommands and descriptions (#1231, @sueann)
- [Docs] Update run selection description in `mlflow_get_run` in R documentation (#1258, @dbczumar)
- [Examples] Update examples to reflect API changes (#1361, @tomasatdatabricks; #1367, @mparkhe)

Small bug fixes and doc updates (#1359, #1350, #1331, #1301, #1270, #1271, #1180, #1144, #1135, #1131, #1358, #1369, #1368, #1387, @aarondav; #1373, @akarloff; #1287, #1344, #1309, @stbof; #1312, @hchiuzhuo; #1348, #1349, #1294, #1227, #1384, @tomasatdatabricks; #1345, @withsmilo; #1316, @ancasarb; #1313, #1310, #1305, #1289, #1256, #1124, #1097, #1162, #1163, #1137, #1351, @smurching; #1319, #1244, #1224, #1195, #1194, #1328, @dbczumar; #1213, #1200, @Kublai-Jing; #1304, #1320, @andrewmchen; #1311, @Zangr; #1306, #1293, #1147, @mateiz; #1303, @gliptak; #1261, #1192, @eedeleon; #1273, #1259, @kevinykuo; #1277, #1247, #1243, #1182, #1376, @mparkhe; #1210, @vgod-dbx; #1199, @ashtuchkin; #1176, #1138, #1365, @sueann; #1157, @cclauss; #1156, @clemens-db; #1152, @pogil; #1146, @srowen; #875, #1251, @jimthompson5802)

## 0.9.1 (2019-04-21)

MLflow 0.9.1 is a patch release on top of 0.9.0 containing mostly bug fixes and internal improvements. We have also included a one breaking API change in preparation for additions in MLflow 1.0 and later. This release also includes significant improvements to the Search API.

Breaking changes:

- [Tracking] Generalized experiment_id to string (from a long) to be more permissive of different ID types in different backend stores. While breaking for the REST API, this change is backwards compatible for python and R clients. (#1067, #1034 @eedeleon)

More features and improvements:

- [Search/API] Moving search filters into a query string based syntax, with Java client, Python client, and UI support. This also improves quote, period, and special character handling in query strings and adds the ability to search on tags in filter string. (#1042, #1055, #1063, #1068, #1099, #1106 @mparkhe; #1025 @andrewmchen; #1060 @smurching)
- [Tracking] Limits and validations to batch-logging APIs in OSS server (#958 @smurching)
- [Tracking/Java] Java client API for batch-logging (#1081 @mparkhe)
- [Tracking] Improved consistency of handling multiple metric values per timestamp across tracking stores (#972, #999 @dbczumar)

Bug fixes and documentation updates:

- [Tracking/Python] Reintroduces the parent_run_id argument to MlflowClient.create_run. This API is planned for removal in MLflow 1.0 (#1137 @smurching)
- [Tracking/Python] Provide default implementations of AbstractStore log methods (#1051 @acroz)
- [R] (Released on CRAN as MLflow 0.9.0.1) Small bug fixes with R (#1123 @smurching; #1045, #1017, #1019, #1039, #1048, #1098, #1101, #1107, #1108, #1119 @tomasatdatabricks)

Small bug fixes and doc updates (#1024, #1029 @bayethiernodiop; #1075 @avflor; #968, #1010, #1070, #1091, #1092 @smurching; #1004, #1085 @dbczumar; #1033, #1046 @sueann; #1053 @tomasatdatabricks; #987 @hanyucui; #935, #941 @jimthompson5802; #963 @amilbourne; #1016 @andrewmchen; #991 @jaroslawk; #1007 @mparkhe)

## 0.9.0.1 (2019-04-09)

Bugfix release (PyPI only) with the following changes:

- Rebuilt MLflow JS assets to fix an issue where form input was broken in MLflow 0.9.0 (identified
  in #1056, #1113 by @shu-yusa, @timothyjlaurent)

  0.9.0 (2019-03-13)

Major features:

- Support for running MLflow Projects in Docker containers. This allows you to include non-Python dependencies in their project environments and provides stronger isolation when running projects. See the [Projects documentation](https://mlflow.org/docs/latest/projects.html) for more information. (#555, @marcusrehm; #819, @mparkhe; #970, @dbczumar)
- Database stores for the MLflow Tracking Server. Support for a scalable and performant backend store was one of the top community requests. This feature enables you to connect to local or remote SQLAlchemy-compatible databases (currently supported flavors include MySQL, PostgreSQL, SQLite, and MS SQL) and is compatible with file backed store. See the [Tracking Store documentation](https://mlflow.org/docs/latest/tracking.html#storage) for more information. (#756, @AndersonReyes; #800, #844, #847, #848, #860, #868, #975, @mparkhe; #980, @dbczumar)
- Simplified custom Python model packaging. You can easily include custom preprocessing and postprocessing logic, as well as data dependencies in models with the `python_function` flavor using updated `mlflow.pyfunc` Python APIs. For more information, see the [Custom Python Models documentation](https://mlflow.org/docs/latest/models.html#custom-python-models). (#791, #792, #793, #830, #910, @dbczumar)
- Plugin systems allowing third party libraries to extend MLflow functionality. The [proposal document](https://gist.github.com/zblz/9e337a55a7ba73314890be68370fa69a) gives the full detail of the three main changes:

  - You can register additional providers of tracking stores using the `mlflow.tracking_store` entrypoint. (#881, @zblz)
  - You can register additional providers of artifact repositories using the `mlflow.artifact_repository` entrypoint. (#882, @mociarain)
  - The logic generating run metadata from the run context (e.g. `source_name`, `source_version`) has been refactored into an extendable system of run context providers. Plugins can register additional providers using the `mlflow.run_context_provider` entrypoint, which add to or overwrite tags set by the base library. (#913, #926, #930, #978, @acroz)

- Support for HTTP authentication to the Tracking Server in the R client. Now you can connect to secure Tracking Servers using credentials set in environment variables, or provide custom plugins for setting the credentials. As an example, this release contains a Databricks plugin that can detect existing Databricks credentials to allow you to connect to the Databricks Tracking Server. (#938, #959, #992, @tomasatdatabricks)

Breaking changes:

- [Scoring] The `pyfunc` scoring server now expects requests with the `application/json` content type to contain json-serialized pandas dataframes in the split format, rather than the records format. See the [documentation on deployment](https://mlflow.org/docs/latest/models.html#deploy-a-python-function-model-as-a-local-rest-api-endpoint) for more detail. (#960, @dbczumar) Also, when reading the pandas dataframes from JSON, the scoring server no longer automatically infers data types as it can result in unintentional conversion of data types (#916, @mparkhe).
- [API] Remove `GetMetric` & `GetParam` from the REST API as they are subsumed by `GetRun`. (#879, @aarondav)

More features and improvements:

- [UI] Add a button for downloading artifacts (#967, @mateiz)
- [CLI] Add CLI commands for runs: now you can `list`, `delete`, `restore`, and `describe` runs through the CLI (#720, @DorIndivo)
- [CLI] The `run` command now can take `--experiment-name` as an argument, as an alternative to the `--experiment-id` argument. You can also choose to set the `_EXPERIMENT_NAME_ENV_VAR` environment variable instead of passing in the value explicitly. (#889, #894, @mparkhe)
- [Examples] Add Image classification example with Keras. (#743, @tomasatdatabricks )
- [Artifacts] Add `get_artifact_uri()` and `_download_artifact_from_uri` convenience functions (#779)
- [Artifacts] Allow writing Spark models directly to the target artifact store when possible (#808, @smurching)
- [Models] PyTorch model persistence improvements to allow persisting definitions and dependencies outside the immediate scope:
  - Add a `code_paths` parameter to `mlflow.pytorch.save_model` and `mlflow.pytorch.log_model` to allow external module dependencies to be specified as paths to python files. (#842, @dbczumar)
  - Improve `mlflow.pytorch.save_model` to capture class definitions from notebooks and the `__main__` scope (#851, #861, @dbczumar)
- [Runs/R] Allow client to infer context info when creating new run in fluent API (#958, @tomasatdatabricks)
- [Runs/UI] Support Git Commit hyperlink for Gitlab and Bitbucket. Previously the clickable hyperlink was generated only for Github pages. (#901)
- [Search]/API] Allow param value to have any content, not just alphanumeric characters, `.`, and `-` (#788, @mparkhe)
- [Search/API] Support "filter" string in the `SearchRuns` API. Corresponding UI improvements are planned for the future (#905, @mparkhe)
- [Logging] Basic support for LogBatch. NOTE: The feature is currently experimental and the behavior is expected to change in the near future. (#950, #951, #955, #1001, @smurching)

Bug fixes and documentation updates:

- [Artifacts] Fix empty-file upload to DBFS in `log_artifact` and `log_artifacts` (#895, #818, @smurching)
- [Artifacts] S3 artifact store: fix path resolution error when artifact root is bucket root (#928, @dbczumar)
- [UI] Fix a bug with Databricks notebook URL links (#891, @smurching)
- [Export] Fix for missing run name in csv export (#864, @jimthompson5802)
- [Example] Correct missing tensorboardX module error in PyTorch example when running in MLflow Docker container (#809, @jimthompson5802)
- [Scoring/R] Fix local serving of rfunc models (#874, @kevinykuo)
- [Docs] Improve flavor-specific documentation in Models documentation (#909, @dbczumar)

Small bug fixes and doc updates (#822, #899, #787, #785, #780, #942, @hanyucui; #862, #904, #954, #806, #857, #845, @stbof; #907, #872, @smurching; #896, #858, #836, #859, #923, #939, #933, #931, #952, @dbczumar; #880, @zblz; #876, @acroz; #827, #812, #816, #829, @jimthompson5802; #837, #790, #897, #974, #900, @mparkhe; #831, #798, @aarondav; #814, @sueann; #824, #912, @mateiz; #922, #947, @tomasatdatabricks; #795, @KevYuen; #676, @mlaradji; #906, @4n4nd; #777, @tmielika; #804, @alkersan)

## 0.8.2 (2019-01-28)

MLflow 0.8.2 is a patch release on top of 0.8.1 containing only bug fixes and no breaking changes or features.

Bug fixes:

- [Python API] CloudPickle has been added to the set of MLflow library dependencies, fixing missing import errors when attempting to save models (#777, @tmielika)
- [Python API] Fixed a malformed logging call that prevented `mlflow.sagemaker.push_image_to_ecr()` invocations from succeeding (#784, @jackblandin)
- [Models] PyTorch models can now be saved with code dependencies, allowing model classes to be loaded successfully in new environments (#842, #836, @dbczumar)
- [Artifacts] Fixed a timeout when logging zero-length files to DBFS artifact stores (#818, @smurching)

Small docs updates (#845, @stbof; #840, @grahamhealy20; #839, @wilderrodrigues)

## 0.8.1 (2018-12-21)

MLflow 0.8.1 introduces several significant improvements:

- Improved UI responsiveness and load time, especially when displaying experiments containing hundreds to thousands of runs.
- Improved visualizations, including interactive scatter plots for MLflow run comparisons
- Expanded support for scoring Python models as Spark UDFs. For more information, see the [updated documentation for this feature](https://mlflow.org/docs/latest/models.html#export-a-python-function-model-as-an-apache-spark-udf).
- By default, saved models will now include a Conda environment specifying all of the dependencies necessary for loading them in a new environment.

Features:

- [API/CLI] Support for running MLflow projects from ZIP files (#759, @jmorefieldexpe)
- [Python API] Support for passing model conda environments as dictionaries to `save_model` and `log_model` functions (#748, @dbczumar)
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

## 0.8.0 (2018-11-08)

MLflow 0.8.0 introduces several major features:

- Dramatically improved UI for comparing experiment run results:

  - Metrics and parameters are by default grouped into a single column, to avoid an explosion of mostly-empty columns. Individual metrics and parameters can be moved into their own column to help compare across rows.
  - Runs that are "nested" inside other runs (e.g., as part of a hyperparameter search or multistep workflow) now show up grouped by their parent run, and can be expanded or collapsed altogether. Runs can be nested by calling `mlflow.start_run` or `mlflow.run` while already within a run.
  - Run names (as opposed to automatically generated run UUIDs) now show up instead of the run ID, making comparing runs in graphs easier.
  - The state of the run results table, including filters, sorting, and expanded rows, is persisted in browser local storage, making it easier to go back and forth between an individual run view and the table.

- Support for deploying models as Docker containers directly to Azure Machine Learning Service Workspace (as opposed to the previously-recommended solution of Azure ML Workbench).

Breaking changes:

- [CLI] `mlflow sklearn serve` has been removed in favor of `mlflow pyfunc serve`, which takes the same arguments but works against any pyfunc model (#690, @dbczumar)

Features:

- [Scoring] pyfunc server and SageMaker now support the pandas "split" JSON format in addition to the "records" format. The split format allows the client to specify the order of columns, which is necessary for some model formats. We recommend switching client code over to use this new format (by sending the Content-Type header `application/json; format=pandas-split`), as it will become the default JSON format in MLflow 0.9.0. (#690, @dbczumar)
- [UI] Add compact experiment view (#546, #620, #662, #665, @smurching)
- [UI] Add support for viewing & tracking nested runs in experiment view (#588, @andrewmchen; #618, #619, @aarondav)
- [UI] Persist experiments view filters and sorting in browser local storage (#687, @smurching)
- [UI] Show run name instead of run ID when present (#476, @smurching)
- [Scoring] Support for deploying Models directly to Azure Machine Learning Service Workspace (#631, @dbczumar)
- [Server/Python/Java] Add `rename_experiment` to Tracking API (#570, @aarondav)
- [Server] Add `get_experiment_by_name` to RestStore (#592, @dmarkhas)
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

## 0.7.0 (2018-10-01)

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

Small bug fixes and doc updates (#467 @drorata; #470, #497, #508, #518 @dbczumar; #455, #466, #492, #504, #527 @aarondav; #481, #475, #484, #496, #515, #517, #498, #521, #522, #573 @smurching; #477 @parkerzf; #494 @jainr; #501, #531, #532, #552 @mparkhe; #503, #520 @dmatrix; #509, #532 @tomasatdatabricks; #484, #486 @stbof; #533, #534 @javierluraschi; #542 @GCBallesteros; #511 @AdamBarnhard)

## 0.6.0 (2018-09-10)

MLflow 0.6.0 introduces several major features:

- A Java client API, available on Maven
- Support for saving and serving SparkML models as MLeap for low-latency serving
- Support for tagging runs with metadata, during and after the run completion
- Support for deleting (and restoring deleted) experiments

In addition to these features, there are a host of improvements and bugfixes to the REST API, Python API, tracking UI, and documentation. The [examples](https://github.com/mlflow/mlflow/tree/master/examples) subdirectory has also been revamped to make it easier to jump in, and examples demonstrating multistep workflows and hyperparameter tuning have been added.

Breaking changes:

We fixed a few inconsistencies in the the `mlflow.tracking` API, as introduced in 0.5.0:

- `MLflowService` has been renamed `MlflowClient` (#461, @mparkhe)
- You get an `MlflowClient` by calling `mlflow.tracking.MlflowClient()` (previously, this was `mlflow.tracking.get_service()`) (#461, @mparkhe)
- `MlflowService.list_runs` was changed to `MlflowService.list_run_infos` to reflect the information actually returned by the call. It now returns a `RunInfo` instead of a `Run` (#334, @aarondav)
- `MlflowService.log_artifact` and `MlflowService.log_artifacts` now take a `run_id` instead of `artifact_uri`. This now matches `list_artifacts` and `download_artifacts` (#444, @aarondav)

Features:

- Java client API added with support for the MLflow Tracking API (analogous to `mlflow.tracking`), allowing users to create and manage experiments, runs, and artifacts. The release includes a [usage example](https://github.com/mlflow/mlflow/blob/master/mlflow/java/client/src/main/java/org/mlflow/tracking/samples/QuickStartDriver.java>)and [Javadocs](https://mlflow.org/docs/latest/java_api/index.html). The client is published to Maven under `mlflow:mlflow` (#380, #394, #398, #409, #410, #430, #452, @aarondav)
- SparkML models are now also saved in MLeap format (https://github.com/combust/mleap), when applicable. Model serving platforms can choose to serve using this format instead of the SparkML format to dramatically decrease prediction latency. SageMaker now does this by default (#324, #327, #331, #395, #428, #435, #438, @dbczumar)
- [API] Experiments can now be deleted and restored via REST API, Python Tracking API, and MLflow CLI (#340, #344, #367, @mparkhe)
- [API] Tags can now be set via a SetTag API, and they have been moved to `RunData` from `RunInfo` (#342, @aarondav)
- [API] Added `list_artifacts` and `download_artifacts` to `MlflowService` to interact with a run's artifactory (#350, @andrewmchen)
- [API] Added `get_experiment_by_name` to Python Tracking API, and equivalent to Java API (#373, @vfdev-5)
- [API/Python] Version is now exposed via `mlflow.__version__`.
- [API/CLI] Added `mlflow artifacts` CLI to list, download, and upload to run artifact repositories (#391, @aarondav)
- [UI] Added icons to source names in MLflow Experiments UI (#381, @andrewmchen)
- [UI] Added support to view `.log` and `.tsv` files from MLflow artifacts UI (#393, @Shenggan; #433, @whiletruelearn)
- [UI] Run names can now be edited from within the MLflow UI (#382, @smurching)
- [Serving] Added `--host` option to `mlflow serve` to allow listening on non-local addressess (#401, @hamroune)
- [Serving/SageMaker] SageMaker serving takes an AWS region argument (#366, @dbczumar)
- [Python] Added environment variables to support providing HTTP auth (username, password, token) when talking to a remote MLflow tracking server (#402, @aarondav)
- [Python] Added support to override S3 endpoint for S3 artifactory (#451, @hamroune)
- MLflow nightly Python wheel and JAR snapshots are now available and linked from https://github.com/mlflow/mlflow (#352, @aarondav)

Bug fixes and documentation updates:

- [Python] `mlflow run` now logs default parameters, in addition to explicitly provided ones (#392, @mparkhe)
- [Python] `log_artifact` in FileStore now requires a relative path as the artifact path (#439, @mparkhe)
- [Python] Fixed string representation of Python entities, so they now display both their type and serialized fields (#371, @smurching)
- [UI] Entry point name is now shown in MLflow UI (#345, @aarondav)
- [Models] Keras model export now includes TensorFlow graph explicitly to ensure the model can always be loaded at deployment time (#440, @tomasatdatabricks)
- [Python] Fixed issue where FileStore ignored provided Run Name (#358, @adrian555)
- [Python] Fixed an issue where any `mlflow run` failing printed an extraneous exception (#365, @smurching)
- [Python] uuid dependency removed (#351, @antonpaquin)
- [Python] Fixed issues with remote execution on Databricks (#357, #361, @smurching; #383, #387, @aarondav)
- [Docs] Added [comprehensive example](https://github.com/mlflow/mlflow/tree/master/examples/multistep_workflow) of doing a multistep workflow, chaining MLflow runs together and reusing results (#338, @aarondav)
- [Docs] Added [comprehensive example](https://github.com/mlflow/mlflow/tree/master/examples/hyperparam) of doing hyperparameter tuning (#368, @tomasatdatabricks)
- [Docs] Added code examples to `mlflow.keras` API (#341, @dmatrix)
- [Docs] Significant improvements to Python API documentation (#454, @stbof)
- [Docs] Examples folder refactored to improve readability. The examples now reside in `examples/` instead of `example/`, too (#399, @mparkhe)
- Small bug fixes and doc updates (#328, #363, @ToonKBC; #336, #411, @aarondav; #284, @smurching; #377, @mparkhe; #389, gioa; #408, @aadamson; #397, @vfdev-5; #420, @adrian555; #459, #463, @stbof)

## 0.5.2 (2018-08-24)

MLflow 0.5.2 is a patch release on top of 0.5.1 containing only bug fixes and no breaking changes or features.

Bug fixes:

- Fix a bug with ECR client creation that caused `mlflow.sagemaker.deploy()` to fail when searching for a deployment Docker image (#366, @dbczumar)

## 0.5.1 (2018-08-23)

MLflow 0.5.1 is a patch release on top of 0.5.0 containing only bug fixes and no breaking changes or features.

Bug fixes:

- Fix `with mlflow.start_run() as run` to actually set `run` to the created Run (previously, it was None) (#322, @tomasatdatabricks)
- Fixes to DBFS artifactory to throw an exception if logging an artifact fails (#309) and to mimic FileStore's behavior of logging subdirectories (#347, @andrewmchen)
- Fix for Python 3.7 support with tarfiles (#329, @tomasatdatabricks)
- Fix spark.load_model not to delete the DFS tempdir (#335, @aarondav)
- MLflow UI now appropriately shows entrypoint if it's not main (#345, @aarondav)
- Make Python API forward-compatible with newer server versions of protos (#348, @aarondav)
- Improved API docs (#305, #284, @smurching)

## 0.5.0 (2018-08-17)

MLflow 0.5.0 offers some major improvements, including Keras and PyTorch first-class support as models, SFTP support as an artifactory, a new scatterplot visualization to compare runs, and a more complete Python SDK for experiment and run management.

Breaking changes:

- The Tracking API has been split into two pieces, a "basic logging" API and a "tracking service" API. The "basic logging" API deals with logging metrics, parameters, and artifacts to the currently-active active run, and is accessible in `mlflow` (e.g., `mlflow.log_param`). The tracking service API allow managing experiments and runs (especially historical runs) and is available in `mlflow.tracking`. The tracking service API will look analogous to the upcoming R and Java Tracking Service SDKs. Please be aware of the following breaking changes:

  - `mlflow.tracking` no longer exposes the basic logging API, only `mlflow`. So, code that was written like `from mlflow.tracking import log_param` will have to be `from mlflow import log_param` (note that almost all examples were already doing this).
  - Access to the service API goes through the `mlflow.tracking.get_service()` function, which relies on the same tracking server set by either the environment variable `MLFLOW_TRACKING_URI` or by code with `mlflow.tracking.set_tracking_uri()`. So code that used to look like `mlflow.tracking.get_run()` will now have to do `mlflow.tracking.get_service().get_run()`. This does not apply to the basic logging API.
  - `mlflow.ActiveRun` has been converted into a lightweight wrapper around `mlflow.entities.Run` to enable the Python `with` syntax. This means that there are no longer any special methods on the object returned when calling `mlflow.start_run()`. These can be converted to the service API.

  - The Python entities returned by the tracking service API are now accessible in `mlflow.entities` directly. Where previously you may have used `mlflow.entities.experiment.Experiment`, you would now just use `mlflow.entities.Experiment`. The previous version still exists, but is deprecated and may be hidden in a future version.

- REST API endpoint `/ajax-api/2.0/preview/mlflow/artifacts/get` has been moved to `$static_prefix/get-artifact`. This change is coversioned in the JavaScript, so should not be noticeable unless you were calling the REST API directly (#293, @andremchen)

Features:

- [Models] Keras integration: we now support logging Keras models directly in the log_model API, model format, and serving APIs (#280, @ToonKBC)
- [Models] PyTorch integration: we now support logging PyTorch models directly in the log_model API, model format, and serving APIs (#264, @vfdev-5)
- [UI] Scatterplot added to "Compare Runs" view to help compare runs using any two metrics as the axes (#268, @ToonKBC)
- [Artifacts] SFTP artifactory store added (#260, @ToonKBC)
- [Sagemaker] Users can specify a custom VPC when deploying SageMaker models (#304, @dbczumar)
- Pyfunc serialization now includes the Python version, and warns if the major version differs (can be suppressed by using `load_pyfunc(suppress_warnings=True)`) (#230, @dbczumar)
- Pyfunc serve/predict will activate conda environment stored in MLModel. This can be disabled by adding `--no-conda` to `mlflow pyfunc serve` or `mlflow pyfunc predict` (#225, @0wu)
- Python SDK formalized in `mlflow.tracking`. This includes adding SDK methods for `get_run`, `list_experiments`, `get_experiment`, and `set_terminated`. (#299, @aarondav)
- `mlflow run` can now be run against projects with no `conda.yaml` specified. By default, an empty conda environment will be created -- previously, it would just fail. You can still pass `--no-conda` to avoid entering a conda environment altogether (#218, @smurching)

Bug fixes:

- Fix numpy array serialization for int64 and other related types, allowing pyfunc to return such results (#240, @arinto)
- Fix DBFS artifactory calling `log_artifacts` with binary data (#295, @aarondav)
- Fix Run Command shown in UI to reproduce a run when the original run is targeted at a subdirectory of a Git repo (#294, @adrian555)
- Filter out ubiquitious dtype/ufunc warning messages (#317, @aarondav)
- Minor bug fixes and documentation updates (#261, @stbof; #279, @dmatrix; #313, @rbang1, #320, @yassineAlouini; #321, @tomasatdatabricks; #266, #282, #289, @smurching; #267, #265, @aarondav; #256, #290, @ToonKBC; #273, #263, @mateiz; #272, #319, @adrian555; #277, @aadamson; #283, #296, @andrewmchen)

## 0.4.2 (2018-08-07)

Breaking changes: None

Features:

- MLflow experiments REST API and `mlflow experiments create` now support providing `--artifact-location` (#232, @aarondav)
- [UI] Runs can now be sorted by columns, and added a Select All button (#227, @ToonKBC)
- Databricks File System (DBFS) artifactory support added (#226, @andrewmchen)
- databricks-cli version upgraded to >= 0.8.0 to support new DatabricksConfigProvider interface (#257, @aarondav)

Bug fixes:

- MLflow client sends REST API calls using snake_case instead of camelCase field names (#232, @aarondav)
- Minor bug fixes (#243, #242, @aarondav; #251, @javierluraschi; #245, @smurching; #252, @mateiz)

## 0.4.1 (2018-08-03)

Breaking changes: None

Features:

- [Projects] MLflow will use the conda installation directory given by the `$MLFLOW_CONDA_HOME`
  if specified (e.g. running conda commands by invoking `$MLFLOW_CONDA_HOME/bin/conda`), defaulting
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

## 0.4.0 (2018-08-01)

Breaking changes:

- [Projects] Removed the `use_temp_cwd` argument to `mlflow.projects.run()`
  (`--new-dir` flag in the `mlflow run` CLI). Runs of local projects now use the local project
  directory as their working directory. Git projects are still fetched into temporary directories
  (#215, @smurching)
- [Tracking] GCS artifact storage is now a pluggable dependency (no longer installed by default).
  To enable GCS support, install `google-cloud-storage` on both the client and tracking server via pip.
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

## 0.3.0 (2018-07-18)

Breaking changes:

- [MLflow Server] Renamed `--artifact-root` parameter to `--default-artifact-root` in `mlflow server` to better reflect its purpose (#165, @aarondav)

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

## 0.2.1 (2018-06-28)

This is a patch release fixing some smaller issues after the 0.2.0 release.

- Switch protobuf implementation to C, fixing a bug related to tensorflow/mlflow import ordering (issues #33 and #77, PR #74, @andrewmchen)
- Enable running mlflow server without git binary installed (#90, @aarondav)
- Fix Spark UDF support when running on multi-node clusters (#92, @aarondav)

## 0.2.0 (2018-06-27)

- Added `mlflow server` to provide a remote tracking server. This is akin to `mlflow ui` with new options:

  - `--host` to allow binding to any ports (#27, @mdagost)
  - `--artifact-root` to allow storing artifacts at a remote location, S3 only right now (#78, @mateiz)
  - Server now runs behind gunicorn to allow concurrent requests to be made (#61, @mateiz)

- TensorFlow integration: we now support logging TensorFlow Models directly in the log_model API, model format, and serving APIs (#28, @juntai-zheng)
- Added `experiments.list_experiments` as part of experiments API (#37, @mparkhe)
- Improved support for unicode strings (#79, @smurching)
- Diabetes progression example dataset and training code (#56, @dennyglee)
- Miscellaneous bug and documentation fixes from @Jeffwan, @yupbank, @ndjido, @xueyumusic, @manugarri, @tomasatdatabricks, @stbof, @andyk, @andrewmchen, @jakeret, @0wu, @aarondav

## 0.1.0 (2018-06-05)

- Initial version of mlflow.
