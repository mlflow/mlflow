Changelog
=========
1.9.0 (2020-06-19)
------------------
MLflow 1.9.0 includes numerous major features and improvements, and a breaking change to
experimental APIs:

Breaking Changes:

- The ``new_name`` argument to ``MlflowClient.update_registered_model``
  has been removed. Call ``MlflowClient.rename_registered_model`` instead. (#2946, @mparkhe)
- The ``stage`` argument to ``MlflowClient.update_model_version``
  has been removed. Call ``MlflowClient.transition_model_version_stage`` instead. (#2946, @mparkhe)

Features:

MLflow Models and Flavors:
- ``log_model`` and ``save_model`` APIs now support saving model signatures (the model's input and output schema)
  and example input along with the model itself  (#2698, #2775, @tomasatdatabricks). Model signatures are used
  to reorder and validate input fields when scoring/serving models using the pyfunc flavor, ``mlflow models``
  CLI commands, or ``mlflow.pyfunc.spark_udf`` (#2920, @tomasatdatabricks and @aarondav)
- Introduce fastai model persistence and autologging APIs under ``mlflow.fastai`` (#2619, #2689 @antoniomdk)
- Add pluggable ``mlflow.deployments`` API and CLI for deploying models to custom serving tools, e.g. RedisAI
  (#2327, @hhsecond)
- Enables loading and scoring models whose conda environments include dependencies in conda-forge (#2797, @dbczumar)
- Add support for scoring ONNX-persisted models that return Python lists (#2742, @andychow-db)

MLflow Projects:
- Add plugin interface for executing MLflow projects against custom backends (#2566, @jdlesage)
- Add ability to specify additional cluster-wide Python and Java libraries when executing
  MLflow projects remotely on Databricks (#2845, @pogil)
- Allow running MLflow projects against remote artifacts stored in any location with a corresponding
  ArtifactRepository implementation (Azure Blob Storage, GCS, etc) (#2774, @trangevi)
- Allow MLflow projects running on Kubernetes to specify a different tracking server to log to via the
KUBE_MLFLOW_TRACKING_URI for passing a different tracking server to the kubernetes job (#2874, @catapulta)

UI:
- Significant performance and scalability improvements to metric comparison and scatter plots in
  the UI (#2447, @mjlbach)
- The main MLflow experiment list UI now includes a link to the model registry UI (#2805, @zhidongqu-db),
- Enable viewing PDFs logged as artifacts from the runs UI  (#2859, @ankmathur96)
- UI accessibility improvements: better color contrast (#2872, @Zangr), add child roles to DOM elements (#2871, @Zangr)

Tracking Client:
- Adds ability to pass client certs as part of REST API requests when using the tracking or model
  registry APIs. (#2843, @PhilipMay)
- New community plugin: support for storing artifacts in Aliyun (Alibaba Cloud) (#2917, @SeaOfOcean)
- Infer and set content type and encoding of objects when logging models and artifacts to S3 (#2881, @hajapy)
- Adds support for logging artifacts to HDFS Federation ViewFs (#2782, @fhoering)

Tracking Server:
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
