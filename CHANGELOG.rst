Changelog
=========

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
