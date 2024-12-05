===============
Artifact Stores
===============

The artifact store is a core component in `MLflow Tracking <../index.html>`_ where MLflow stores (typicaly large) artifacts
for each run such as model weights (e.g. a pickled scikit-learn model), images (e.g. PNGs), model and data files (e.g. `Parquet <https://parquet.apache.org/>`_ file). 
Note that metadata like parameters, metrics, and tags are stored in a `backend store <backend-stores.html>`_ (e.g., PostGres, MySQL, or MSSQL Database), the other component of the MLflow Tracking.

Configuring an Artifact Store
=============================
MLflow by default stores artifacts in local ``./mlruns`` directory, but also supports various locations suitable for large data:
Amazon S3, Azure Blob Storage, Google Cloud Storage, SFTP server, and NFS. You can connect those remote storages via the MLflow Tracking server.
See :ref:`tracking server setup <tracking-server-artifact-store>` and the specific section for your storage in :ref:`supported storages <artifacts-store-supported-storages>` for guidance on 
how to connect to your remote storage of choice.

.. _artifacts-stores-manage-access:

Managing Artifact Store Access
------------------------------
To allow the server and clients to access the artifact location, you should configure your cloud
provider credentials as you would for accessing them in any other capacity. For example, for S3, you can set the ``AWS_ACCESS_KEY_ID``
and ``AWS_SECRET_ACCESS_KEY`` environment variables, use an IAM role, or configure a default
profile in ``~/.aws/credentials``.

.. important::
    Access credentials and configuration for the artifact storage location are configured **once during server initialization** in the place
    of having users handle access credentials for artifact-based operations. Note that **all users who have access to the
    Tracking Server in this mode will have access to artifacts served through this assumed role**.

Setting an access Timeout
-------------------------
You can set an environment variable ``MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT`` (in seconds) to configure the timeout for artifact uploads and downloads.
If it's not set, MLflow will use the default timeout for the underlying storage client library (e.g. boto3 for S3).
Note that this is experimental feature, may be changed or removed.

Setting a Default Artifact Location for Logging
-----------------------------------------------
MLflow automatically records the ``artifact_uri`` property as a part of :py:class:`mlflow.entities.RunInfo`, so you can
retrieve the location of the artifacts for historical runs using the :py:func:`mlflow.get_artifact_uri` API. 
Also, ``artifact_location`` is a property recorded on :py:class:`mlflow.entities.Experiment` for setting the 
default location to store artifacts for all runs in a given experiment.

.. important::

  If you do not specify a ``--default-artifact-root`` or an artifact URI when creating the experiment
  (for example, ``mlflow experiments create --artifact-location s3://<my-bucket>``), the artifact root
  will be set as a path inside the local file store (the hard drive of the computer executing your run). Typically this is not an appropriate location, as the client and
  server probably refer to different physical locations (that is, the same path on different disks).

.. _artifacts-store-supported-storages:

Supported storage types for the Artifact Store
==============================================

Amazon S3 and S3-compatible storage
-----------------------------------

To store artifacts in S3 (whether on Amazon S3 or on an S3-compatible alternative, such as
`MinIO <https://min.io/>`_ or `Digital Ocean Spaces <https://www.digitalocean.com/products/spaces>`_), specify a URI of the form ``s3://<bucket>/<path>``. MLflow obtains
credentials to access S3 from your machine's IAM role, a profile in ``~/.aws/credentials``, or
the environment variables ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY`` depending on which of
these are available. For more information on how to set credentials, see
`Set up AWS Credentials and Region for Development <https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/setup-credentials.html>`_.

Followings are commonly used environment variables for configuring S3 storage access. The complete list of configurable parameters for an S3 client is available in the 
`boto3 documentation <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#configuration>`_.


Passsing Extra Arguments to S3 Upload
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To add S3 file upload extra arguments, set ``MLFLOW_S3_UPLOAD_EXTRA_ARGS`` to a JSON object of key/value pairs.
For example, if you want to upload to a KMS Encrypted bucket using the KMS Key 1234:

.. code-block:: bash

  export MLFLOW_S3_UPLOAD_EXTRA_ARGS='{"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "1234"}'

For a list of available extra args see `Boto3 ExtraArgs Documentation <https://github.com/boto/boto3/blob/develop/docs/source/guide/s3-uploading-files.rst#the-extraargs-parameter>`_.

Setting Custom S3 Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~
To store artifacts in a custom endpoint, set the ``MLFLOW_S3_ENDPOINT_URL`` to your endpoint's URL. For example, if you are using Digital Ocean Spaces:

.. code-block:: bash

  export MLFLOW_S3_ENDPOINT_URL=https://<region>.digitaloceanspaces.com

If you have a MinIO server at 1.2.3.4 on port 9000:

.. code-block:: bash

  export MLFLOW_S3_ENDPOINT_URL=http://1.2.3.4:9000

Using Non-TLS Authentication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the MinIO server is configured with using SSL self-signed or signed using some internal-only CA certificate, you could set ``MLFLOW_S3_IGNORE_TLS`` or ``AWS_CA_BUNDLE`` variables (not both at the same time!) to disable certificate signature check, or add a custom CA bundle to perform this check, respectively:

.. code-block:: bash

  export MLFLOW_S3_IGNORE_TLS=true
  #or
  export AWS_CA_BUNDLE=/some/ca/bundle.pem

Setting Bucket Region
~~~~~~~~~~~~~~~~~~~~~
Additionally, if MinIO server is configured with non-default region, you should set ``AWS_DEFAULT_REGION`` variable:

.. code-block:: bash

  export AWS_DEFAULT_REGION=my_region

.. warning::

        The MLflow tracking server utilizes specific reserved keywords to generate a qualified path. These environment configurations, if present in the client environment, can create path resolution issues.
        For example, providing ``--default-artifact-root $MLFLOW_S3_ENDPOINT_URL`` on the server side **and** ``MLFLOW_S3_ENDPOINT_URL`` on the client side will create a client path resolution issue for the artifact storage location.
        Upon resolving the artifact storage location, the MLflow client will use the value provided by ``--default-artifact-root`` and suffixes the location with the values provided in the environment variable  ``MLFLOW_S3_ENDPOINT_URL``.
        Depending on the value set for the environment variable ``MLFLOW_S3_ENDPOINT_URL``, the resulting artifact storage path for this scenario would be one of the following invalid object store paths:  ``https://<bucketname>.s3.<region>.amazonaws.com/<key>/<bucketname>/<key>`` or  ``s3://<bucketname>/<key>/<bucketname>/<key>``.
        To prevent path parsing issues, **ensure that reserved environment variables are removed (``unset``) from client environments**.

Azure Blob Storage
------------------

To store artifacts in Azure Blob Storage, specify a URI of the form
``wasbs://<container>@<storage-account>.blob.core.windows.net/<path>``.
MLflow expects that your Azure Storage access credentials are located in the
``AZURE_STORAGE_CONNECTION_STRING`` and ``AZURE_STORAGE_ACCESS_KEY`` environment variables
or having your credentials configured such that the `DefaultAzureCredential()
<https://docs.microsoft.com/en-us/python/api/overview/azure/identity-readme?view=azure-python>`_. class can pick them up.
The order of precedence is:

#. ``AZURE_STORAGE_CONNECTION_STRING``
#. ``AZURE_STORAGE_ACCESS_KEY``
#. ``DefaultAzureCredential()``

You must set one of these options on **both your client application and your MLflow tracking server**.
Also, you must run ``pip install azure-storage-blob`` separately (on both your client and the server) to access Azure Blob Storage.
Finally, if you want to use DefaultAzureCredential, you must ``pip install azure-identity``;
MLflow does not declare a dependency on these packages by default.

You may set an MLflow environment variable to configure the timeout for artifact uploads and downloads:

- ``MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT`` - (Experimental, may be changed or removed) Sets the timeout for artifact upload/download in seconds (Default: 600 for Azure blob).

Google Cloud Storage
--------------------

To store artifacts in Google Cloud Storage, specify a URI of the form ``gs://<bucket>/<path>``.
You should configure credentials for accessing the GCS container on the client and server as described
in the `GCS documentation <https://google-cloud.readthedocs.io/en/latest/core/auth.html>`_.
Finally, you must run ``pip install google-cloud-storage`` (on both your client and the server)
to access Google Cloud Storage; MLflow does not declare a dependency on this package by default.



You may set some MLflow environment variables to troubleshoot GCS read-timeouts (eg. due to slow transfer speeds) using the following variables:

- ``MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT`` - (Experimental, may be changed or removed) Sets the standard timeout for transfer operations in seconds (Default: 60 for GCS). Use -1 for indefinite timeout.
- ``MLFLOW_GCS_DEFAULT_TIMEOUT`` - (Deprecated, please use ``MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT``) Sets the standard timeout for transfer operations in seconds (Default: 60). Use -1 for indefinite timeout.
- ``MLFLOW_GCS_UPLOAD_CHUNK_SIZE`` - Sets the standard upload chunk size for bigger files in bytes (Default: 104857600 ≙ 100MiB), must be multiple of 256 KB.
- ``MLFLOW_GCS_DOWNLOAD_CHUNK_SIZE`` - Sets the standard download chunk size for bigger files in bytes (Default: 104857600 ≙ 100MiB), must be multiple of 256 KB

FTP server
----------

To store artifacts in a FTP server, specify a URI of the form ftp://user@host/path/to/directory .
The URI may optionally include a password for logging into the server, e.g. ``ftp://user:pass@host/path/to/directory``

SFTP Server
-----------

To store artifacts in an SFTP server, specify a URI of the form ``sftp://user@host/path/to/directory``.
You should configure the client to be able to log in to the SFTP server without a password over SSH (e.g. public key, identity file in ssh_config, etc.).

The format ``sftp://user:pass@host/`` is supported for logging in. However, for safety reasons this is not recommended.

When using this store, ``pysftp`` must be installed on both the server and the client. Run ``pip install pysftp`` to install the required package.

NFS
---

To store artifacts in an NFS mount, specify a URI as a normal file system path, e.g., ``/mnt/nfs``.
This path must be the same on both the server and the client -- you may need to use symlinks or remount
the client in order to enforce this property.


HDFS
----

To store artifacts in HDFS, specify a ``hdfs:`` URI. It can contain host and port: ``hdfs://<host>:<port>/<path>`` or just the path: ``hdfs://<path>``.

There are also two ways to authenticate to HDFS:

- Use current UNIX account authorization
- Kerberos credentials using the following environment variables:

.. code-block:: bash

  export MLFLOW_KERBEROS_TICKET_CACHE=/tmp/krb5cc_22222222
  export MLFLOW_KERBEROS_USER=user_name_to_use

The HDFS artifact store is accessed using the ``pyarrow.fs`` module, refer to the
`PyArrow Documentation <https://arrow.apache.org/docs/python/filesystems.html#filesystem-hdfs>`_ for configuration and environment variables needed.


Deletion Behavior
=================
In order to allow MLflow Runs to be restored, Run metadata and artifacts are not automatically removed
from the backend store or artifact store when a Run is deleted. The :ref:`mlflow gc <cli>` CLI is provided
for permanently removing Run metadata and artifacts for deleted runs.

Multipart upload for proxied artifact access
============================================

.. note::
    This feature is experimental and may be changed or removed in a future release without notice.

Tracking Server supports uploading large artifacts using multipart upload for proxied artifact access.
To enable this feature, set ``MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD`` to ``true``.

.. code-block:: bash

    export MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=true

Under the hood, the Tracking Server will create a multipart upload request with the underlying storage,
generate presigned urls for each part, and let the client upload the parts directly to the storage.
Once all parts are uploaded, the Tracking Server will complete the multipart upload.
None of the data will pass through the Tracking Server.

If the underlying storage does not support multipart upload, the Tracking Server will fallback to a single part upload.
If multipart upload is supported but fails for any reason, an exception will be thrown.

MLflow supports multipart upload for the following storage for proxied artifact access:

- Amazon S3
- Google Cloud Storage

You can configure the following environment variables:

- ``MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE`` - Specifies the minimum file size in bytes to use multipart upload
  when logging artifacts (Default: 500 MB)
- ``MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE`` - Specifies the chunk size in bytes to use when performing multipart upload
  (Default: 100 MB)
