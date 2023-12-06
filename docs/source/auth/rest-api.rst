
.. _auth-rest-api:

==============================
MLflow Authentication REST API
==============================


The MLflow Authentication REST API allows you to create, get, update and delete users, 
experiment permissions and registered model permissions.
The API is hosted under the ``/api`` route on the MLflow tracking server. For example, to list
experiments on a tracking server hosted at ``http://localhost:5000``, access
``http://localhost:5000/api/2.0/mlflow/users/create``.

.. important::
    The MLflow REST API requires content type ``application/json`` for all POST requests.

.. contents:: Table of Contents
    :local:
    :depth: 1

===========================

.. _mlflowAuthServiceCreateUser:

Create User
===========

+-----------------------------+-------------+
|          Endpoint           | HTTP Method |
+=============================+=============+
| ``2.0/mlflow/users/create`` | ``POST``    |
+-----------------------------+-------------+

.. _mlflowCreateUser:

Request Structure
-----------------

+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| username   | ``STRING`` | Username.   |
+------------+------------+-------------+
| password   | ``STRING`` | Password.   |
+------------+------------+-------------+

.. _mlflowCreateUserResponse:

Response Structure
------------------

+------------+-------------------+----------------+
| Field Name |       Type        |  Description   |
+============+===================+================+
| user       | :ref:`mlflowUser` | A user object. |
+------------+-------------------+----------------+

===========================

.. _mlflowAuthServiceGetUser:

Get User
========

+--------------------------+-------------+
|         Endpoint         | HTTP Method |
+==========================+=============+
| ``2.0/mlflow/users/get`` | ``GET``     |
+--------------------------+-------------+

.. _mlflowGetUser:

Request Structure
-----------------

+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| username   | ``STRING`` | Username.   |
+------------+------------+-------------+

.. _mlflowGetUserResponse:

Response Structure
------------------

+------------+-------------------+----------------+
| Field Name |       Type        |  Description   |
+============+===================+================+
| user       | :ref:`mlflowUser` | A user object. |
+------------+-------------------+----------------+

===========================

.. _mlflowAuthServiceUpdateUserPassword:

Update User Password
====================

+--------------------------------------+-------------+
|               Endpoint               | HTTP Method |
+======================================+=============+
| ``2.0/mlflow/users/update-password`` | ``PATCH``   |
+--------------------------------------+-------------+

.. _mlflowUpdateUserPassword:

Request Structure
-----------------

+------------+------------+---------------+
| Field Name | Type       | Description   |
+============+============+===============+
| username   | ``STRING`` | Username.     |
+------------+------------+---------------+
| password   | ``STRING`` | New password. |
+------------+------------+---------------+

===========================

.. _mlflowAuthServiceUpdateUserAdmin:

Update User Admin
=================

+-----------------------------------+-------------+
|             Endpoint              | HTTP Method |
+===================================+=============+
| ``2.0/mlflow/users/update-admin`` | ``PATCH``   |
+-----------------------------------+-------------+

.. _mlflowUpdateUserAdmin:

Request Structure
-----------------

+------------+-------------+-------------------+
| Field Name |    Type     |    Description    |
+============+=============+===================+
| username   | ``STRING``  | Username.         |
+------------+-------------+-------------------+
| is_admin   | ``BOOLEAN`` | New admin status. |
+------------+-------------+-------------------+

===========================

.. _mlflowAuthServiceDeleteUser:

Delete User
===========

+-----------------------------+-------------+
|          Endpoint           | HTTP Method |
+=============================+=============+
| ``2.0/mlflow/users/delete`` | ``DELETE``  |
+-----------------------------+-------------+

.. _mlflowDeleteUser:

Request Structure
-----------------

+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| username   | ``STRING`` | Username.   |
+------------+------------+-------------+

===========================

.. _mlflowAuthServiceCreateExperimentPermission:

Create Experiment Permission
============================

+-----------------------------------------------+-------------+
|                   Endpoint                    | HTTP Method |
+===============================================+=============+
| ``2.0/mlflow/experiments/permissions/create`` | ``POST``    |
+-----------------------------------------------+-------------+

.. _mlflowCreateExperimentPermission:

Request Structure
-----------------

+---------------+-------------------------+----------------------+
|  Field Name   |          Type           |     Description      |
+===============+=========================+======================+
| experiment_id | ``STRING``              | Experiment id.       |
+---------------+-------------------------+----------------------+
| username      | ``STRING``              | Username.            |
+---------------+-------------------------+----------------------+
| permission    | :ref:`mlflowPermission` | Permission to grant. |
+---------------+-------------------------+----------------------+

.. _mlflowCreateExperimentPermissionResponse:

Response Structure
------------------

+-----------------------+-----------------------------------+----------------------------------+
|      Field Name       |               Type                |           Description            |
+=======================+===================================+==================================+
| experiment_permission | :ref:`mlflowExperimentPermission` | An experiment permission object. |
+-----------------------+-----------------------------------+----------------------------------+

===========================

.. _mlflowAuthServiceGetExperimentPermission:

Get Experiment Permission
=========================

+--------------------------------------------+-------------+
|                  Endpoint                  | HTTP Method |
+============================================+=============+
| ``2.0/mlflow/experiments/permissions/get`` | ``GET``     |
+--------------------------------------------+-------------+

.. _mlflowGetExperimentPermission:

Request Structure
-----------------

+---------------+------------+----------------+
|  Field Name   |    Type    |  Description   |
+===============+============+================+
| experiment_id | ``STRING`` | Experiment id. |
+---------------+------------+----------------+
| username      | ``STRING`` | Username.      |
+---------------+------------+----------------+

.. _mlflowGetExperimentPermissionResponse:

Response Structure
------------------

+-----------------------+-----------------------------------+----------------------------------+
|      Field Name       |               Type                |           Description            |
+=======================+===================================+==================================+
| experiment_permission | :ref:`mlflowExperimentPermission` | An experiment permission object. |
+-----------------------+-----------------------------------+----------------------------------+

===========================

.. _mlflowAuthServiceUpdateExperimentPermission:

Update Experiment Permission
============================

+-----------------------------------------------+-------------+
|                   Endpoint                    | HTTP Method |
+===============================================+=============+
| ``2.0/mlflow/experiments/permissions/update`` | ``PATCH``   |
+-----------------------------------------------+-------------+

.. _mlflowUpdateExperimentPermission:

Request Structure
-----------------

+---------------+-------------------------+--------------------------+
|  Field Name   |          Type           |       Description        |
+===============+=========================+==========================+
| experiment_id | ``STRING``              | Experiment id.           |
+---------------+-------------------------+--------------------------+
| username      | ``STRING``              | Username.                |
+---------------+-------------------------+--------------------------+
| permission    | :ref:`mlflowPermission` | New permission to grant. |
+---------------+-------------------------+--------------------------+

===========================

.. _mlflowAuthServiceDeleteExperimentPermission:

Delete Experiment Permission
============================

+-----------------------------------------------+-------------+
|                   Endpoint                    | HTTP Method |
+===============================================+=============+
| ``2.0/mlflow/experiments/permissions/delete`` | ``DELETE``  |
+-----------------------------------------------+-------------+

.. _mlflowDeleteExperimentPermission:

Request Structure
-----------------

+---------------+------------+----------------+
|  Field Name   |    Type    |  Description   |
+===============+============+================+
| experiment_id | ``STRING`` | Experiment id. |
+---------------+------------+----------------+
| username      | ``STRING`` | Username.      |
+---------------+------------+----------------+

===========================

.. _mlflowAuthServiceCreateRegisteredModelPermission:

Create Registered Model Permission
==================================

+-----------------------------------------------------+-------------+
|                      Endpoint                       | HTTP Method |
+=====================================================+=============+
| ``2.0/mlflow/registered-models/permissions/create`` | ``CREATE``  |
+-----------------------------------------------------+-------------+

.. _mlflowCreateRegisteredModelPermission:

Request Structure
-----------------

+------------+-------------------------+------------------------+
| Field Name |          Type           |      Description       |
+============+=========================+========================+
| name       | ``STRING``              | Registered model name. |
+------------+-------------------------+------------------------+
| username   | ``STRING``              | Username.              |
+------------+-------------------------+------------------------+
| permission | :ref:`mlflowPermission` | Permission to grant.   |
+------------+-------------------------+------------------------+

.. _mlflowCreateRegisteredModelPermissionResponse:

Response Structure
------------------

+-----------------------------+----------------------------------------+---------------------------------------+
|         Field Name          |                  Type                  |              Description              |
+=============================+========================================+=======================================+
| registered_model_permission | :ref:`mlflowRegisteredModelPermission` | A registered model permission object. |
+-----------------------------+----------------------------------------+---------------------------------------+

===========================

.. _mlflowAuthServiceGetRegisteredModelPermission:

Get Registered Model Permission
===============================

+--------------------------------------------------+-------------+
|                     Endpoint                     | HTTP Method |
+==================================================+=============+
| ``2.0/mlflow/registered-models/permissions/get`` | ``GET``     |
+--------------------------------------------------+-------------+

.. _mlflowGetRegisteredModelPermission:

Request Structure
-----------------

+------------+------------+------------------------+
| Field Name |    Type    |      Description       |
+============+============+========================+
| name       | ``STRING`` | Registered model name. |
+------------+------------+------------------------+
| username   | ``STRING`` | Username.              |
+------------+------------+------------------------+

.. _mlflowGetRegisteredModelPermissionResponse:

Response Structure
------------------

+-----------------------------+----------------------------------------+---------------------------------------+
|         Field Name          |                  Type                  |              Description              |
+=============================+========================================+=======================================+
| registered_model_permission | :ref:`mlflowRegisteredModelPermission` | A registered model permission object. |
+-----------------------------+----------------------------------------+---------------------------------------+

===========================

.. _mlflowAuthServiceUpdateRegisteredModelPermission:

Update Registered Model Permission
==================================

+-----------------------------------------------------+-------------+
|                      Endpoint                       | HTTP Method |
+=====================================================+=============+
| ``2.0/mlflow/registered-models/permissions/update`` | ``PATCH``   |
+-----------------------------------------------------+-------------+

.. _mlflowUpdateRegisteredModelPermission:

Request Structure
-----------------

+------------+-------------------------+--------------------------+
| Field Name |          Type           |       Description        |
+============+=========================+==========================+
| name       | ``STRING``              | Registered model name.   |
+------------+-------------------------+--------------------------+
| username   | ``STRING``              | Username.                |
+------------+-------------------------+--------------------------+
| permission | :ref:`mlflowPermission` | New permission to grant. |
+------------+-------------------------+--------------------------+

===========================

.. _mlflowAuthServiceDeleteRegisteredModelPermission:

Delete Registered Model Permission
==================================

+-----------------------------------------------------+-------------+
|                      Endpoint                       | HTTP Method |
+=====================================================+=============+
| ``2.0/mlflow/registered-models/permissions/delete`` | ``DELETE``  |
+-----------------------------------------------------+-------------+

.. _mlflowDeleteRegisteredModelPermission:

Request Structure
-----------------

+------------+------------+------------------------+
| Field Name |    Type    |      Description       |
+============+============+========================+
| name       | ``STRING`` | Registered model name. |
+------------+------------+------------------------+
| username   | ``STRING`` | Username.              |
+------------+------------+------------------------+


.. _auth-rest-struct:

Data Structures
===============


.. _mlflowUser:

User
----

+------------------------------+----------------------------------------------------+------------------------------------------------------------------+
|          Field Name          |                        Type                        |                            Description                           |
+==============================+====================================================+==================================================================+
| id                           | ``STRING``                                         | User ID.                                                         |
+------------------------------+----------------------------------------------------+------------------------------------------------------------------+
| username                     | ``STRING``                                         | Username.                                                        |
+------------------------------+----------------------------------------------------+------------------------------------------------------------------+
| is_admin                     | ``BOOLEAN``                                        | Whether the user is an admin.                                    |
+------------------------------+----------------------------------------------------+------------------------------------------------------------------+
| experiment_permissions       | An array of :ref:`mlflowExperimentPermission`      | All experiment permissions explicitly granted to the user.       |
+------------------------------+----------------------------------------------------+------------------------------------------------------------------+
| registered_model_permissions | An array of :ref:`mlflowRegisteredModelPermission` | All registered model permissions explicitly granted to the user. |
+------------------------------+----------------------------------------------------+------------------------------------------------------------------+

.. _mlflowPermission:

Permission
----------

Permission of a user to an experiment or a registered model.

+----------------+--------------------------------------+
|      Name      |             Description              |
+================+======================================+
| READ           | Can read.                            |
+----------------+--------------------------------------+
| EDIT           | Can read and update.                 |
+----------------+--------------------------------------+
| MANAGE         | Can read, update, delete and manage. |
+----------------+--------------------------------------+
| NO_PERMISSIONS | No permissions.                      |
+----------------+--------------------------------------+

.. _mlflowExperimentPermission:

ExperimentPermission
--------------------

+---------------+-------------------------+---------------------+
|  Field Name   |          Type           |     Description     |
+===============+=========================+=====================+
| experiment_id | ``STRING``              | Experiment id.      |
+---------------+-------------------------+---------------------+
| user_id       | ``STRING``              | User id.            |
+---------------+-------------------------+---------------------+
| permission    | :ref:`mlflowPermission` | Permission granted. |
+---------------+-------------------------+---------------------+

.. _mlflowRegisteredModelPermission:

RegisteredModelPermission
-------------------------

+------------+-------------------------+------------------------+
| Field Name |          Type           |      Description       |
+============+=========================+========================+
| name       | ``STRING``              | Registered model name. |
+------------+-------------------------+------------------------+
| user_id    | ``STRING``              | User id.               |
+------------+-------------------------+------------------------+
| permission | :ref:`mlflowPermission` | Permission granted.    |
+------------+-------------------------+------------------------+
