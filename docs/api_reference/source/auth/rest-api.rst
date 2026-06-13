
.. _auth-rest-api:

==============================
MLflow Authentication REST API
==============================


The MLflow Authentication REST API allows you to create, get, update, and delete users,
manage user permissions, and manage roles and role-based access control (RBAC).
The API supports both legacy ``2.0`` endpoints for user management and new ``3.0`` 
endpoints for unified permission and role management introduced in MLflow 3.13.0.
The API is hosted under the ``/api`` route on the MLflow tracking server. For example, to create
a user on a tracking server hosted at ``http://localhost:5000``, access
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

.. _mlflowAuthServiceListUsers:

List Users
==========

+-----------------------------+-------------+
|          Endpoint           | HTTP Method |
+=============================+=============+
| ``2.0/mlflow/users/list``   | ``GET``     |
+-----------------------------+-------------+

.. _mlflowListUsersResponse:

Response Structure
------------------

+------------+---------------------------+------------------+
| Field Name |           Type            |   Description    |
+============+===========================+==================+
| users      | An array of               | A list of all    |
|            | :ref:`mlflowUser`         | user objects.    |
+------------+---------------------------+------------------+

===========================

.. _mlflowAuthServiceGetCurrentUser:

Get Current User
================

+--------------------------------+-------------+
|           Endpoint             | HTTP Method |
+================================+=============+
| ``2.0/mlflow/users/current``   | ``GET``     |
+--------------------------------+-------------+

.. _mlflowGetCurrentUserResponse:

Response Structure
------------------

+------------+-------------------+------------------------------+
| Field Name |       Type        |         Description          |
+============+===================+==============================+
| user       | :ref:`mlflowUser` | The current user object.     |
+------------+-------------------+------------------------------+

===========================

.. _mlflowAuthServiceGrantUserPermission:

Grant User Permission
=====================

+------------------------------------------+-------------+
|                 Endpoint                 | HTTP Method |
+==========================================+=============+
| ``3.0/mlflow/users/permissions/grant``   | ``POST``    |
+------------------------------------------+-------------+

.. _mlflowGrantUserPermission:

Request Structure
-----------------

+---------------+------------+---------------------------+
|  Field Name   |    Type    |        Description        |
+===============+============+===========================+
| username      | ``STRING`` | Username.                 |
+---------------+------------+---------------------------+
| resource_type | ``STRING`` | Resource type             |
|               |            | (``experiment`` or        |
|               |            | ``registered_model``).    |
+---------------+------------+---------------------------+
| resource_id   | ``STRING`` | Resource ID or name.      |
+---------------+------------+---------------------------+
| permission    | ``STRING`` | Permission to grant       |
|               |            | (``READ``, ``EDIT``,      |
|               |            | ``MANAGE``,               |
|               |            | ``NO_PERMISSIONS``).      |
+---------------+------------+---------------------------+

===========================

.. _mlflowAuthServiceRevokeUserPermission:

Revoke User Permission
======================

+------------------------------------------+-------------+
|                 Endpoint                 | HTTP Method |
+==========================================+=============+
| ``3.0/mlflow/users/permissions/revoke``  | ``POST``    |
+------------------------------------------+-------------+

.. _mlflowRevokeUserPermission:

Request Structure
-----------------

+---------------+------------+---------------------------+
|  Field Name   |    Type    |        Description        |
+===============+============+===========================+
| username      | ``STRING`` | Username.                 |
+---------------+------------+---------------------------+
| resource_type | ``STRING`` | Resource type             |
|               |            | (``experiment`` or        |
|               |            | ``registered_model``).    |
+---------------+------------+---------------------------+
| resource_id   | ``STRING`` | Resource ID or name.      |
+---------------+------------+---------------------------+

===========================

.. _mlflowAuthServiceGetUserPermission:

Get User Permission
===================

+------------------------------------------+-------------+
|                 Endpoint                 | HTTP Method |
+==========================================+=============+
| ``3.0/mlflow/users/permissions/get``     | ``GET``     |
+------------------------------------------+-------------+

.. _mlflowGetUserPermission:

Request Structure
-----------------

+---------------+------------+---------------------------+
|  Field Name   |    Type    |        Description        |
+===============+============+===========================+
| username      | ``STRING`` | Username.                 |
+---------------+------------+---------------------------+
| resource_type | ``STRING`` | Resource type             |
|               |            | (``experiment`` or        |
|               |            | ``registered_model``).    |
+---------------+------------+---------------------------+
| resource_id   | ``STRING`` | Resource ID or name.      |
+---------------+------------+---------------------------+

.. _mlflowGetUserPermissionResponse:

Response Structure
------------------

+------------+------------+------------------------------+
| Field Name |    Type    |         Description          |
+============+============+==============================+
| permission | ``STRING`` | The effective permission     |
|            |            | for the user on the          |
|            |            | specified resource.          |
+------------+------------+------------------------------+

===========================

.. _mlflowAuthServiceListUserPermissions:

List User Permissions
=====================

+------------------------------------------+-------------+
|                 Endpoint                 | HTTP Method |
+==========================================+=============+
| ``3.0/mlflow/users/permissions/list``    | ``GET``     |
+------------------------------------------+-------------+

.. _mlflowListUserPermissions:

Request Structure
-----------------

+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| username   | ``STRING`` | Username.   |
+------------+------------+-------------+

.. _mlflowListUserPermissionsResponse:

Response Structure
------------------

+-------------+------------+--------------------------------+
|  Field Name |    Type    |          Description           |
+=============+============+================================+
| permissions | ``ARRAY``  | List of permissions for        |
|             |            | the user across all resources. |
+-------------+------------+--------------------------------+

===========================

.. _mlflowAuthServiceListCurrentUserPermissions:

List Current User Permissions
==============================

+--------------------------------------------------+-------------+
|                    Endpoint                      | HTTP Method |
+==================================================+=============+
| ``3.0/mlflow/users/current/permissions``         | ``GET``     |
+--------------------------------------------------+-------------+

.. _mlflowListCurrentUserPermissionsResponse:

Response Structure
------------------

+-------------+------------+----------------------------------------+
|  Field Name |    Type    |              Description               |
+=============+============+========================================+
| permissions | ``ARRAY``  | List of permissions for the            |
|             |            | currently authenticated user.          |
+-------------+------------+----------------------------------------+

===========================

.. _mlflowAuthServiceCreateRole:

Create Role
===========

+--------------------------------+-------------+
|           Endpoint             | HTTP Method |
+================================+=============+
| ``3.0/mlflow/roles/create``    | ``POST``    |
+--------------------------------+-------------+

.. _mlflowCreateRole:

Request Structure
-----------------

+-------------+------------+----------------------+
| Field Name  |    Type    |     Description      |
+=============+============+======================+
| name        | ``STRING`` | Role name.           |
+-------------+------------+----------------------+
| description | ``STRING`` | Role description.    |
+-------------+------------+----------------------+

.. _mlflowCreateRoleResponse:

Response Structure
------------------

+------------+--------------------+----------------+
| Field Name |        Type        |  Description   |
+============+====================+================+
| role       | :ref:`mlflowRole`  | A role object. |
+------------+--------------------+----------------+

===========================

.. _mlflowAuthServiceGetRole:

Get Role
========

+-----------------------------+-------------+
|          Endpoint           | HTTP Method |
+=============================+=============+
| ``3.0/mlflow/roles/get``    | ``GET``     |
+-----------------------------+-------------+

.. _mlflowGetRole:

Request Structure
-----------------

+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| name       | ``STRING`` | Role name.  |
+------------+------------+-------------+

.. _mlflowGetRoleResponse:

Response Structure
------------------

+------------+--------------------+----------------+
| Field Name |        Type        |  Description   |
+============+====================+================+
| role       | :ref:`mlflowRole`  | A role object. |
+------------+--------------------+----------------+

===========================

.. _mlflowAuthServiceListRoles:

List Roles
==========

+-----------------------------+-------------+
|          Endpoint           | HTTP Method |
+=============================+=============+
| ``3.0/mlflow/roles/list``   | ``GET``     |
+-----------------------------+-------------+

.. _mlflowListRolesResponse:

Response Structure
------------------

+------------+---------------------------+------------------+
| Field Name |           Type            |   Description    |
+============+===========================+==================+
| roles      | An array of               | A list of all    |
|            | :ref:`mlflowRole`         | role objects.    |
+------------+---------------------------+------------------+

===========================

.. _mlflowAuthServiceUpdateRole:

Update Role
===========

+------------------------------+-------------+
|           Endpoint           | HTTP Method |
+==============================+=============+
| ``3.0/mlflow/roles/update``  | ``PATCH``   |
+------------------------------+-------------+

.. _mlflowUpdateRole:

Request Structure
-----------------

+-------------+------------+----------------------+
| Field Name  |    Type    |     Description      |
+=============+============+======================+
| name        | ``STRING`` | Role name.           |
+-------------+------------+----------------------+
| description | ``STRING`` | New role description.|
+-------------+------------+----------------------+

===========================

.. _mlflowAuthServiceDeleteRole:

Delete Role
===========

+------------------------------+-------------+
|           Endpoint           | HTTP Method |
+==============================+=============+
| ``3.0/mlflow/roles/delete``  | ``DELETE``  |
+------------------------------+-------------+

.. _mlflowDeleteRole:

Request Structure
-----------------

+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| name       | ``STRING`` | Role name.  |
+------------+------------+-------------+

===========================

.. _mlflowAuthServiceAssignRole:

Assign Role
===========

+------------------------------+-------------+
|           Endpoint           | HTTP Method |
+==============================+=============+
| ``3.0/mlflow/roles/assign``  | ``POST``    |
+------------------------------+-------------+

.. _mlflowAssignRole:

Request Structure
-----------------

+------------+------------+-------------------+
| Field Name |    Type    |    Description    |
+============+============+===================+
| username   | ``STRING`` | Username.         |
+------------+------------+-------------------+
| role_name  | ``STRING`` | Role to assign.   |
+------------+------------+-------------------+

===========================

.. _mlflowAuthServiceUnassignRole:

Unassign Role
=============

+--------------------------------+-------------+
|           Endpoint             | HTTP Method |
+================================+=============+
| ``3.0/mlflow/roles/unassign``  | ``POST``    |
+--------------------------------+-------------+

.. _mlflowUnassignRole:

Request Structure
-----------------

+------------+------------+--------------------+
| Field Name |    Type    |    Description     |
+============+============+====================+
| username   | ``STRING`` | Username.          |
+------------+------------+--------------------+
| role_name  | ``STRING`` | Role to unassign.  |
+------------+------------+--------------------+

===========================

.. _mlflowAuthServiceAddRolePermission:

Add Role Permission
===================

+------------------------------------------+-------------+
|                 Endpoint                 | HTTP Method |
+==========================================+=============+
| ``3.0/mlflow/roles/permissions/add``     | ``POST``    |
+------------------------------------------+-------------+

.. _mlflowAddRolePermission:

Request Structure
-----------------

+---------------+------------+---------------------------+
|  Field Name   |    Type    |        Description        |
+===============+============+===========================+
| role_name     | ``STRING`` | Role name.                |
+---------------+------------+---------------------------+
| resource_type | ``STRING`` | Resource type             |
|               |            | (``experiment`` or        |
|               |            | ``registered_model``).    |
+---------------+------------+---------------------------+
| resource_id   | ``STRING`` | Resource ID or name.      |
+---------------+------------+---------------------------+
| permission    | ``STRING`` | Permission to add.        |
+---------------+------------+---------------------------+

===========================

.. _mlflowAuthServiceRemoveRolePermission:

Remove Role Permission
======================

+------------------------------------------+-------------+
|                 Endpoint                 | HTTP Method |
+==========================================+=============+
| ``3.0/mlflow/roles/permissions/remove``  | ``POST``    |
+------------------------------------------+-------------+

.. _mlflowRemoveRolePermission:

Request Structure
-----------------

+---------------+------------+---------------------------+
|  Field Name   |    Type    |        Description        |
+===============+============+===========================+
| role_name     | ``STRING`` | Role name.                |
+---------------+------------+---------------------------+
| resource_type | ``STRING`` | Resource type.            |
+---------------+------------+---------------------------+
| resource_id   | ``STRING`` | Resource ID or name.      |
+---------------+------------+---------------------------+

===========================

.. _mlflowAuthServiceListRolePermissions:

List Role Permissions
=====================

+------------------------------------------+-------------+
|                 Endpoint                 | HTTP Method |
+==========================================+=============+
| ``3.0/mlflow/roles/permissions/list``    | ``GET``     |
+------------------------------------------+-------------+

.. _mlflowListRolePermissions:

Request Structure
-----------------

+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| role_name  | ``STRING`` | Role name.  |
+------------+------------+-------------+

.. _mlflowListRolePermissionsResponse:

Response Structure
------------------

+-------------+-----------+----------------------------------+
|  Field Name |   Type    |           Description            |
+=============+===========+==================================+
| permissions | ``ARRAY`` | List of permissions for the      |
|             |           | role across all resources.       |
+-------------+-----------+----------------------------------+

===========================

.. _mlflowAuthServiceUpdateRolePermission:

Update Role Permission
======================

+------------------------------------------+-------------+
|                 Endpoint                 | HTTP Method |
+==========================================+=============+
| ``3.0/mlflow/roles/permissions/update``  | ``PATCH``   |
+------------------------------------------+-------------+

.. _mlflowUpdateRolePermission:

Request Structure
-----------------

+---------------+------------+---------------------------+
|  Field Name   |    Type    |        Description        |
+===============+============+===========================+
| role_name     | ``STRING`` | Role name.                |
+---------------+------------+---------------------------+
| resource_type | ``STRING`` | Resource type.            |
+---------------+------------+---------------------------+
| resource_id   | ``STRING`` | Resource ID or name.      |
+---------------+------------+---------------------------+
| permission    | ``STRING`` | New permission.           |
+---------------+------------+---------------------------+

===========================

.. _mlflowAuthServiceListUserRoles:

List User Roles
===============

+----------------------------------+-------------+
|            Endpoint              | HTTP Method |
+==================================+=============+
| ``3.0/mlflow/users/roles/list``  | ``GET``     |
+----------------------------------+-------------+

.. _mlflowListUserRoles:

Request Structure
-----------------

+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| username   | ``STRING`` | Username.   |
+------------+------------+-------------+

.. _mlflowListUserRolesResponse:

Response Structure
------------------

+------------+---------------------------+----------------------+
| Field Name |           Type            |     Description      |
+============+===========================+======================+
| roles      | An array of               | List of roles        |
|            | :ref:`mlflowRole`         | assigned to user.    |
+------------+---------------------------+----------------------+

===========================

.. _mlflowAuthServiceListRoleUsers:

List Role Users
===============

+----------------------------------+-------------+
|            Endpoint              | HTTP Method |
+==================================+=============+
| ``3.0/mlflow/roles/users/list``  | ``GET``     |
+----------------------------------+-------------+

.. _mlflowListRoleUsers:

Request Structure
-----------------

+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| role_name  | ``STRING`` | Role name.  |
+------------+------------+-------------+

.. _mlflowListRoleUsersResponse:

Response Structure
------------------

+------------+---------------------------+----------------------+
| Field Name |           Type            |     Description      |
+============+===========================+======================+
| users      | An array of               | List of users        |
|            | :ref:`mlflowUser`         | assigned to role.    |
+------------+---------------------------+----------------------+

===========================

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
| roles                        | An array of :ref:`mlflowRole`                      | Roles assigned to the user.                                      |
+------------------------------+----------------------------------------------------+------------------------------------------------------------------+

.. _mlflowPermission:

Permission
----------

Permission level for a user on a resource.

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

.. _mlflowRole:

Role
----

+-------------+------------+---------------------------+
| Field Name  |    Type    |        Description        |
+=============+============+===========================+
| name        | ``STRING`` | Role name.                |
+-------------+------------+---------------------------+
| description | ``STRING`` | Role description.         |
+-------------+------------+---------------------------+
| permissions | ``ARRAY``  | List of permissions       |
|             |            | associated with the role. |
+-------------+------------+---------------------------+