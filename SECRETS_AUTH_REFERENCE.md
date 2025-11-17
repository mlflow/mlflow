# Secrets Authentication & Authorization Reference

This document provides a complete reference for the authentication and authorization (auth) implementation for the MLflow Secrets feature. It is intended for documentation work and UI development.

---

## Overview

The Secrets feature uses MLflow's built-in RBAC (Role-Based Access Control) system to control access to secrets and their operations. Permissions are assigned per-secret, per-user and enforce fine-grained access control across all secret operations.

**Key Concepts:**
- **Permissions** are assigned to individual users for specific secrets
- **Permission levels** define what operations a user can perform
- **Default permission** applies when no explicit permission is set (configurable, default: `READ`)
- **Admin users** bypass all permission checks
- **Auto-grant**: Users who create a secret automatically receive `MANAGE` permission

---

## Permission Levels

MLflow Secrets uses four permission levels:

| Permission | can_read | can_update | can_delete | can_manage | Description |
|------------|----------|------------|------------|------------|-------------|
| `READ` | ✅ | ❌ | ❌ | ❌ | View secret metadata and bindings only |
| `EDIT` | ✅ | ✅ | ❌ | ❌ | Read + update secret value, create routes/bindings |
| `MANAGE` | ✅ | ✅ | ✅ | ✅ | Full control: read, update, delete, manage permissions |
| `NO_PERMISSIONS` | ❌ | ❌ | ❌ | ❌ | No access to the secret |

**Capability Details:**

- **`can_read`**: View secret info, list bindings
- **`can_update`**: Update secret value, bind to resources, create new routes, unbind from resources
- **`can_delete`**: Delete secrets (requires MANAGE)
- **`can_manage`**: Modify secret value, delete secrets, grant/revoke permissions to other users

---

## API Endpoints and Required Permissions

### Secret Operations

| Endpoint | HTTP Method | Permission Required | Validator | Notes |
|----------|-------------|---------------------|-----------|-------|
| **CreateAndBindSecret** | `POST /api/3.0/mlflow/secrets/create-and-bind` | None (creates new) | N/A | Auto-grants `MANAGE` to creator |
| **GetSecretInfo** | `GET /api/3.0/mlflow/secrets/get-info` | `READ` | `validate_can_read_secret` | Returns metadata only (not value) |
| **ListSecrets** | `GET /api/3.0/mlflow/secrets/list` | `READ` (per-secret) | None | Filters results based on user permissions |
| **UpdateSecret** | `POST /api/3.0/mlflow/secrets/update` | `MANAGE` | `validate_can_manage_secret` | Updates secret value |
| **DeleteSecret** | `DELETE /api/3.0/mlflow/secrets/delete` | `MANAGE` | `validate_can_manage_secret` | Deletes secret and all bindings |

### Route & Binding Operations

| Endpoint | HTTP Method | Permission Required | Validator | Notes |
|----------|-------------|---------------------|-----------|-------|
| **BindSecret** | `POST /api/3.0/mlflow/secrets/bind` | `EDIT` | `validate_can_update_secret` | Bind existing secret to resource |
| **CreateRouteAndBind** | `POST /api/3.0/mlflow/secrets/create-route-and-bind` | `EDIT` | `validate_can_update_secret` | Create new route for existing secret |
| **UnbindSecret** | `POST /api/3.0/mlflow/secrets/unbind` | `EDIT` | `validate_can_update_secret_binding` | Unbind secret from resource |
| **ListSecretBindings** | `GET /api/3.0/mlflow/secrets/list-bindings` | `READ` | `validate_can_read_secret` | List all bindings for a secret |

### Permission Management APIs

| Endpoint | HTTP Method | Permission Required | Notes |
|----------|-------------|---------------------|-------|
| **CreateSecretPermission** | `POST /api/2.0/mlflow/secrets/permissions/create` | `MANAGE` | Grant permission to another user |
| **GetSecretPermission** | `GET /api/2.0/mlflow/secrets/permissions/get` | `MANAGE` | View permissions for a user |
| **UpdateSecretPermission** | `POST /api/2.0/mlflow/secrets/permissions/update` | `MANAGE` | Update user's permission level |
| **DeleteSecretPermission** | `DELETE /api/2.0/mlflow/secrets/permissions/delete` | `MANAGE` | Revoke user's permission |

---

## Permission Validators

The auth layer uses three validator functions:

### `validate_can_read_secret()`
- **Used by**: `GetSecretInfo`, `ListSecretBindings`
- **Checks**: `permission.can_read == True`
- **Extracts**: `secret_id` from request parameters
- **Returns**: 403 if user lacks READ permission

### `validate_can_update_secret()`
- **Used by**: `BindSecret`, `CreateRouteAndBind`
- **Checks**: `permission.can_update == True`
- **Extracts**: `secret_id` from request parameters
- **Returns**: 403 if user lacks EDIT permission
- **Note**: Used for operations that create NEW bindings/routes using an existing secret

### `validate_can_update_secret_binding()`
- **Used by**: `UnbindSecret`
- **Checks**: `permission.can_update == True`
- **Extracts**: `secret_id` by looking up existing binding via `(resource_type, resource_id, field_name)`
- **Returns**: 403 if user lacks EDIT permission, or 404 if binding doesn't exist
- **Note**: Used for operations on EXISTING bindings

### `validate_can_manage_secret()`
- **Used by**: `UpdateSecret`, `DeleteSecret`
- **Checks**: `permission.can_manage == True`
- **Extracts**: `secret_id` from request parameters
- **Returns**: 403 if user lacks MANAGE permission

---

## Special Behaviors

### 1. Auto-Grant on Creation

When a user creates a secret via `CreateAndBindSecret`:
- The creating user is automatically granted `MANAGE` permission
- This happens in the `set_can_manage_secret_permission()` after-request handler
- Admin users do NOT receive auto-grant (they already have full access)

**Implementation**: `mlflow/server/auth/__init__.py:965-973`

```python
def set_can_manage_secret_permission(resp: Response):
    if sender_is_admin():
        return

    response_message = CreateAndBindSecret.Response()
    parse_dict(resp.json, response_message)
    secret_id = response_message.secret.secret_id
    username = authenticate_request().username
    store.create_secret_permission(secret_id, username, MANAGE.name)
```

### 2. Admin Bypass

**Admin users bypass ALL permission checks:**
- No validators are executed for admin users
- Admins can perform any operation on any secret
- Admins are not auto-granted permissions (unnecessary)
- Admin status checked via: `store.get_user(username).is_admin`

**Implementation**: `mlflow/server/auth/__init__.py:430-433`

### 3. List Filtering

The `ListSecrets` endpoint does NOT have a before-request validator. Instead, it uses an **after-request filter**:
- All secrets are fetched from the database
- Results are filtered based on user's READ permissions
- Secrets the user cannot read are removed from response
- Admin users see all secrets without filtering

**Implementation**: `mlflow/server/auth/__init__.py:927-951`

```python
def filter_list_secrets(resp: Response):
    """Filter out unreadable secrets from the list results."""
    if sender_is_admin():
        return

    username = authenticate_request().username
    perms = store.list_secret_permissions(username)
    can_read = {p.secret_id: get_permission(p.permission).can_read for p in perms}
    default_can_read = get_permission(auth_config.default_permission).can_read

    # filter out unreadable
    for secret in list(response_message.secrets):
        if not can_read.get(secret.secret_id, default_can_read):
            response_message.secrets.remove(secret)
```

### 4. Default Permission

When a user has no explicit permission for a secret, the **default permission** applies:
- Configurable in auth config file (default: `READ`)
- Typically set to `READ` to allow discovery of shared secrets
- Can be set to `NO_PERMISSIONS` for stricter security
- Configured in: `mlflow/server/auth/basic_auth.ini`

---

## Permission Operation Matrix

| Operation | None/Owner | READ | EDIT | MANAGE | Admin |
|-----------|------------|------|------|--------|-------|
| **Create secret** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **View secret info** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **List secret bindings** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Update secret value** | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Delete secret** | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Bind to resource** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Unbind from resource** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Create new route** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Grant permissions** | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Revoke permissions** | ❌ | ❌ | ❌ | ✅ | ✅ |

---

## Common Use Cases

### Scenario 1: Data Scientist Creates Secret

1. Alice creates a secret for OpenAI API key: `POST /api/3.0/mlflow/secrets/create-and-bind`
2. Alice is auto-granted `MANAGE` permission
3. Alice can update the secret value, bind it to jobs, and share with team members

### Scenario 2: Team Member Needs Access

1. Alice (MANAGE) grants Bob `EDIT` permission: `POST /api/2.0/mlflow/secrets/permissions/create`
2. Bob can now:
   - View the secret info (not the value)
   - Bind the secret to his own scorer jobs
   - Create new routes for different models
3. Bob CANNOT:
   - Update the secret value
   - Delete the secret
   - Grant permissions to others

### Scenario 3: Read-Only Access

1. Alice grants Carol `READ` permission
2. Carol can:
   - View secret metadata (name, provider, created_at, etc.)
   - List where the secret is bound
   - Discover shared secrets in her organization
3. Carol CANNOT:
   - Use the secret in her own jobs
   - Modify or delete the secret

### Scenario 4: Revoking Access

1. Alice revokes Bob's permission: `DELETE /api/2.0/mlflow/secrets/permissions/delete`
2. Bob loses access immediately (falls back to default permission)
3. Existing bindings Bob created remain active (bindings are not user-scoped)

### Scenario 5: Shared Organizational Secret

1. Admin creates a shared OpenAI secret
2. Default permission is `READ` (configurable)
3. All users can discover and view the secret
4. Users with `EDIT` can bind it to their jobs
5. Only admin (or users with `MANAGE`) can update the API key

---

## Implementation Files

**Auth Configuration:**
- `mlflow/server/auth/__init__.py`: Main auth logic, validators, filters
- `mlflow/server/auth/permissions.py`: Permission level definitions
- `mlflow/server/auth/config.py`: Auth configuration structure
- `mlflow/server/auth/basic_auth.ini`: Default auth config file
- `mlflow/server/auth/routes.py`: Permission management API routes

**Tests:**
- `tests/server/auth/test_auth.py`: Comprehensive auth test coverage
  - `test_secret_permissions_read` (line 786)
  - `test_secret_permissions_edit` (line 848)
  - `test_secret_permissions_manage` (line 910)
  - `test_secret_permission_boundaries` (line 972)
  - `test_secret_permission_management` (line 1034)
  - `test_admin_bypass_secrets` (line 1118)
  - `test_create_secret_auto_grant` (line 1203)
  - `test_create_route_rbac` (line 1118)
  - `test_list_secrets` (line 1212)

---

## Security Considerations

1. **Secrets are never returned in API responses** - only metadata is exposed
2. **Permission checks happen before database operations** - fail fast
3. **Admin bypass is explicit and logged** - audit trail for compliance
4. **Default permission allows discoverability** - balance security with usability
5. **Auto-grant ensures creators can manage their secrets** - prevents lockout
6. **Bindings are not user-scoped** - secrets can be used by multiple teams
7. **Filtering happens post-query** - performance vs security tradeoff for list operations

---

## Future Considerations

- **Group-based permissions**: Grant permissions to teams instead of individuals
- **Time-limited access**: Temporary permissions that expire
- **Audit logging**: Track all permission changes and secret access
- **Secret rotation policies**: Enforce regular secret updates
- **Integration with external secret managers**: AWS Secrets Manager, HashiCorp Vault, etc.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-12
**Author**: Generated from code analysis for `stack/secrets/auth` branch
