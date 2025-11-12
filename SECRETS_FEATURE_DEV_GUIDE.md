# Secrets Management Feature - Development Guide

This guide outlines the stacked PR approach for implementing secrets management in MLflow.

---

## PR 1: Database Schema & Models ✅ COMPLETE

**Branch**: `stack/secrets/db-migration-tables`

**Scope**: Database foundation for secrets management

### Changes Included:

1. **SQLAlchemy Models** (`mlflow/store/tracking/dbmodels/models.py`)
   - `SqlSecret` - Stores encrypted credentials with envelope encryption metadata
   - `SqlSecretBinding` - Maps secrets to resources with CASCADE delete

2. **Alembic Migration** (`mlflow/store/db_migrations/versions/1b49d398cd23_add_secrets_tables.py`)
   - Creates `secrets` table
   - Creates `secrets_bindings` table
   - All constraints have explicit names for MSSQL compatibility

3. **Schema Updates**
   - `tests/resources/db/latest_schema.sql`
   - `tests/db/schemas/sqlite.sql`

**Note**: Tests will be added in PR 2 through the store interface (following MLflow's testing pattern)

### Key Design Decisions:

- **Envelope Encryption**: KEK wraps DEK, DEK encrypts secret value
- **AES-GCM-256**: NIST-approved AEAD algorithm
- **Two-table design**: Secrets can be shared across multiple resources
- **CASCADE delete**: Bindings auto-delete when secret is deleted
- **Named constraints**: All FKs and indexes have explicit names for MSSQL

---

## PR 2: Entity Layer & Store Interface (NEXT)

**Branch**: TBD (create from PR 1)

**Scope**: MLflow entity classes and store CRUD operations

### Files to Create:

1. **Entity Classes**
   - `mlflow/entities/secret.py` - Secret metadata entity (no crypto fields!)
   - `mlflow/entities/secret_binding.py` - SecretBinding entity
   - Update `mlflow/entities/__init__.py` - Export new entities

2. **Store Interface**
   - `mlflow/store/tracking/secrets_store.py` - Secrets CRUD operations
   - Or integrate directly into `SqlAlchemyStore`

3. **Tests** (add to `tests/store/tracking/test_sqlalchemy_store.py`)
   - `test_secret_operations` - Create, get, list, update, delete secrets
   - `test_secret_binding_operations` - Bind, unbind, list bindings
   - `test_secret_cascade_delete` - Verify CASCADE delete of bindings
   - `test_secret_constraints` - Unique names, foreign key enforcement
   - Plus tests in:
     - `tests/entities/test_secret.py`
     - `tests/entities/test_secret_binding.py`

### Key Changes to Existing Files:

**`mlflow/store/tracking/dbmodels/models.py`:**
- Add `SqlSecret.to_mlflow_entity()` method
- Add `SqlSecretBinding.to_mlflow_entity()` method

**Reference patterns:**
- `SqlRun.to_mlflow_entity()` → `Run`
- `SqlExperiment.to_mlflow_entity()` → `Experiment`
- `SqlScorer.to_mlflow_entity()` → `Scorer`

### Critical Security Rule:

**Entity layer MUST NOT expose crypto fields:**
- ❌ NO `ciphertext`
- ❌ NO `iv` (initialization vector)
- ❌ NO `wrapped_dek`
- ❌ NO `kek_version`
- ❌ NO `aad_hash`

Only metadata fields:
- ✅ `secret_id`, `secret_name`
- ✅ `is_shared`, `state`
- ✅ Timestamps and audit fields

Crypto fields stay internal to store layer only!

### Store Methods to Implement:

```python
# Secrets CRUD
create_secret(secret_name, value, is_shared=False, created_by=None) -> Secret
get_secret(secret_id) -> Secret  # metadata only
get_secret_value(secret_id) -> dict  # decrypted (internal use)
list_secrets(is_shared=None) -> List[Secret]
update_secret(secret_id, value, updated_by=None) -> Secret
revoke_secret(secret_id, revoked_by=None) -> Secret
delete_secret(secret_id) -> None

# Bindings CRUD
bind_secret(secret_id, resource_type, resource_id, binding_name, created_by=None) -> SecretBinding
get_binding(binding_id) -> SecretBinding
list_bindings(resource_type=None, resource_id=None, secret_id=None) -> List[SecretBinding]
unbind_secret(binding_id) -> None

# Resource helpers
get_secret_for_resource(resource_type, resource_id, binding_name) -> dict  # decrypted
get_all_secrets_for_resource(resource_type, resource_id) -> Dict[str, dict]
```

---

## PR 3: Cryptography Implementation ✅ COMPLETE

**Branch**: `crypto` (stacked on PR 1)

**Scope**: KEK/DEK encryption/decryption, key management

### Files Created:

1. **Crypto Module**
   - `mlflow/utils/crypto.py` - Encryption/decryption operations with KEK management
   - Dataclasses: `AESGCMResult`, `EncryptedSecret`, `RotatedSecret`
   - Functions: `encrypt_secret()`, `decrypt_secret()`, `rotate_secret_encryption()`

2. **KEK Rotation CLI**
   - `mlflow/cli/secrets.py` - CLI command for KEK rotation
   - Registered in `mlflow/cli/__init__.py`

3. **Tests**
   - `tests/utils/test_crypto.py` - 43 comprehensive tests covering all crypto operations

### Testing Status:

**✅ Completed (43 tests passing):**
- KEK derivation from passphrase (PBKDF2-HMAC-SHA256)
- DEK generation and wrapping
- AES-256-GCM encryption/decryption
- AAD (Additional Authenticated Data) validation
- Secret masking for display
- Envelope encryption roundtrips
- Error handling for invalid keys, AAD mismatch, tampering

**⚠️ Deferred to PR4/PR5 (requires store layer):**
- `tests/cli/test_secrets_cli.py` - Integration tests for `mlflow secrets rotate-kek`
  - Reason: CLI depends on `SqlSecret` model, database queries, and actual secrets
  - Plan: Add integration tests once store layer is implemented
  - Current coverage: Only basic crypto operations tested, not full CLI workflow
  - TODO: Test database transaction handling, rollback on failure, kek_version updates

### Crypto Operations:

**Encryption Flow:**
```python
1. Load active KEK from /etc/mlflow/keystore/
2. Generate random 256-bit DEK
3. Encrypt secret value with AES-GCM (DEK, random IV)
4. Wrap DEK with KEK (encrypt DEK using KEK)
5. Compute AAD hash (SHA-256 of secret_id || secret_name)
6. Store: ciphertext, iv, wrapped_dek, kek_version, aad_hash
```

**Decryption Flow:**
```python
1. Load KEK version X from keystore
2. Unwrap DEK (decrypt wrapped_dek using KEK)
3. Verify AAD hash matches (secret_id || secret_name)
4. Decrypt ciphertext with AES-GCM (DEK, IV, AAD)
5. Return plaintext secret value
```

### CLI Commands:

```bash
# KEK Management
mlflow secrets kek generate [--passphrase-file PATH] [--passphrase-stdin]
mlflow secrets kek rotate
mlflow secrets kek list
mlflow secrets kek set-active --version N

# Secret rewrap after KEK rotation
mlflow secrets rewrap --kek-version N
```

### Environment Variables:

- `MLFLOW_SECRETS_KEYSTORE_PATH` - Path to KEK directory (default: `/etc/mlflow/keystore`)
- `MLFLOW_SECRETS_KEK_PASSPHRASE_FILE` - Path to passphrase file (default: `/root/.mlflow_kek_pass`)
- `MLFLOW_SECRETS_TEST_MODE` - Use fixed test KEK (testing only)

---

## PR 4: REST API Endpoints

**Branch**: TBD (create from PR 3)

**Scope**: REST API for secrets management

### Files to Create:

1. **Service Layer**
   - `mlflow/server/handlers/secrets_handlers.py` - Request handlers
   - Or extend existing handlers

2. **Protobuf Definitions** (if using protobuf)
   - `mlflow/protos/secrets.proto` - Secret, SecretBinding messages
   - Request/response messages

3. **Tests**
   - `tests/server/test_secrets_handlers.py`
   - `tests/store/tracking/test_rest_store.py` - Add secrets tests
   - **`tests/cli/test_secrets_cli.py` - ADD CLI ROTATION TESTS HERE** ⚠️
     - Test `mlflow secrets rotate-kek` command end-to-end
     - Test transaction handling, rollback on failure
     - Test kek_version updates in database
     - Test confirmation prompts and --yes flag
     - Deferred from PR3 due to missing store layer

### API Endpoints:

```python
# Secrets
POST   /api/2.0/mlflow/secrets/create
GET    /api/2.0/mlflow/secrets/get
GET    /api/2.0/mlflow/secrets/list
PATCH  /api/2.0/mlflow/secrets/update
POST   /api/2.0/mlflow/secrets/revoke
DELETE /api/2.0/mlflow/secrets/delete

# Bindings
POST   /api/2.0/mlflow/secrets/bind
GET    /api/2.0/mlflow/secrets/get-binding
GET    /api/2.0/mlflow/secrets/list-bindings
DELETE /api/2.0/mlflow/secrets/unbind
```

### Security Considerations:

- **Never return plaintext secrets** via GET endpoints
- Use POST for create/update to avoid logging secrets in URLs
- Implement rate limiting for decrypt operations
- Add audit logging for all secret access

---

## PR 5: Integration with Scorers

**Branch**: TBD (create from PR 4)

**Scope**: Use secrets in scorer jobs instead of environment variables

### Files to Modify:

1. **Scorer Configuration**
   - `mlflow/scorers/` - Update to use secrets
   - Modify scorer creation to accept secret references

2. **Job Execution**
   - `mlflow/jobs/` - Fetch secrets before job execution
   - Inject decrypted secrets into job environment

3. **Tests**
   - Update scorer tests to use secrets
   - Add integration tests

### Migration Path:

**Old (env vars):**
```python
scorer = mlflow.scorers.create_llm_judge(
    name="my-judge",
    openai_api_key=os.environ["OPENAI_API_KEY"]  # Direct env var
)
```

**New (secrets):**
```python
# Option 1: Pass secret directly (auto-create private secret)
scorer = mlflow.scorers.create_llm_judge(
    name="my-judge",
    openai_api_key="sk-proj-abc123"  # Auto-encrypted
)

# Option 2: Reference existing shared secret
scorer = mlflow.scorers.create_llm_judge(
    name="my-judge",
    openai_api_key_secret="my-openai-key"  # Reference shared secret
)
```

---

## PR 6: Secrets Management CLI (User-facing)

**Branch**: TBD (create from PR 5)

**Scope**: User-facing CLI for managing secrets

### CLI Commands:

```bash
# Create secrets
mlflow secrets create my-openai-key --value "sk-proj-abc" --shared
mlflow secrets create scorer-key --value "sk-proj-xyz"  # Private (default)

# List secrets
mlflow secrets list
mlflow secrets list --shared

# Get secret metadata (not value!)
mlflow secrets get my-openai-key

# Update secret
mlflow secrets update my-openai-key --value "sk-proj-new"

# Revoke secret
mlflow secrets revoke my-openai-key

# Delete secret
mlflow secrets delete my-openai-key

# Manage bindings
mlflow secrets bind my-openai-key --resource-type SCORER_JOB --resource-id scorer-123 --name api_key
mlflow secrets unbind <binding-id>
mlflow secrets list-bindings --resource-type SCORER_JOB
```

---

## PR 7: UI for Secrets Management (Future)

**Scope**: Web UI for managing secrets

### Features:

- List shared secrets
- Create/update/delete shared secrets
- View which resources use each secret
- Rotate secrets
- Audit log viewer

### Pages:

- `/secrets` - List all shared secrets
- `/secrets/:id` - Secret detail page
- Integration into scorer creation form

---

## Open Design Questions

### 1. Entity Serialization Format

**Question**: Should entities use protobuf or Python dataclasses?

**Options**:
- A: Protobuf (like Run, Experiment) - Required for REST API compatibility
- B: Dataclasses - Simpler, but need conversion layer for REST

**Recommendation**: Use protobuf for consistency with existing MLflow entities

### 2. Store Architecture

**Question**: Should SecretsStore be a separate class or integrated into SqlAlchemyStore?

**Options**:
- A: Separate `SecretsStore` class - Better separation of concerns
- B: Methods on `SqlAlchemyStore` - Follows existing pattern

**Recommendation**: Look at how TraceStore is integrated for consistency

### 3. Secret Value Format

**Question**: How to handle string vs dict values in the API?

**Current Design**:
- String → auto-wrap as `{"value": "..."}`
- Dict → store as-is
- Get → auto-unwrap if only "value" key

**Edge Case**: User provides `{"value": "sk-...", "org": "..."}`
- Should we unwrap or not?

**Recommendation**: Keep auto-wrap/unwrap, document edge cases

### 4. Permissions Model

**Question**: How to control access to secrets?

**Options**:
- A: Private secrets = only resource owner can access
- B: Shared secrets = workspace-scoped (anyone in workspace)
- C: RBAC = fine-grained permissions

**Recommendation**: Start with A+B for MVP, add C in future

---

## Testing Strategy

### Unit Tests
- Entity serialization/deserialization
- Store CRUD operations (with mocked DB)
- Crypto operations (encryption/decryption)
- CLI command parsing

### Integration Tests
- Full encrypt → store → retrieve → decrypt flow
- KEK rotation and secret rewrap
- CASCADE delete behavior
- Secrets in scorer jobs (end-to-end)

### Security Tests
- Verify crypto fields never exposed in entities
- Verify secrets never logged
- Verify AAD validation prevents tampering
- Verify foreign secret_id rejection

### Performance Tests
- Decryption performance (should be < 10ms per secret)
- Batch secret fetching for resources
- KEK caching effectiveness

---

## Documentation Needs

### Developer Docs
- This dev guide
- Architecture overview
- Security model explanation

### User Docs
- Getting started guide
- CLI reference
- REST API reference
- Best practices
- Troubleshooting guide

### Admin Docs
- KEK generation and rotation
- Backup and disaster recovery
- Multi-server deployment
- Container/Kubernetes setup

---

## Security Review Checklist

Before merging the full feature:

- [ ] Crypto fields isolated to store layer
- [ ] No plaintext secrets in logs
- [ ] No secrets in error messages
- [ ] AAD validation prevents tampering
- [ ] KEK passphrase properly protected
- [ ] Audit logging for all access
- [ ] Rate limiting for decrypt operations
- [ ] SQL injection prevention (use parameterized queries)
- [ ] No secrets in URLs or query params
- [ ] HTTPS required for REST API

---

## Timeline Estimate

- PR 1: Database Schema ✅ **COMPLETE**
- PR 2: Entities & Store - 2-3 days
- PR 3: Cryptography - 3-4 days
- PR 4: REST API - 2-3 days
- PR 5: Scorer Integration - 2-3 days
- PR 6: User CLI - 1-2 days
- PR 7: UI - 5-7 days (optional)

**Total MVP (PR 1-6)**: ~2-3 weeks

---

## Reference Documentation

See detailed design documents:
- `SECRETS_MANAGEMENT_DESIGN.md` - Full technical design
- `KEK_PASSPHRASE_MANAGEMENT.md` - KEK passphrase handling
- `SECRETS_DESIGN_USABILITY_REVIEW.md` - Usability analysis
