# Secrets Backend Enhancement Guide

## Overview

This document outlines backend changes to enhance the Secrets system in MLflow by adding optional `provider` and `model` fields. These fields will enable storing LLM provider metadata alongside secret credentials for future integration features.

**Current State:** Secrets UI is functional with CRUD operations, bindings, and scope management (private vs global).

**This Phase:** Add optional backend fields without changing the UI.

## Current Secrets UI (Active)

The current implementation provides a complete Secrets management UI:

- ✅ Complete React/TypeScript UI in `mlflow/server/js/src/secrets/`
- ✅ Create Secret Modal with scope selector (private vs global)
- ✅ Secrets table with sorting, filtering, and row actions
- ✅ Update Secret Modal with bindings warning
- ✅ Delete Secret Modal with name confirmation requirement
- ✅ Secret Detail Drawer with metadata and bindings list
- ✅ Unbind functionality for shared secrets
- ✅ Success notifications with i18n support
- ✅ Full integration with backend REST API v3.0

**UI Status:** Active and working. Minor bugs to be fixed (see below).

## Backend Changes Required

### 1. Database Schema Updates

**File:** `mlflow/store/tracking/dbmodels/models.py` (or equivalent secrets model file)

Add new optional columns to existing `secrets` table:

```python
class SqlSecret(Base):
    __tablename__ = "secrets"

    # Existing fields
    secret_id = Column(String(36), primary_key=True)
    secret_name = Column(String(256), nullable=False)
    secret_value_encrypted = Column(LargeBinary, nullable=False)
    field_name = Column(String(256), nullable=False)  # e.g., "ANTHROPIC_API_KEY"
    is_shared = Column(Boolean, nullable=False, default=False)
    created_by = Column(String(256))
    created_at = Column(BigInteger, nullable=False)
    last_updated_at = Column(BigInteger, nullable=False)

    # NEW OPTIONAL FIELDS
    provider = Column(
        String(64), nullable=True
    )  # e.g., "anthropic", "openai", "cohere"
    model = Column(String(256), nullable=True)  # e.g., "claude-3-5-sonnet-20241022"
```

**Migration Script:** `mlflow/store/db_migrations/versions/XXXX_add_provider_model_fields.py`

```python
"""Add provider and model fields to secrets table

Revision ID: xxxxxxxxxxxx
Revises: 1b49d398cd23
Create Date: 2025-XX-XX XX:XX:XX.XXXXXX
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "xxxxxxxxxxxx"
down_revision = "1b49d398cd23"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("secrets") as batch_op:
        batch_op.add_column(sa.Column("provider", sa.String(64), nullable=True))
        batch_op.add_column(sa.Column("model", sa.String(256), nullable=True))


def downgrade():
    with op.batch_alter_table("secrets") as batch_op:
        batch_op.drop_column("model")
        batch_op.drop_column("provider")
```

### 2. Entity Updates

**File:** `mlflow/entities/secret.py`

Add optional provider and model fields:

```python
class Secret:
    def __init__(
        self,
        secret_id,
        secret_name,
        secret_value_encrypted,
        field_name,
        is_shared,
        created_by,
        created_at,
        last_updated_at,
        provider=None,  # NEW
        model=None,  # NEW
    ):
        self._secret_id = secret_id
        self._secret_name = secret_name
        self._secret_value_encrypted = secret_value_encrypted
        self._field_name = field_name
        self._is_shared = is_shared
        self._created_by = created_by
        self._created_at = created_at
        self._last_updated_at = last_updated_at
        self._provider = provider  # NEW
        self._model = model  # NEW

    @property
    def provider(self):
        """Optional LLM provider identifier (e.g., 'anthropic', 'openai')."""
        return self._provider

    @property
    def model(self):
        """Optional LLM model identifier (e.g., 'claude-3-5-sonnet-20241022')."""
        return self._model

    def to_proto(self):
        proto = ProtoSecret()
        proto.secret_id = self.secret_id
        proto.secret_name = self.secret_name
        proto.field_name = self.field_name
        proto.is_shared = self.is_shared
        proto.created_by = self.created_by
        proto.created_at = self.created_at
        proto.last_updated_at = self.last_updated_at
        if self.provider:  # NEW
            proto.provider = self.provider
        if self.model:  # NEW
            proto.model = self.model
        return proto
```

### 3. Protobuf Updates

**File:** `mlflow/protos/service.proto`

Add optional fields to Secret message and request messages:

```protobuf
message Secret {
  optional string secret_id = 1;
  optional string secret_name = 2;
  optional string field_name = 3;
  optional bool is_shared = 4;
  optional string created_by = 5;
  optional int64 created_at = 6;
  optional int64 last_updated_at = 7;
  optional string provider = 8;   // NEW: LLM provider (e.g., "anthropic", "openai")
  optional string model = 9;      // NEW: LLM model (e.g., "claude-3-5-sonnet-20241022")
}

message CreateAndBindSecretRequest {
  optional string secret_name = 1;
  optional string secret_value = 2;
  optional string field_name = 3;
  optional string resource_type = 4;
  optional string resource_id = 5;
  optional string provider = 6;    // NEW: Optional LLM provider
  optional string model = 7;       // NEW: Optional LLM model
}

message UpdateSecretRequest {
  optional string secret_id = 1;
  optional string secret_value = 2;
  optional string provider = 3;    // NEW: Optional provider update
  optional string model = 4;       // NEW: Optional model update
}
```

**After editing:** Run `./dev/generate-protos.sh` to regenerate Python protobuf files.

### 4. Store Layer Updates

**File:** `mlflow/store/tracking/file_store.py` and `mlflow/store/tracking/sqlalchemy_store.py`

Update `create_secret()` method to accept optional parameters:

```python
def create_secret(
    self,
    secret_name,
    secret_value,
    field_name,
    is_shared=False,
    provider=None,  # NEW: Optional
    model=None,  # NEW: Optional
):
    """
    Create a new secret.

    Args:
        secret_name: User-friendly name for the secret
        secret_value: The actual secret value (will be encrypted)
        field_name: Environment variable name (e.g., "ANTHROPIC_API_KEY")
        is_shared: Whether this is a global/shared secret
        provider: Optional LLM provider identifier (e.g., "anthropic", "openai")
        model: Optional LLM model identifier (e.g., "claude-3-5-sonnet-20241022")

    Returns:
        Secret: The created secret entity
    """
    secret_id = uuid.uuid4().hex
    encrypted_value = self._encrypt_secret(secret_value)

    secret = Secret(
        secret_id=secret_id,
        secret_name=secret_name,
        secret_value_encrypted=encrypted_value,
        field_name=field_name,
        is_shared=is_shared,
        created_by=self._get_current_user(),
        created_at=get_current_time_millis(),
        last_updated_at=get_current_time_millis(),
        provider=provider,  # NEW
        model=model,  # NEW
    )

    self._save_secret(secret)
    return secret
```

Update `update_secret()` method to accept optional parameters:

```python
def update_secret(
    self,
    secret_id,
    secret_value,
    provider=None,  # NEW: Optional
    model=None,  # NEW: Optional
):
    """
    Update an existing secret's value and optionally its provider/model metadata.

    Args:
        secret_id: ID of the secret to update
        secret_value: New secret value (will be encrypted)
        provider: Optional new provider identifier
        model: Optional new model identifier
    """
    # Implementation details...
```

### 5. REST API Updates

**File:** `mlflow/server/handlers.py`

Update existing endpoints to accept and store new optional fields:

```python
@catch_mlflow_exception
def create_and_bind_secret():
    request_message = _get_request_message(CreateAndBindSecretRequest())

    # Extract optional fields if present
    provider = (
        request_message.provider if request_message.HasField("provider") else None
    )
    model = request_message.model if request_message.HasField("model") else None

    secret = get_tracking_store().create_secret(
        secret_name=request_message.secret_name,
        secret_value=request_message.secret_value,
        field_name=request_message.field_name,
        is_shared=(request_message.resource_type == "GLOBAL"),
        provider=provider,
        model=model,
    )

    # ... rest of binding logic
    return Response(
        response=message_to_json(CreateAndBindSecretResponse(secret=secret.to_proto())),
        status=200,
    )


@catch_mlflow_exception
def update_secret():
    request_message = _get_request_message(UpdateSecretRequest())

    # Extract optional fields if present
    provider = (
        request_message.provider if request_message.HasField("provider") else None
    )
    model = request_message.model if request_message.HasField("model") else None

    secret = get_tracking_store().update_secret(
        secret_id=request_message.secret_id,
        secret_value=request_message.secret_value,
        provider=provider,
        model=model,
    )

    return Response(
        response=message_to_json(secret.to_proto()),
        status=200,
    )
```

### 6. Testing Updates

**Files:** `tests/store/tracking/test_file_store.py`, `tests/store/tracking/test_sqlalchemy_store.py`

Add tests for new optional fields:

```python
def test_create_secret_with_provider_and_model(store):
    """Test creating a secret with provider and model metadata."""
    secret = store.create_secret(
        secret_name="claude-prod",
        secret_value="sk-ant-12345",
        field_name="ANTHROPIC_API_KEY",
        is_shared=True,
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
    )

    assert secret.provider == "anthropic"
    assert secret.model == "claude-3-5-sonnet-20241022"

    # Verify persisted
    retrieved = store.get_secret(secret.secret_id)
    assert retrieved.provider == "anthropic"
    assert retrieved.model == "claude-3-5-sonnet-20241022"


def test_create_secret_without_provider_model(store):
    """Test creating a secret without provider/model (backward compatibility)."""
    secret = store.create_secret(
        secret_name="my-key",
        secret_value="abc123",
        field_name="API_KEY",
    )

    assert secret.provider is None
    assert secret.model is None

    # Should still work normally
    retrieved = store.get_secret(secret.secret_id)
    assert retrieved.secret_name == "my-key"


def test_update_secret_with_provider_model(store):
    """Test updating a secret's provider and model."""
    secret = store.create_secret(
        secret_name="test-key",
        secret_value="value1",
        field_name="TEST_KEY",
    )

    # Update with provider/model
    updated = store.update_secret(
        secret_id=secret.secret_id,
        secret_value="value2",
        provider="openai",
        model="gpt-4-turbo",
    )

    assert updated.provider == "openai"
    assert updated.model == "gpt-4-turbo"
```

**File:** `tests/server/test_handlers.py`

Add REST API tests:

```python
def test_create_and_bind_secret_with_provider_model():
    """Test creating a secret via REST API with provider/model."""
    payload = {
        "secret_name": "claude-key",
        "secret_value": "sk-ant-12345",
        "field_name": "ANTHROPIC_API_KEY",
        "resource_type": "GLOBAL",
        "resource_id": "global",
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
    }

    response = client.post("/api/3.0/mlflow/secrets/create-and-bind", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["secret"]["provider"] == "anthropic"
    assert data["secret"]["model"] == "claude-3-5-sonnet-20241022"
```

## Migration Path & Backward Compatibility

### 1. Existing Secrets

All existing secrets without `provider`/`model` continue to work normally:

- Fields are nullable in database
- `None` values are valid
- UI can choose to display these differently or not at all

### 2. API Compatibility

Creating secrets without `provider`/`model` still works:

```python
# Old API calls still work
store.create_secret(
    secret_name="my-key",
    secret_value="abc123",
    field_name="API_KEY",
)
```

### 3. Database Migration

- Add columns as nullable (no default values needed)
- No data migration required
- Existing rows have `NULL` for new fields

### 4. Frontend Compatibility

Current UI does not send these fields, which is fine:

- Backend accepts them as optional
- Frontend can be updated later to send/display them
- No breaking changes

## Summary of Changes

| Layer         | Files Changed                                                  | Changes                                  |
| ------------- | -------------------------------------------------------------- | ---------------------------------------- |
| **Database**  | `dbmodels/models.py`                                           | Add nullable `provider`, `model` columns |
| **Migration** | `db_migrations/versions/XXX.py`                                | Add column migration script              |
| **Entity**    | `entities/secret.py`                                           | Add optional provider/model properties   |
| **Proto**     | `protos/service.proto`                                         | Add optional provider/model fields       |
| **Store**     | `file_store.py`, `sqlalchemy_store.py`                         | Accept optional provider/model params    |
| **API**       | `server/handlers.py`                                           | Extract and store optional fields        |
| **Tests**     | test_file_store.py, test_sqlalchemy_store.py, test_handlers.py | Add coverage for new fields              |

## Known UI Bugs to Fix

The current Secrets UI has some minor issues to address:

1. **Bug 1:** [To be identified]
2. **Bug 2:** [To be identified]
3. **Bug 3:** [To be identified]

_(These will be cataloged and fixed after this backend enhancement is complete)_

---

## APPENDIX: Alternative Gateway UI Approach (Future Option)

### Overview

An alternative approach was considered that would rebrand the UI as "Gateway" and provide a more opinionated provider/model selection experience. This approach is documented here for future reference but is **NOT being implemented at this time**.

### Key Differences from Current Approach

| Aspect             | Current Approach (Active)  | Gateway Approach (Option)           |
| ------------------ | -------------------------- | ----------------------------------- |
| **UI Name**        | "Secrets"                  | "Gateway"                           |
| **Provider/Model** | Optional backend fields    | Required during creation            |
| **Create Flow**    | Name + Value + Scope       | Provider → Model → Key Name → Value |
| **API Key Name**   | User enters manually       | Pre-populated, editable             |
| **Provider List**  | Not needed                 | New endpoint required               |
| **Table Columns**  | Name, Value, Type, Created | Name, Provider, Model, Key, Type    |

### Gateway Approach: Additional Requirements

If this approach is chosen in the future, the following would be needed:

#### 1. New API Endpoint: List Providers

**File:** `mlflow/server/handlers.py`

```python
@catch_mlflow_exception
def list_gateway_providers():
    """Return list of supported LLM providers and models."""
    providers = get_gateway_provider_list()
    response = ListGatewayProvidersResponse()
    response.providers.extend([p.to_proto() for p in providers])
    return response


# Register route
app.add_url_rule(
    rule="/ajax-api/3.0/mlflow/secrets/gateway-providers",
    view_func=list_gateway_providers,
    methods=["GET"],
)
```

#### 2. Provider Definitions

**File:** `mlflow/gateway/providers.py` (NEW FILE)

```python
"""Gateway provider and model definitions."""

from dataclasses import dataclass
from typing import List


@dataclass
class GatewayModel:
    model_id: str
    display_name: str


@dataclass
class GatewayProvider:
    provider_id: str
    display_name: str
    default_key_name: str
    models: List[GatewayModel]


GATEWAY_PROVIDERS = [
    GatewayProvider(
        provider_id="anthropic",
        display_name="Anthropic",
        default_key_name="ANTHROPIC_API_KEY",
        models=[
            GatewayModel("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet (New)"),
            GatewayModel("claude-3-opus-20240229", "Claude 3 Opus"),
        ],
    ),
    GatewayProvider(
        provider_id="openai",
        display_name="OpenAI",
        default_key_name="OPENAI_API_KEY",
        models=[
            GatewayModel("gpt-4-turbo", "GPT-4 Turbo"),
            GatewayModel("gpt-4o", "GPT-4o"),
        ],
    ),
    # ... more providers
]
```

#### 3. UI Changes Required

- Multi-step modal for secret creation
- Provider/model dropdowns
- Pre-populated key name field
- Table updates to show provider/model columns
- "Gateway" terminology throughout UI

### Why This Approach Is Not Currently Implemented

1. **Current UI works:** The existing Secrets UI is functional and meeting needs
2. **Simpler migration:** Adding optional fields is non-breaking
3. **Flexibility:** Provider/model can be added manually or via future automation
4. **Less scope:** Avoid UI overhaul while fixing current bugs
5. **Future option:** Can be implemented later if product direction changes

This approach remains documented as a potential future enhancement if the product strategy shifts toward a more opinionated LLM gateway experience.
