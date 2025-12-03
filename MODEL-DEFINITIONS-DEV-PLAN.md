# Model Definitions Refactoring - Development Plan

## Overview

Refactor the Gateway backend to support **reusable model definitions** that can be shared across multiple endpoints. This enables a future "Models" tab in the UI and decouples model configuration from endpoint lifecycle.

## Current Architecture

```
Endpoint (1) ──────► (N) EndpointModel
                         ├─ model_id (PK)
                         ├─ endpoint_id (FK, NOT NULL, CASCADE DELETE)
                         ├─ secret_id (FK)
                         ├─ provider
                         └─ model_name
```

**Problems with current design:**
- Models are tightly coupled to endpoints (cannot exist independently)
- Deleting an endpoint cascades to delete all its models
- No way to reuse a model configuration across multiple endpoints
- Models have no user-friendly name for identification

## Proposed Architecture

```
                                    ┌─────────────────────────┐
                                    │    ModelDefinition      │
                                    ├─────────────────────────┤
                                    │ model_definition_id (PK)│
                                    │ name (unique)           │
                                    │ secret_id (FK)          │
                                    │ provider                │
                                    │ model_name              │
                                    │ created_at              │
                                    │ created_by              │
                                    └───────────┬─────────────┘
                                                │
                                                │ (1)
                                                │
                                                ▼
┌─────────────────────┐         ┌───────────────────────────────┐
│      Endpoint       │         │   EndpointModelMapping        │
├─────────────────────┤         ├───────────────────────────────┤
│ endpoint_id (PK)    │◄────────│ endpoint_id (FK, CASCADE)     │
│ name                │   (N)   │ model_definition_id (FK)      │
│ created_at          │         │ mapping_id (PK)               │
│ created_by          │         │ weight (optional, for routing)│
└─────────────────────┘         │ created_at                    │
                                └───────────────────────────────┘
```

**Key changes:**
- Model definitions exist independently of endpoints
- Junction table links endpoints to model definitions (M:N relationship)
- Deleting an endpoint only removes the mapping, not the model definition
- Model definitions have a unique `name` for identification and reuse

---

## PR Stack Mapping

All changes will be made to existing PRs in the stack. Changes propagate down through child branches.

| PR | Branch | Changes Required |
|----|--------|------------------|
| #19002 | `stack/endpoints/db` | Add migration for `model_definitions` table + rename `endpoint_models` → `endpoint_model_mappings` |
| #19003 | `stack/endpoints/crypto` | No changes |
| #19004 | `stack/endpoints/entities` | Add `ModelDefinition` entity, update `EndpointModel` → `EndpointModelMapping` |
| #19005 | `stack/endpoints/abstract` | Add abstract methods for model definition CRUD, update endpoint methods |
| #19006 | `stack/endpoints/sql-store` | Implement model definition store methods, update cascade behavior |
| #19007 | `stack/endpoints/rest` | Add model definition REST handlers |
| #19008 | `stack/endpoints/rest-2` | Update endpoint handlers to reference model definitions |
| #19014 | `stack/endpoints/cache` | Update cache keys if model definitions are cached |
| #19009 | `stack/endpoints/litellm` | Update to use model definitions |
| #19010 | `stack/endpoints/ui-apis` | Add model definition API types and hooks |
| #19042 | `stack/endpoints/ui-create-endpoint` | Update endpoint creation to select existing models |
| #19070 | `stack/endpoints/ui-tabs` | Add "Models" tab to side nav |
| #19071 | `stack/endpoints/key-management` | Update API keys page to show model usage |

---

## Implementation Details

### Phase 1: Database Schema

**New table: `model_definitions`**

```python
class SqlModelDefinition(Base):
    __tablename__ = "model_definitions"

    model_definition_id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    secret_id = Column(String(36), ForeignKey("secrets.secret_id"), nullable=False)
    provider = Column(String(64), nullable=False)
    model_name = Column(String(256), nullable=False)
    created_by = Column(String(255), nullable=True)
    created_at = Column(BigInteger, nullable=False)
    last_updated_by = Column(String(255), nullable=True)
    last_updated_at = Column(BigInteger, nullable=False)

    # Relationships
    secret = relationship("SqlSecret")
    endpoint_mappings = relationship("SqlEndpointModelMapping", back_populates="model_definition")
```

**Modified table: `endpoint_models` → `endpoint_model_mappings`**

```python
class SqlEndpointModelMapping(Base):
    __tablename__ = "endpoint_model_mappings"

    mapping_id = Column(String(36), primary_key=True)
    endpoint_id = Column(
        String(36),
        ForeignKey("endpoints.endpoint_id", ondelete="CASCADE"),
        nullable=False
    )
    model_definition_id = Column(
        String(36),
        ForeignKey("model_definitions.model_definition_id", ondelete="RESTRICT"),
        nullable=False
    )
    weight = Column(Integer, default=1)  # For future traffic routing
    created_by = Column(String(255), nullable=True)
    created_at = Column(BigInteger, nullable=False)

    # Relationships
    endpoint = relationship("SqlEndpoint", back_populates="model_mappings")
    model_definition = relationship("SqlModelDefinition", back_populates="endpoint_mappings")
```

**Key constraint changes:**
- `endpoint_id` FK: `ondelete="CASCADE"` - mapping deleted when endpoint deleted
- `model_definition_id` FK: `ondelete="RESTRICT"` - prevent deletion of models in use

### Phase 2: Entity Definitions

**File: `mlflow/entities/model_definition.py`**

```python
@dataclass
class ModelDefinition(_MlflowObject):
    model_definition_id: str
    name: str
    secret_id: str
    secret_name: str  # Populated via JOIN
    provider: str
    model_name: str
    created_at: int
    last_updated_at: int
    created_by: str | None = None
    last_updated_by: str | None = None
    endpoint_count: int = 0  # Number of endpoints using this model
```

**Updated: `mlflow/entities/endpoint.py`**

```python
@dataclass
class EndpointModelMapping(_MlflowObject):
    mapping_id: str
    endpoint_id: str
    model_definition_id: str
    model_definition: ModelDefinition | None = None  # Populated via JOIN
    weight: int = 1
    created_at: int
    created_by: str | None = None
```

### Phase 3: Abstract Store Interface

**File: `mlflow/store/tracking/abstract_store.py`**

```python
# New methods to add:

@abstractmethod
def create_model_definition(
    self,
    name: str,
    secret_id: str,
    provider: str,
    model_name: str,
    created_by: str | None = None,
) -> ModelDefinition:
    """Create a reusable model definition."""
    pass

@abstractmethod
def get_model_definition(self, model_definition_id: str) -> ModelDefinition:
    """Get a model definition by ID."""
    pass

@abstractmethod
def list_model_definitions(
    self,
    provider: str | None = None,
    secret_id: str | None = None,
) -> list[ModelDefinition]:
    """List all model definitions, optionally filtered."""
    pass

@abstractmethod
def update_model_definition(
    self,
    model_definition_id: str,
    name: str | None = None,
    secret_id: str | None = None,
    model_name: str | None = None,
    updated_by: str | None = None,
) -> ModelDefinition:
    """Update a model definition."""
    pass

@abstractmethod
def delete_model_definition(self, model_definition_id: str) -> None:
    """Delete a model definition. Fails if in use by any endpoint."""
    pass

# Modified methods:

@abstractmethod
def create_endpoint(
    self,
    name: str,
    model_definition_ids: list[str],  # Changed from inline model specs
    created_by: str | None = None,
) -> Endpoint:
    """Create endpoint with references to existing model definitions."""
    pass

@abstractmethod
def attach_model_to_endpoint(
    self,
    endpoint_id: str,
    model_definition_id: str,
    weight: int = 1,
    created_by: str | None = None,
) -> EndpointModelMapping:
    """Attach an existing model definition to an endpoint."""
    pass

@abstractmethod
def detach_model_from_endpoint(
    self,
    endpoint_id: str,
    model_definition_id: str,
) -> None:
    """Remove a model definition from an endpoint (does not delete the model)."""
    pass
```

### Phase 4: REST API

**New endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/2.0/mlflow/model-definitions` | Create model definition |
| GET | `/api/2.0/mlflow/model-definitions` | List model definitions |
| GET | `/api/2.0/mlflow/model-definitions/{id}` | Get model definition |
| PUT | `/api/2.0/mlflow/model-definitions/{id}` | Update model definition |
| DELETE | `/api/2.0/mlflow/model-definitions/{id}` | Delete model definition |

**Modified endpoints:**

| Method | Path | Change |
|--------|------|--------|
| POST | `/api/2.0/mlflow/endpoints` | Accept `model_definition_ids` instead of inline model specs |
| POST | `/api/2.0/mlflow/endpoints/{id}/models` | Attach existing model definition |
| DELETE | `/api/2.0/mlflow/endpoints/{id}/models/{model_id}` | Detach (not delete) model |

### Phase 5: UI Changes

**New components:**
- `ModelDefinitionsList.tsx` - List all model definitions
- `CreateModelDefinitionModal.tsx` - Create new model definition
- `ModelDefinitionDetails.tsx` - View/edit model definition
- `useModelDefinitionsQuery.ts` - React Query hook

**Modified components:**
- `GatewaySideNav.tsx` - Add "Models" tab
- `CreateEndpointModal.tsx` - Select from existing models instead of inline creation
- `EndpointDetailsPage.tsx` - Show attached models with detach option

---

## Migration Strategy

### Data Migration

Existing `endpoint_models` data needs to be migrated:

1. For each unique `(secret_id, provider, model_name)` combination, create a `ModelDefinition`
2. Auto-generate names like `{provider}-{model_name}` (ensure uniqueness)
3. Create `EndpointModelMapping` entries linking endpoints to new definitions
4. Drop old `endpoint_models` table

```python
def upgrade():
    # 1. Create new tables
    op.create_table('model_definitions', ...)
    op.create_table('endpoint_model_mappings', ...)

    # 2. Migrate data
    connection = op.get_bind()

    # Get unique model configurations
    existing_models = connection.execute(
        "SELECT DISTINCT secret_id, provider, model_name FROM endpoint_models"
    ).fetchall()

    # Create model definitions
    for secret_id, provider, model_name in existing_models:
        model_def_id = uuid.uuid4().hex
        name = f"{provider}-{model_name}"  # May need uniqueness handling
        connection.execute(
            "INSERT INTO model_definitions (...) VALUES (...)"
        )

        # Create mappings for all endpoints using this model
        connection.execute("""
            INSERT INTO endpoint_model_mappings (mapping_id, endpoint_id, model_definition_id, ...)
            SELECT uuid(), endpoint_id, :model_def_id, ...
            FROM endpoint_models
            WHERE secret_id = :secret_id AND provider = :provider AND model_name = :model_name
        """, {...})

    # 3. Drop old table
    op.drop_table('endpoint_models')
```

### Backward Compatibility

**Option A: Breaking change**
- Require migration before using new version
- Simpler implementation

**Option B: Soft migration**
- Support both old and new API formats during transition
- Auto-create model definitions when endpoints created with inline specs
- Deprecation warnings in logs

**Recommendation:** Option A for initial implementation (this is a pre-release feature)

---

## Testing Requirements

### Unit Tests
- [ ] Model definition CRUD operations
- [ ] Endpoint creation with model definition references
- [ ] Cascade behavior (endpoint delete doesn't delete model definitions)
- [ ] RESTRICT behavior (can't delete model definition in use)
- [ ] Migration script correctness

### Integration Tests
- [ ] End-to-end model definition lifecycle
- [ ] Model sharing across multiple endpoints
- [ ] LiteLLM integration with new model structure

### UI Tests
- [ ] Models tab rendering
- [ ] Model definition CRUD modals
- [ ] Endpoint creation with model selection

---

## Open Questions

1. **Model definition deletion policy:** Should we allow force-delete that also removes from all endpoints, or always require manual detachment first?

2. **Default model naming:** When migrating existing data, how should we generate unique names? Options:
   - `{provider}-{model_name}` (may collide)
   - `{provider}-{model_name}-{short_uuid}`
   - Let users rename after migration

3. **Traffic routing:** Should the `weight` field be implemented now or deferred?

4. **Audit trail:** Should we track which endpoints a model was attached to/detached from?

---

## Implementation Order

Work through the stack from bottom to top, syncing changes as we go:

1. **`stack/endpoints/db`** - Add migration, sync down
2. **`stack/endpoints/entities`** - Update entities, sync down
3. **`stack/endpoints/abstract`** - Update interface, sync down
4. **`stack/endpoints/sql-store`** - Implement store methods, sync down
5. **`stack/endpoints/rest` + `rest-2`** - Update handlers, sync down
6. **`stack/endpoints/litellm`** - Update integration, sync down
7. **`stack/endpoints/ui-*`** - Update UI components

---

## References

- Current endpoint models: `mlflow/store/tracking/dbmodels/models.py:2196-2280`
- Endpoint entity: `mlflow/entities/endpoint.py`
- SQL store: `mlflow/store/tracking/sqlalchemy_store.py:4890-5205`
- REST handlers: `mlflow/server/handlers.py:3898-4097`
