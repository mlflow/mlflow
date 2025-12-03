# Endpoint Entity Refactor Plan

This document outlines changes needed across the PR stack to address PR feedback.

## COMPLETED on `stack/endpoints/entities`

The following changes have been applied and will cause merge conflicts on stack sync:

### 1. File Renames

- `mlflow/entities/endpoint.py` → `mlflow/entities/gateway_endpoint.py`
- `mlflow/entities/secrets.py` → `mlflow/entities/gateway_secrets.py`
- `tests/entities/test_endpoint.py` → `tests/entities/test_gateway_endpoint.py`
- `tests/entities/test_secrets.py` → `tests/entities/test_gateway_secrets.py`

### 2. Class Renames (all prefixed with `Gateway`)

| Old Name               | New Name                      |
| ---------------------- | ----------------------------- |
| `ModelDefinition`      | `GatewayModelDefinition`      |
| `EndpointModelMapping` | `GatewayEndpointModelMapping` |
| `Endpoint`             | `GatewayEndpoint`             |
| `EndpointBinding`      | `GatewayEndpointBinding`      |
| `ModelConfig`          | `GatewayModelConfig`          |
| `EndpointConfig`       | `GatewayEndpointConfig`       |
| `ResourceType`         | `GatewayResourceType`         |
| `Secret`               | `GatewaySecret`               |

### 3. Removed Fields

- `endpoint_count` from `GatewayModelDefinition`
- `endpoint_name` from `GatewayEndpointBinding`
- `model_mappings` from `GatewayEndpointBinding`

### 4. Updated Files

- `mlflow/entities/__init__.py` - new imports and exports
- `docs/api_reference/api_inventory.txt` - updated class names

---

## MERGE CONFLICT RESOLUTION GUIDE

When running `git stack sync`, resolve conflicts at each branch as follows:

### Branch: `stack/endpoints/abstract`

**Conflict Files:**

- `mlflow/store/_abstract_store.py` (if it imports entities)

**Resolution:**

- Update any imports to use new `Gateway` prefixed names
- Update type hints: `Endpoint` → `GatewayEndpoint`, etc.

**Add abstract method for resource config resolution:**

```python
def get_resource_endpoint_config(
    self, resource_type: str, resource_id: str
) -> GatewayEndpointConfig:
    """Get complete endpoint configuration for a resource (server-side only)."""
    raise NotImplementedError()
```

---

### Branch: `stack/endpoints/sql-store`

**Conflict Files:**

- `mlflow/store/tracking/dbmodels/models.py`
- `mlflow/store/tracking/sqlalchemy_store.py`
- `tests/store/tracking/test_sqlalchemy_store.py`

**Resolution in `models.py`:**

```python
# Update imports
from mlflow.entities import (
    GatewayEndpoint,
    GatewayEndpointBinding,
    GatewayEndpointModelMapping,
    GatewayModelDefinition,
    GatewayResourceType,
    GatewaySecret,
)

# Update to_mlflow_entity methods to return Gateway-prefixed classes
# REMOVE endpoint_count from SqlModelDefinition.to_mlflow_entity()
# REMOVE endpoint_name and model_mappings from SqlEndpointBinding.to_mlflow_entity()
```

**Resolution in `sqlalchemy_store.py`:**

- Update all entity imports to use `Gateway` prefix
- Update all type hints and return types

**Resolution in tests:**

- Update imports
- Remove assertions on `endpoint_count`, `endpoint_name`, `model_mappings`

---

### Branch: `stack/endpoints/rest`

**Conflict Files:**

- `mlflow/protos/service.proto`
- `mlflow/entities/gateway_endpoint.py` (proto methods)

**Resolution in `service.proto`:**

```protobuf
// Rename messages to Gateway prefix
message GatewayModelDefinition { ... }
message GatewayEndpointModelMapping { ... }
message GatewayEndpoint { ... }
message GatewayEndpointBinding { ... }
message GatewaySecret { ... }

// REMOVE from GatewayModelDefinition:
// optional int32 endpoint_count = X;

// REMOVE from GatewayEndpointBinding:
// optional string endpoint_name = X;
// repeated GatewayEndpointModelMapping model_mappings = X;
```

**After resolving:** Run `bash dev/generate-protos.sh`

**Add proto round-trip tests** in `tests/entities/test_gateway_endpoint.py`:

```python
def test_model_definition_proto_round_trip():
    model_def = GatewayModelDefinition(...)
    proto = model_def.to_proto()
    model_def2 = GatewayModelDefinition.from_proto(proto)
    assert model_def2.model_definition_id == model_def.model_definition_id
    # ... verify all fields

def test_endpoint_proto_round_trip():
    endpoint = GatewayEndpoint(...)
    proto = endpoint.to_proto()
    endpoint2 = GatewayEndpoint.from_proto(proto)
    assert len(endpoint2.model_mappings) == len(endpoint.model_mappings)

def test_secret_proto_round_trip():
    secret = GatewaySecret(...)
    proto = secret.to_proto()
    secret2 = GatewaySecret.from_proto(proto)
    assert secret2.secret_id == secret.secret_id
```

---

### Branch: `stack/endpoints/rest-2`

**Conflict Files:**

- `mlflow/server/handlers.py`

**Resolution:**

- Update imports to use `Gateway` prefix
- Update handler type hints and return types

---

### Branch: `stack/endpoints/cache`

**Conflict Files:**

- `mlflow/store/tracking/sqlalchemy_store.py` (cache integration)

**Resolution:**

- Update entity imports to `Gateway` prefix
- Update `get_resource_endpoint_config()` return type to `GatewayEndpointConfig`

---

### Branch: `stack/endpoints/litellm`

**Likely no conflicts.** LiteLLM uses `GatewayModelConfig`/`GatewayEndpointConfig` internally.

- Verify imports if any entity types are used directly

---

### Branch: `stack/endpoints/ui-apis`

**Conflict Files:**

- `mlflow/server/js/src/gateway/types.ts`
- `mlflow/server/js/src/gateway/api.ts`

**Resolution in `types.ts`:**

```typescript
// Rename interfaces to Gateway prefix (or keep as-is for UI simplicity)
// If keeping JS names without Gateway prefix, document the mapping

// REMOVE from ModelDefinition interface:
// endpoint_count?: number;

// REMOVE from EndpointBinding interface:
// endpoint_name?: string;
// model_mappings?: EndpointModelMapping[];
```

**Update `api_inventory.txt`** if not already synced from entities branch.

---

### Branch: `stack/endpoints/ui-create-endpoint`

**Review:** Ensure no references to removed fields in endpoint creation UI.

---

### Branch: `stack/endpoints/ui-tabs`

**Review:** Ensure no references to removed fields in tab display.

---

### Branch: `stack/endpoints/key-management`

**Conflict Files:**

- `mlflow/server/js/src/gateway/components/api-keys/BindingsUsingKeyDrawer.tsx`

**Resolution:**

- `BindingsUsingKeyDrawer` uses `binding.endpoint_name` - now removed!
- Implement client-side join since `ApiKeysPage` already fetches `allEndpoints`:

```typescript
// In ApiKeysPage.tsx or BindingsUsingKeyDrawer.tsx
const getEndpointName = (endpointId: string) =>
  allEndpoints?.find((e) => e.endpoint_id === endpointId)?.name;
```

---

### Branch: `stack/endpoints/models`

**No changes needed!** UI already computes endpoint count client-side via `useMemo`.

---

### Branch: `stack/endpoints/passphrase`

**Likely no conflicts.** Review for any entity imports.

---

## NEW FEATURE: Resource Endpoint Config Resolution API

### Background

Server-side resources (e.g., scorer jobs) need to fetch their LLM configuration. The implementation already exists in `sqlalchemy_store.py` but needs REST API exposure.

### Existing Implementation

**Entities** (in `mlflow/entities/gateway_endpoint.py`):

```python
@dataclass
class GatewayModelConfig(_MlflowObject):
    model_definition_id: str
    provider: str
    model_name: str
    secret_value: str          # Decrypted API key
    credential_name: str | None = None
    auth_config: dict | None = None

@dataclass
class GatewayEndpointConfig(_MlflowObject):
    endpoint_id: str
    endpoint_name: str
    models: list[GatewayModelConfig] = field(default_factory=list)
```

**SQLAlchemy** (exists at `sqlalchemy_store.py`):

```python
def get_resource_endpoint_config(
    self,
    resource_type: str,
    resource_id: str,
) -> GatewayEndpointConfig:
    """Get complete endpoint configuration for a resource (server-side only)."""
    # Implementation with:
    # - Binding lookup
    # - Endpoint resolution
    # - Secret decryption with caching
```

### What Needs to be Added

| Branch                     | Implementation                                            |
| -------------------------- | --------------------------------------------------------- |
| `stack/endpoints/abstract` | Add abstract method signature                             |
| `stack/endpoints/rest`     | Add proto messages (NOT exposing secret_value over wire!) |
| `stack/endpoints/rest-2`   | Add handler (internal use only)                           |

### Important Security Note

This API is for **internal server-side use only**:

- Scorer jobs running on tracking server call this internally
- NOT exposed via REST to external clients
- Decrypted secrets never leave the server process

---

## Testing Checklist

At each branch after resolving conflicts:

```bash
# Python entity tests
uv run pytest tests/entities/test_gateway_endpoint.py tests/entities/test_gateway_secrets.py -v

# SQL store tests (at sql-store branch)
uv run pytest tests/store/tracking/test_sqlalchemy_store.py -k endpoint -v
uv run pytest tests/store/tracking/test_sqlalchemy_store.py -k secret -v

# Proto generation (at rest branch)
bash dev/generate-protos.sh

# UI type check (at ui-* branches)
cd mlflow/server/js && yarn type-check && yarn test
```

---

## Quick Reference: Import Updates

When resolving merge conflicts, use these imports:

```python
# Python
from mlflow.entities import (
    GatewayEndpoint,
    GatewayEndpointBinding,
    GatewayEndpointConfig,
    GatewayEndpointModelMapping,
    GatewayModelConfig,
    GatewayModelDefinition,
    GatewayResourceType,
    GatewaySecret,
)

# Or from specific modules
from mlflow.entities.gateway_endpoint import GatewayEndpoint, ...
from mlflow.entities.gateway_secrets import GatewaySecret
```

```typescript
// TypeScript - update types.ts interfaces
// Then import from '@mlflow/gateway/types'
```
