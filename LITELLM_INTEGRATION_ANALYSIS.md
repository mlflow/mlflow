# LiteLLM Integration Analysis

## Executive Summary

This document analyzes the current MLflow secrets/routes system and provides a concrete plan for integrating LiteLLM to enable dynamic provider/model selection in the UI.

**Key Findings:**

- Current schema is already well-designed for LiteLLM integration
- LiteLLM is already used in MLflow (genai/judges module) but not bundled by default
- No major schema changes needed - current structure supports the requirements
- Main work is: (1) Add provider/model listing APIs, (2) Update UI to use dropdowns instead of text inputs, (3) Add `mlflow[genai]` extra

## Current Architecture

### Database Schema

The current schema (from `mlflow/store/tracking/dbmodels/models.py`):

```
SqlSecret (secrets table)
├── secret_id (PK)
├── secret_name
├── encrypted_value + wrapped_dek (API key encryption)
├── provider (String 64) ← Already has provider field!
├── encrypted_auth_config + wrapped_auth_config_dek (Complex auth like AWS)
├── is_shared (Boolean)
└── metadata (created_by, created_at, etc.)

SqlSecretRoute (secret_routes table)
├── route_id (PK)
├── secret_id (FK → secrets)
├── model_name (String 256) ← Model identifier
├── name (display name)
├── encrypted_model_config + wrapped_model_config_dek (Runtime params)
└── metadata

SqlSecretBinding (secrets_bindings table)
├── binding_id (PK)
├── route_id (FK → secret_routes)
├── resource_type (e.g., SCORER_JOB)
├── resource_id
└── field_name (e.g., OPENAI_API_KEY)
```

**Relationship:**

```
Secrets (API keys) → Routes (model configs) → Bindings (resources like scorers)
```

**Key Observation:** The existing `provider` field in `SqlSecret` already supports what we need. No schema changes required!

### Existing LiteLLM Integration

LiteLLM is already used in MLflow:

1. **`mlflow/litellm/`**: Autologging for tracing
2. **`mlflow/genai/judges/adapters/litellm_adapter.py`**: Judge model invocation
3. **Pattern**: Uses `litellm.completion()` with model URIs like `"openai/gpt-4"`

However, LiteLLM is in `requirements/extra-ml-requirements.txt` (not bundled by default).

### Legacy Gateway Module

**Important:** The `mlflow/gateway/` directory contains a legacy AI Gateway implementation with config-file based routing. **We are NOT using this.** Our new implementation is the secrets/routes system integrated with the tracking store.

## LiteLLM Model Catalog

LiteLLM provides a comprehensive model catalog at `/tmp/litellm/model_prices_and_context_window.json` (according to the improvement plan).

**Structure per model:**

```json
{
  "gpt-4": {
    "litellm_provider": "openai",
    "mode": "chat",
    "max_input_tokens": 8192,
    "max_output_tokens": 4096,
    "supports_vision": true,
    "supports_function_calling": true,
    "input_cost_per_token": 0.00003,
    "output_cost_per_token": 0.00006
  }
}
```

**Endpoint types (`mode` field):**

- `chat`: LLM chat completions
- `embedding`: Text embeddings
- `completion`: Legacy completion API
- `image_generation`: DALL-E, Stable Diffusion
- `audio_transcription`: Whisper
- `audio_speech`: TTS
- `moderation`: Content moderation
- `rerank`: Search reranking

## Implementation Plan

### Phase 1: Add `mlflow[genai]` Extra

**File:** `pyproject.toml`

Add a new optional dependency group:

```toml
[project.optional-dependencies]
# ... existing extras ...

genai = [
  "litellm>=1.0.0,!=1.67.4",  # Exclude broken version from constraints.txt
  # Add other genai-specific dependencies as needed
]
```

**Why:** Users installing `pip install mlflow[genai]` will get LiteLLM automatically.

### Phase 2: Provider/Model Listing APIs

**Goal:** Create backend endpoints that expose LiteLLM's model catalog to the frontend.

#### 2.1 Add Protobuf Definitions

**File:** `mlflow/protos/service.proto`

```protobuf
// List available LLM providers
message ListProviders {
  message Response {
    repeated Provider providers = 1;
  }

  rpc listProviders(ListProviders) returns (ListProviders.Response) {
    option (rpc) = {
      endpoints: [
        {
          method: "GET"
          path: "/mlflow/gateway/providers/list"
          since: {
            major: 3
            minor: 0
          }
        }
      ]
      visibility: PUBLIC
    };
  }
}

message Provider {
  string name = 1;              // e.g., "openai"
  string display_name = 2;      // e.g., "OpenAI"
  repeated string supported_modes = 3;  // ["chat", "embedding"]
  optional string description = 4;
}

// List models for a specific provider
message ListModels {
  optional string provider = 1;  // Required query param
  optional string mode = 2;      // Optional filter by endpoint type

  message Response {
    repeated Model models = 1;
  }

  rpc listModels(ListModels) returns (ListModels.Response) {
    option (rpc) = {
      endpoints: [
        {
          method: "GET"
          path: "/mlflow/gateway/models/list"
          since: {
            major: 3
            minor: 0
          }
        }
      ]
      visibility: PUBLIC
    };
  }
}

message Model {
  string model_name = 1;         // e.g., "gpt-4"
  string provider = 2;           // e.g., "openai"
  string mode = 3;               // e.g., "chat"
  optional int64 max_input_tokens = 4;
  optional int64 max_output_tokens = 5;
  optional bool supports_vision = 6;
  optional bool supports_function_calling = 7;
  optional bool supports_streaming = 8;
}
```

#### 2.2 Add Backend Handlers

**File:** `mlflow/server/handlers.py`

```python
from mlflow.utils.lazy_load import LazyLoader

# Lazy load to avoid import errors when litellm not installed
_litellm_catalog = LazyLoader("_litellm_catalog", globals(), "mlflow.gateway.litellm_catalog")


@catch_mlflow_exception
def _list_providers():
    """
    List available LLM providers from LiteLLM catalog.
    """
    try:
        providers = _litellm_catalog.list_providers()
    except ImportError:
        raise MlflowException(
            "LiteLLM is not installed. Install with: pip install mlflow[genai]",
            error_code=INVALID_PARAMETER_VALUE,
        )

    response_message = ListProviders.Response()
    for provider in providers:
        response_message.providers.add().CopyFrom(provider.to_proto())
    return _wrap_response(response_message)


@catch_mlflow_exception
def _list_models():
    """
    List models for a specific provider.
    """
    request_message = _get_request_message(ListModels())

    if not request_message.provider:
        raise MlflowException(
            "provider parameter is required",
            error_code=INVALID_PARAMETER_VALUE,
        )

    try:
        models = _litellm_catalog.list_models(
            provider=request_message.provider,
            mode=request_message.mode if request_message.HasField("mode") else None,
        )
    except ImportError:
        raise MlflowException(
            "LiteLLM is not installed. Install with: pip install mlflow[genai]",
            error_code=INVALID_PARAMETER_VALUE,
        )

    response_message = ListModels.Response()
    for model in models:
        response_message.models.add().CopyFrom(model.to_proto())
    return _wrap_response(response_message)
```

#### 2.3 Create LiteLLM Catalog Module

**New File:** `mlflow/gateway/litellm_catalog.py`

```python
"""
LiteLLM model catalog wrapper for provider/model listing.
"""
import logging
from typing import Optional
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


@dataclass
class Provider:
    name: str
    display_name: str
    supported_modes: list[str]
    description: Optional[str] = None

    def to_proto(self):
        from mlflow.protos.service_pb2 import Provider as ProviderProto
        return ProviderProto(
            name=self.name,
            display_name=self.display_name,
            supported_modes=self.supported_modes,
            description=self.description,
        )


@dataclass
class Model:
    model_name: str
    provider: str
    mode: str
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_streaming: bool = False

    def to_proto(self):
        from mlflow.protos.service_pb2 import Model as ModelProto
        return ModelProto(
            model_name=self.model_name,
            provider=self.provider,
            mode=self.mode,
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
            supports_vision=self.supports_vision,
            supports_function_calling=self.supports_function_calling,
            supports_streaming=self.supports_streaming,
        )


def list_providers() -> list[Provider]:
    """
    Extract unique providers from LiteLLM's model catalog.

    Returns:
        List of Provider objects with metadata.

    Raises:
        ImportError: If litellm is not installed.
    """
    try:
        import litellm
    except ImportError:
        raise ImportError("LiteLLM is required. Install with: pip install mlflow[genai]")

    # LiteLLM exposes model_cost dict with all models
    from litellm import model_cost

    # Extract unique providers and their capabilities
    provider_data = {}
    for model_name, model_info in model_cost.items():
        provider = model_info.get("litellm_provider")
        mode = model_info.get("mode", "chat")

        if provider not in provider_data:
            provider_data[provider] = {
                "modes": set(),
                "display_name": _format_provider_name(provider),
            }
        provider_data[provider]["modes"].add(mode)

    # Convert to Provider objects
    providers = [
        Provider(
            name=name,
            display_name=data["display_name"],
            supported_modes=sorted(data["modes"]),
            description=_get_provider_description(name),
        )
        for name, data in sorted(provider_data.items())
    ]

    return providers


def list_models(provider: str, mode: Optional[str] = None) -> list[Model]:
    """
    List models for a specific provider, optionally filtered by mode.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        mode: Optional endpoint type filter (e.g., "chat", "embedding")

    Returns:
        List of Model objects.

    Raises:
        ImportError: If litellm is not installed.
    """
    try:
        import litellm
    except ImportError:
        raise ImportError("LiteLLM is required. Install with: pip install mlflow[genai]")

    from litellm import model_cost

    models = []
    for model_name, model_info in model_cost.items():
        if model_info.get("litellm_provider") != provider:
            continue

        model_mode = model_info.get("mode", "chat")
        if mode and model_mode != mode:
            continue

        models.append(Model(
            model_name=model_name,
            provider=provider,
            mode=model_mode,
            max_input_tokens=model_info.get("max_input_tokens"),
            max_output_tokens=model_info.get("max_output_tokens"),
            supports_vision=model_info.get("supports_vision", False),
            supports_function_calling=model_info.get("supports_function_calling", False),
            supports_streaming=True,  # Most models support streaming
        ))

    return sorted(models, key=lambda m: m.model_name)


def _format_provider_name(provider: str) -> str:
    """Convert provider slug to display name."""
    name_map = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "cohere": "Cohere",
        "bedrock": "Amazon Bedrock",
        "vertex_ai": "Google Vertex AI",
        "azure": "Azure OpenAI",
        "databricks": "Databricks",
    }
    return name_map.get(provider, provider.replace("_", " ").title())


def _get_provider_description(provider: str) -> Optional[str]:
    """Get provider description."""
    descriptions = {
        "openai": "OpenAI's GPT models and APIs",
        "anthropic": "Anthropic's Claude models",
        "cohere": "Cohere's language models",
        "bedrock": "Amazon Bedrock managed AI service",
        "vertex_ai": "Google Cloud Vertex AI",
    }
    return descriptions.get(provider)
```

### Phase 3: Frontend Updates

#### 3.1 Create Provider/Model Selector Hooks

**New File:** `mlflow/server/js/src/gateway/hooks/useListProviders.ts`

```typescript
import { useQuery } from "@tanstack/react-query";

interface Provider {
  name: string;
  display_name: string;
  supported_modes: string[];
  description?: string;
}

export function useListProviders() {
  return useQuery<Provider[]>({
    queryKey: ["providers"],
    queryFn: async () => {
      const response = await fetch("/ajax-api/3.0/mlflow/gateway/providers/list");
      if (!response.ok) {
        throw new Error("Failed to fetch providers");
      }
      const data = await response.json();
      return data.providers;
    },
    staleTime: 1000 * 60 * 60, // Cache for 1 hour (providers don't change)
  });
}
```

**New File:** `mlflow/server/js/src/gateway/hooks/useListModels.ts`

```typescript
import { useQuery } from "@tanstack/react-query";

interface Model {
  model_name: string;
  provider: string;
  mode: string;
  max_input_tokens?: number;
  max_output_tokens?: number;
  supports_vision?: boolean;
  supports_function_calling?: boolean;
  supports_streaming?: boolean;
}

export function useListModels(provider: string | null, mode?: string) {
  return useQuery<Model[]>({
    queryKey: ["models", provider, mode],
    queryFn: async () => {
      if (!provider) return [];

      const params = new URLSearchParams({ provider });
      if (mode) {
        params.append("mode", mode);
      }

      const response = await fetch(`/ajax-api/3.0/mlflow/gateway/models/list?${params}`);
      if (!response.ok) {
        throw new Error("Failed to fetch models");
      }
      const data = await response.json();
      return data.models;
    },
    enabled: !!provider, // Only fetch when provider is selected
    staleTime: 1000 * 60 * 60, // Cache for 1 hour
  });
}
```

#### 3.2 Update Route Creation UI

**Update File:** `mlflow/server/js/src/secrets/hooks/useCreateRoute.ts`

Change from text inputs to dropdowns:

```typescript
// Before: User manually types provider and model
<Input name="provider" placeholder="e.g., openai" />
<Input name="model_name" placeholder="e.g., gpt-4" />

// After: User selects from dropdowns
<ProviderSelector
  value={selectedProvider}
  onChange={setSelectedProvider}
/>
<ModelSelector
  provider={selectedProvider}
  value={selectedModel}
  onChange={setSelectedModel}
/>
```

#### 3.3 Create Selector Components

**New File:** `mlflow/server/js/src/gateway/components/ProviderSelector.tsx`

```typescript
import { Select } from "@databricks/design-system";
import { useListProviders } from "../hooks/useListProviders";

interface ProviderSelectorProps {
  value: string | null;
  onChange: (value: string) => void;
}

export function ProviderSelector({ value, onChange }: ProviderSelectorProps) {
  const { data: providers, isLoading } = useListProviders();

  return (
    <Select value={value} onChange={onChange} loading={isLoading} placeholder="Select provider">
      {providers?.map((provider) => (
        <Select.Option key={provider.name} value={provider.name}>
          {provider.display_name}
        </Select.Option>
      ))}
    </Select>
  );
}
```

**New File:** `mlflow/server/js/src/gateway/components/ModelSelector.tsx`

```typescript
import { Select } from "@databricks/design-system";
import { useListModels } from "../hooks/useListModels";

interface ModelSelectorProps {
  provider: string | null;
  value: string | null;
  onChange: (value: string) => void;
  mode?: string;
}

export function ModelSelector({ provider, value, onChange, mode }: ModelSelectorProps) {
  const { data: models, isLoading } = useListModels(provider, mode);

  return (
    <Select
      value={value}
      onChange={onChange}
      loading={isLoading}
      disabled={!provider}
      placeholder={provider ? "Select model" : "Select provider first"}
    >
      {models?.map((model) => (
        <Select.Option key={model.model_name} value={model.model_name}>
          {model.model_name}
          {model.max_input_tokens && (
            <span style={{ color: "gray", fontSize: "0.85em" }}>
              {" "}
              ({model.max_input_tokens} tokens)
            </span>
          )}
        </Select.Option>
      ))}
    </Select>
  );
}
```

## Key Decisions & Rationale

### 1. No Schema Changes

**Decision:** Keep existing schema as-is.

**Rationale:**

- Current `provider` field in `SqlSecret` already supports what we need
- `model_name` in `SqlSecretRoute` is sufficient for model identifiers
- Adding an `endpoint_models` table (as suggested in original plan) is premature optimization
- We can add multi-model support later if needed

### 2. Lazy Loading LiteLLM

**Decision:** Use lazy imports and graceful error handling when LiteLLM is not installed.

**Rationale:**

- LiteLLM is an optional dependency (not everyone needs it)
- Core MLflow should work without it
- Clear error messages guide users to install `mlflow[genai]`

### 3. Cache Provider/Model Lists

**Decision:** Cache provider and model lists in frontend for 1 hour.

**Rationale:**

- Provider/model catalogs don't change frequently
- Reduces API calls and improves UX
- 1 hour is reasonable balance between freshness and performance

### 4. API Version 3.0

**Decision:** Use `/ajax-api/3.0/` for new gateway endpoints.

**Rationale:**

- Consistent with recent backend-info endpoint
- Separate from legacy gateway module (which may use different versions)

## Implementation Order

1. **Add `mlflow[genai]` extra** to `pyproject.toml` ✅ Quick win
2. **Add protobuf definitions** and generate code
3. **Create `litellm_catalog.py`** module with provider/model listing
4. **Add backend handlers** for new APIs
5. **Write backend tests** for provider/model listing
6. **Create frontend hooks** (`useListProviders`, `useListModels`)
7. **Create selector components** (`ProviderSelector`, `ModelSelector`)
8. **Update route creation UI** to use selectors
9. **Add frontend tests** for new components
10. **Update documentation**

## Testing Strategy

### Backend Tests

```python
def test_list_providers():
    """Test provider listing returns valid providers."""
    response = client.get("/ajax-api/3.0/mlflow/gateway/providers/list")
    assert response.status_code == 200
    providers = response.json()["providers"]
    assert len(providers) > 0
    assert any(p["name"] == "openai" for p in providers)


def test_list_models():
    """Test model listing for OpenAI."""
    response = client.get("/ajax-api/3.0/mlflow/gateway/models/list?provider=openai")
    assert response.status_code == 200
    models = response.json()["models"]
    assert len(models) > 0
    assert any(m["model_name"] == "gpt-4" for m in models)


def test_list_models_with_mode_filter():
    """Test model listing filtered by mode."""
    response = client.get("/ajax-api/3.0/mlflow/gateway/models/list?provider=openai&mode=embedding")
    assert response.status_code == 200
    models = response.json()["models"]
    assert all(m["mode"] == "embedding" for m in models)
```

### Frontend Tests

```typescript
describe("ProviderSelector", () => {
  it("renders provider options", async () => {
    render(<ProviderSelector value={null} onChange={jest.fn()} />);
    await waitFor(() => {
      expect(screen.getByText("OpenAI")).toBeInTheDocument();
      expect(screen.getByText("Anthropic")).toBeInTheDocument();
    });
  });
});

describe("ModelSelector", () => {
  it("disables when no provider selected", () => {
    render(<ModelSelector provider={null} value={null} onChange={jest.fn()} />);
    expect(screen.getByRole("combobox")).toBeDisabled();
  });

  it("fetches models when provider selected", async () => {
    render(<ModelSelector provider="openai" value={null} onChange={jest.fn()} />);
    await waitFor(() => {
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
    });
  });
});
```

## Future Enhancements (Not in Scope)

These are **out of scope** for initial implementation but worth noting:

1. **Dynamic endpoint management** (start/stop endpoints) - Requires FastAPI integration
2. **Multi-model support** (traffic splitting, failover) - Requires `endpoint_models` table
3. **Endpoint status monitoring** - Requires metrics collection
4. **Integration with Databricks Model Serving** - Requires Databricks-specific code
5. **Custom provider plugins** - Requires plugin architecture

## Questions for Review

1. Should we add endpoint_type (chat/embedding/etc) to `SqlSecretRoute` now, or wait?
2. Do we want to support "custom" providers (not in LiteLLM catalog)?
3. Should model selector show pricing information from catalog?
4. How should we handle provider authentication complexity (e.g., AWS Bedrock requires region + credentials)?

## Next Steps

After review and approval:

1. Create feature branch
2. Implement Phase 1 (add genai extra)
3. Implement Phase 2 (backend APIs)
4. Implement Phase 3 (frontend updates)
5. Write tests
6. Create PR with demo video

---

**Document Version:** 1.0
**Last Updated:** 2025-11-17
**Author:** Analysis by Claude Code
