# AI Gateway Improvements Plan

## Overview

This document outlines the planned improvements to transform the current secrets/routes system into a full-featured AI Gateway implementation using LiteLLM.

## Current Issues to Address First

- [ ] Fix 500 errors in UI
- [ ] Stabilize current implementation

---

## Phase 1: Terminology & UI Refinements

### 1.1 Rename "Secrets" → "Keys"

**Files to update:**

- Frontend components (all references to "secret")
- API routes and handlers
- Database schema (consider migration)
- Documentation

**Rationale:** "Keys" is more accurate terminology for API credentials.

### 1.2 Change "Routes" → "Endpoints"

**Files to update:**

- Frontend components
- Backend routes/handlers
- Database schema
- Protobuf definitions

**Rationale:** Aligns with Databricks terminology and allows for multiple models per endpoint.

### 1.3 ACL Sharing UI

**Requirements:**

- [ ] Verify sharing UI exists for keys
- [ ] Ensure permission levels (READ, EDIT, MANAGE) are properly exposed
- [ ] Test multi-user sharing scenarios

---

## Phase 2: LiteLLM Integration

### 2.1 Add LiteLLM Dependency

**File:** `pyproject.toml`

```toml
[project.optional-dependencies]
genai = [
    "litellm>=1.0.0",  # Version TBD after research
    # ... existing dependencies
]
```

### 2.2 LiteLLM Model Catalog Integration

**Research findings from `/tmp/litellm/model_prices_and_context_window.json`:**

- Each model entry contains:
  - `litellm_provider`: Provider name (openai, anthropic, bedrock, etc.)
  - `mode`: Endpoint type (chat, embedding, completion, image_generation, etc.)
  - `max_input_tokens`, `max_output_tokens`: Token limits
  - `supports_*`: Feature flags (vision, function_calling, reasoning, etc.)
  - Cost information
  - Regional availability

**Key structure example:**

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

### 2.3 Provider Listing API

**New Backend Route:** `/api/3.0/mlflow/gateway/providers/list`

**Implementation approach:**

1. Import LiteLLM's model catalog
2. Extract unique provider names
3. Return provider metadata (name, supported modes, description)

**Response format:**

```json
{
  "providers": [
    {
      "name": "openai",
      "display_name": "OpenAI",
      "supported_modes": ["chat", "embedding", "image_generation"],
      "description": "OpenAI's GPT models and APIs"
    },
    {
      "name": "anthropic",
      "display_name": "Anthropic",
      "supported_modes": ["chat"],
      "description": "Anthropic's Claude models"
    }
  ]
}
```

### 2.4 Model Listing API

**New Backend Route:** `/api/3.0/mlflow/gateway/models/list`

**Query parameters:**

- `provider`: Filter by provider name (required)
- `mode`: Filter by endpoint type (optional)

**Implementation approach:**

1. Filter LiteLLM catalog by provider
2. Optionally filter by mode
3. Return model metadata with capabilities

**Response format:**

```json
{
  "models": [
    {
      "model_name": "gpt-4",
      "provider": "openai",
      "mode": "chat",
      "max_input_tokens": 8192,
      "max_output_tokens": 4096,
      "supports_vision": true,
      "supports_function_calling": true,
      "supports_streaming": true
    }
  ]
}
```

### 2.5 UI Workflow Updates

**Current flow (deprecated):**

1. User enters provider manually
2. User enters model name manually

**New flow:**

1. User selects provider from dropdown (populated via `/providers/list`)
2. User selects model from dropdown (populated via `/models/list?provider=X`)
3. User selects endpoint type (chat, embedding, etc.)
4. User configures endpoint-specific settings

**Files to update:**

- `mlflow/server/js/src/secrets/components/CreateSecretModal.tsx` → `CreateKeyModal.tsx`
- `mlflow/server/js/src/secrets/components/RouteDetailDrawer.tsx` → `EndpointDetailDrawer.tsx`
- Add new route selection components

---

## Phase 3: Data Model Restructure

### 3.1 New "Models" Table

**Rationale:** Endpoints can have multiple models (for traffic splitting, A/B testing, failover)

**Schema:**

```sql
CREATE TABLE endpoint_models (
    id VARCHAR(32) PRIMARY KEY,
    endpoint_id VARCHAR(32) NOT NULL,  -- FK to routes/endpoints
    model_name VARCHAR(256) NOT NULL,
    provider VARCHAR(128) NOT NULL,
    weight FLOAT DEFAULT 1.0,  -- For traffic splitting
    priority INTEGER DEFAULT 0,  -- For failover ordering
    litellm_params JSON,  -- Provider-specific configuration
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    FOREIGN KEY (endpoint_id) REFERENCES routes(route_id)
);
```

**New relationship:**

```
keys -> endpoints <-> models -> resources (bindings)
```

### 3.2 Endpoint Table Updates

**Add columns:**

- `endpoint_type` VARCHAR(64): 'chat', 'embedding', 'image_generation', etc.
- `status` VARCHAR(32): 'running', 'stopped', 'starting', 'error'
- `started_at` TIMESTAMP: When endpoint was last started
- `stopped_at` TIMESTAMP: When endpoint was last stopped

**Migration considerations:**

- Backfill endpoint_type from existing model configurations
- Default status to 'stopped' for existing endpoints

---

## Phase 4: Dynamic Endpoint Management

### 4.1 Endpoint Lifecycle APIs

**New Backend Routes:**

1. **Start Endpoint:** `POST /api/3.0/mlflow/gateway/endpoints/{endpoint_id}/start`

   - Creates dynamic FastAPI route for the endpoint
   - Initializes LiteLLM router with configured models
   - Updates status to 'running'
   - Returns endpoint URL

2. **Stop Endpoint:** `POST /api/3.0/mlflow/gateway/endpoints/{endpoint_id}/stop`

   - Removes dynamic FastAPI route
   - Cleans up LiteLLM router
   - Updates status to 'stopped'

3. **Get Endpoint Status:** `GET /api/3.0/mlflow/gateway/endpoints/{endpoint_id}/status`
   - Returns current status and metadata
   - Includes request counts, error rates, latency metrics

**Implementation approach:**

- Use FastAPI's dynamic route registration
- Integrate with LiteLLM's Router for multi-model support
- Store route handles for cleanup

**Reference:** LiteLLM proxy server implementation at `/tmp/litellm/litellm/proxy/proxy_server.py`

### 4.2 UI Controls

**Add to Endpoint Detail Page:**

```tsx
// Buttons
<Button onClick={handleStartEndpoint}>Start Endpoint</Button>
<Button onClick={handleStopEndpoint}>Stop Endpoint</Button>
<Button onClick={handleDeleteEndpoint} danger>Delete Endpoint</Button>

// Status indicator
<StatusBadge status={endpoint.status} />
// running (green), stopped (gray), starting (yellow), error (red)
```

**Status polling:**

- Poll endpoint status every 5 seconds when status is 'starting'
- Show loading indicator during state transitions

---

## Phase 5: Scorer Integration

### 5.1 Create Endpoint from Scorer UI

**Requirements:**

- Add "Create Endpoint" button in scorer creation flow
- Pre-populate endpoint configuration with scorer model requirements
- Link scorer to endpoint_id after creation

**Future work:** Keep this in mind but not immediate priority.

---

## Phase 6: Multi-Model Support

### 6.1 Traffic Splitting Configuration

**UI for configuring multiple models:**

```tsx
<ModelList>
  <ModelItem model="gpt-4" weight={0.7} />
  <ModelItem model="gpt-4-turbo" weight={0.3} />
</ModelList>
```

**Backend:**

- LiteLLM Router handles weighted routing automatically
- Validate that weights sum to 1.0

### 6.2 Failover Configuration

**UI for configuring failover priority:**

```tsx
<ModelList>
  <ModelItem model="gpt-4" priority={1} />
  <ModelItem model="gpt-3.5-turbo" priority={2} fallback />
</ModelList>
```

---

## Phase 7: Endpoint Types & Multi-Modal Support

### 7.1 Supported Endpoint Types

Based on LiteLLM's `mode` field:

- `chat`: LLM chat completions (OpenAI `/v1/chat/completions` compatible)
- `embedding`: Text embeddings
- `completion`: Legacy completion API
- `image_generation`: Image generation (DALL-E, Stable Diffusion)
- `audio_transcription`: Whisper-style transcription
- `audio_speech`: Text-to-speech
- `moderation`: Content moderation
- `rerank`: Search result reranking
- `search`: Semantic search

### 7.2 Provider-Specific Passthrough

**LiteLLM supports native provider APIs:**

- `gemini`: Google Gemini API format
- `bedrock`: AWS Bedrock format
- `anthropic`: Anthropic Messages API
- `vertex`: Vertex AI format

**Configuration:**

```json
{
  "endpoint_type": "gemini",
  "passthrough": true,
  "model": "gemini-pro"
}
```

---

## Research Questions & TODOs

### LiteLLM Integration

- [ ] Investigate LiteLLM Router initialization and lifecycle
- [ ] Understand LiteLLM's authentication/key management
- [ ] Test dynamic route creation/deletion in FastAPI
- [ ] Review LiteLLM's error handling and retry logic
- [ ] Check LiteLLM's metrics/observability integration

### Database Migrations

- [ ] Design migration path from current schema to new schema
- [ ] Plan data backfill strategy for new columns
- [ ] Consider backwards compatibility requirements

### Security & Auth

- [ ] Key encryption at rest (already implemented?)
- [ ] Key rotation strategy
- [ ] Audit logging for endpoint start/stop operations

### Performance

- [ ] Caching strategy for provider/model listings
- [ ] Connection pooling for model providers
- [ ] Rate limiting per endpoint

---

## Demo Plan (Wednesday)

### Demo with Judges

- [ ] Show end-to-end scorer creation with automatic endpoint setup
- [ ] Demonstrate endpoint management (start/stop/status)
- [ ] Show multi-model configuration
- [ ] Display request routing and failover
- [ ] Metrics and observability integration

**Prep work:**

- [ ] Create sample scorers with different providers
- [ ] Set up test endpoints with traffic splitting
- [ ] Prepare monitoring dashboard
- [ ] Document demo script

---

## Implementation Phases

### Phase 1: Foundation (Week 1)

1. Fix current 500 errors
2. Rename terminology (Secrets → Keys, Routes → Endpoints)
3. Add LiteLLM dependency
4. Implement provider/model listing APIs

### Phase 2: Dynamic Endpoints (Week 2)

1. Add endpoint_type to schema
2. Implement start/stop APIs
3. Add UI controls for endpoint lifecycle
4. Test dynamic route creation

### Phase 3: Multi-Model Support (Week 3)

1. Create endpoint_models table
2. Implement model association APIs
3. Add traffic splitting configuration UI
4. Test LiteLLM Router integration

### Phase 4: Polish & Demo (Week 4)

1. Add metrics and monitoring
2. Improve error handling and UX
3. Write documentation
4. Prepare demo

---

## Files to Create/Modify

### Backend (Python)

- [ ] `mlflow/protos/service.proto` - Add new endpoint lifecycle RPCs
- [ ] `mlflow/server/handlers.py` - Add provider/model listing handlers
- [ ] `mlflow/server/auth/sqlalchemy_store.py` - Add endpoint lifecycle methods
- [ ] `mlflow/gateway/` - New module for LiteLLM integration
- [ ] `mlflow/gateway/router.py` - Dynamic endpoint router
- [ ] `mlflow/gateway/providers.py` - Provider/model catalog wrapper
- [ ] Database migration scripts

### Frontend (TypeScript/React)

- [ ] `mlflow/server/js/src/gateway/` - Rename from secrets/
- [ ] `EndpointDetailDrawer.tsx` - Add lifecycle controls
- [ ] `ProviderSelector.tsx` - New provider selection component
- [ ] `ModelSelector.tsx` - New model selection component
- [ ] `EndpointStatusBadge.tsx` - Status indicator component
- [ ] Update all imports and references

### Tests

- [ ] `tests/gateway/test_providers.py` - Provider listing tests
- [ ] `tests/gateway/test_models.py` - Model listing tests
- [ ] `tests/gateway/test_endpoint_lifecycle.py` - Start/stop tests
- [ ] `tests/server/auth/test_auth.py` - Update auth tests for new terminology

---

## References

### LiteLLM Documentation

- Provider docs: https://docs.litellm.ai/docs/providers
- Router docs: https://docs.litellm.ai/docs/routing
- Proxy docs: https://docs.litellm.ai/docs/proxy/quick_start

### LiteLLM Source Code (Cloned to /tmp/litellm)

- Model catalog: `/tmp/litellm/model_prices_and_context_window.json`
- Proxy server: `/tmp/litellm/litellm/proxy/proxy_server.py`
- Model management: `/tmp/litellm/litellm/proxy/management_endpoints/model_management_endpoints.py`
- Types: `/tmp/litellm/litellm/proxy/_types.py`

---

## Notes

- Keep backwards compatibility where possible
- Maintain existing auth/permissions system
- Ensure smooth migration path for existing users
- Document all breaking changes
- Consider feature flags for gradual rollout
