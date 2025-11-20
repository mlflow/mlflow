# Secrets UI Update Plan - Route-Based Architecture

## Current State vs Target State

### What We Have (Stashed)

- ✅ Basic UI structure (SecretsPage, SecretsTable, DetailDrawer)
- ✅ CRUD modals (Create, Update, Delete)
- ✅ API client layer with React Query hooks
- ✅ Sidebar navigation integration
- ⚠️ **Built for old direct binding model**: Secret → Binding (2-table)

### What We Need (Route-Based)

- ✅ Same UI structure (can reuse most components)
- 🔄 **Updated data model**: Secret → Route → Binding (3-table)
- 🆕 **New API**: `CreateRouteAndBind` endpoint
- 🔄 **Updated modals**: Handle `model_name` and route metadata
- 🆕 **Routes display**: Show routes in addition to bindings

---

## Key Architecture Changes

### Old Model (Stashed UI)

```
Secret ─┬─> Binding 1 (resource_type, resource_id, field_name)
        ├─> Binding 2
        └─> Binding 3
```

### New Model (Route-Based)

```
Secret (provider: "openai") ─┬─> Route 1 (model_name: "gpt-4") ─┬─> Binding 1 (resource_type, resource_id, field_name)
                              │                                   └─> Binding 2
                              └─> Route 2 (model_name: "gpt-4o") ──> Binding 3
```

**Key Differences:**

1. **Provider is bound to the SECRET** (e.g., "openai", "anthropic", "azure")
   - All routes for this secret must use models from this provider
   - Can't select a bedrock model if the provider is "anthropic"
2. Each binding now belongs to a route
3. Routes have `model_name` (e.g., "gpt-4", "gpt-4o") which must be compatible with the secret's provider
4. Multiple bindings can share the same route (same model config)
5. Tags can be applied to routes (not just secrets)

---

## API Changes Required

### Updated Endpoints

| Old API (Stashed)     | New API (Route-Based)       | Changes                                               |
| --------------------- | --------------------------- | ----------------------------------------------------- |
| `CreateAndBindSecret` | `CreateAndBindSecret`       | **ADD**: `model_name` required field                  |
| `BindSecret`          | `BindSecret`                | **Deprecated** - use `CreateRouteAndBind` instead     |
| N/A                   | **`CreateRouteAndBind`** 🆕 | New endpoint for adding routes to existing secrets    |
| `UnbindSecret`        | `UnbindSecret`              | **No change** (still unbinds by resource coordinates) |
| `ListSecretBindings`  | `ListSecretBindings`        | **RETURNS**: Route info included in binding objects   |

### New TypeScript Interfaces

```typescript
// OLD (Stashed)
interface Secret {
  secret_id: string;
  secret_name: string;
  masked_value: string;
  is_shared: boolean;
  provider?: string; // Optional
  // ...
}

interface SecretBinding {
  binding_id: string;
  secret_id: string;
  resource_type: string;
  resource_id: string;
  field_name: string;
  // ...
}

// NEW (Route-Based)
interface Secret {
  secret_id: string;
  secret_name: string;
  masked_value: string;
  is_shared: boolean;
  provider: string; // REQUIRED: "openai", "anthropic", "azure", etc.
  // All routes for this secret must use models from this provider
  // ...
}

interface SecretRoute {
  route_id: string;
  secret_id: string;
  model_name: string; // REQUIRED (e.g., "gpt-4")
  // Must be compatible with secret's provider
  route_name?: string; // Optional friendly name
  route_description?: string;
  created_at: number;
  // ...
}

interface SecretBinding {
  binding_id: string;
  route_id: string; // CHANGED: was secret_id
  secret_id: string; // ADDED: for convenience
  resource_type: string;
  resource_id: string;
  field_name: string;
  // Route info included:
  route?: SecretRoute; // ADDED: populated by API
  // ...
}
```

---

## Component Updates Needed

### 1. CreateSecretModal (MAJOR UPDATE)

**Changes:**

- ✅ Keep: Secret name, secret value, is_shared toggle
- 🆕 **ADD**: `provider` field (REQUIRED at secret level)
  - Dropdown: "openai", "anthropic", "azure", etc.
  - This locks the secret to models from this provider
- 🆕 **ADD**: Secret tags (Optional)
  - Tag input component for adding key-value tags to the secret
- 🔄 **CHANGE**: Initial binding fields - **GLOBAL only**
  - `resource_type`: Fixed to "GLOBAL" (no dropdown)
  - `resource_id`: Auto-generated or fixed value
  - `field_name`: Text input (e.g., "OPENAI_API_KEY")
- 🆕 **ADD**: `model_name` field (REQUIRED for initial route)
  - **Filtered dropdown** showing only chat/reasoning models
  - Filtered by provider (e.g., only show gpt-4, gpt-4o, not dall-e or whisper)
  - Examples: "gpt-4", "gpt-4o", "claude-3-opus"
- 🆕 **ADD**: Route metadata (Optional)
  - `route_name` (friendly name)
  - `route_description`
  - `route_tags` (key-value tags)

**New Form Structure:**

```tsx
<Modal title="Create Secret" size="large">
  {/* Section 1: Secret Details */}
  <Typography.Title level={4}>Secret Details</Typography.Title>
  <Input
    label="Secret Name"
    required
    placeholder="my-openai-prod-key"
  />
  <PasswordInput
    label="Secret Value"
    required
    placeholder="sk-..."
  />
  <Select
    label="Provider"
    options={["openai", "anthropic", "azure", "bedrock"]}
    required
    description="All routes for this secret will use models from this provider"
  />
  <Switch
    label="Shared Secret"
    description="Allow this secret to be used with multiple model configurations"
  />
  <TagInput
    label="Secret Tags (Optional)"
    description="Add metadata tags to organize and identify this secret"
    placeholder="Add tag..."
  />

  {/* Section 2: Initial Route Configuration */}
  <Divider />
  <Typography.Title level={4}>Initial Route Configuration</Typography.Title>
  <Select
    label="Model"
    options={getFilteredModels(provider)} // Only chat/reasoning models
    required
    placeholder="Select a model"
    description={`Chat and reasoning models for ${provider}`}
  />
  <Input
    label="Route Name (Optional)"
    placeholder="Production GPT-4"
  />
  <Textarea
    label="Route Description (Optional)"
    placeholder="Used for production customer-facing chat..."
  />
  <TagInput
    label="Route Tags (Optional)"
    description="Add metadata tags to this route configuration"
    placeholder="Add tag..."
  />

  {/* Section 3: Global Binding */}
  <Divider />
  <Typography.Title level={4}>Environment Variable</Typography.Title>
  <Alert type="info">
    This secret will be available globally as an environment variable
  </Alert>
  <Input
    label="Field Name"
    required
    placeholder="OPENAI_API_KEY"
    description="Environment variable name for accessing this secret"
  />
</Modal>
```

### 2. SecretsTable (MINOR UPDATE)

**Changes:**

- ❌ **REMOVE**: `provider` column (moved to routes)
- 🔄 **UPDATE**: "Bindings" column → "Routes & Bindings"
  - Show count: "2 routes, 5 bindings"
- ✅ Keep: All other columns (name, masked_value, type, created_by, etc.)

**New Columns:**

```typescript
[
  { field: "secret_name", headerName: "Name" },
  { field: "masked_value", headerName: "Value" },
  { field: "is_shared", headerName: "Type", cellRenderer: typeBadge },
  { field: "route_count", headerName: "Routes" }, // NEW
  { field: "binding_count", headerName: "Bindings" }, // UPDATED
  { field: "created_by", headerName: "Created By" },
  { field: "last_updated_at", headerName: "Updated" },
  { field: "actions", headerName: "Actions" },
];
```

### 3. SecretDetailDrawer (MAJOR UPDATE)

**OLD Structure:**

```
Secret Details
├── Metadata (name, created_by, etc.)
└── Bindings List (flat list)
```

**NEW Structure:**

```
Secret Details
├── Metadata (name, created_by, etc.)
├── Routes Section 🆕
│   ├── Route 1 (gpt-4, openai)
│   │   ├── Binding 1 (SCORER_JOB:job-123:API_KEY)
│   │   └── Binding 2 (SCORER_JOB:job-456:API_KEY)
│   └── Route 2 (gpt-4o, openai)
│       └── Binding 3 (SCORER_JOB:job-789:API_KEY)
└── Actions
    ├── "Add Route" button 🆕
    └── "Update Secret" button
```

**Implementation:**

```tsx
<Drawer title="Secret Details">
  {/* Metadata */}
  <Section>
    <KeyValueList>
      <Item label="Name" value={secret.secret_name} />
      <Item label="Value" value={secret.masked_value} />
      <Item label="Type" value={secret.is_shared ? "Shared" : "Private"} />
      <Item label="Created By" value={secret.created_by} />
    </KeyValueList>
  </Section>

  {/* Routes & Bindings */}
  <Section title="Routes">
    {routes.map((route) => (
      <RouteCard key={route.route_id}>
        <RouteHeader>
          <Badge>{route.model_name}</Badge>
          <Typography.Text type="secondary">{route.provider}</Typography.Text>
          {route.route_name && <Typography.Text>{route.route_name}</Typography.Text>}
        </RouteHeader>

        {/* Bindings for this route */}
        <BindingsList>
          {route.bindings.map((binding) => (
            <BindingItem key={binding.binding_id}>
              <span>
                {binding.resource_type}:{binding.resource_id}
              </span>
              <span>{binding.field_name}</span>
              <IconButton icon={<DeleteIcon />} onClick={() => unbind(binding)} />
            </BindingItem>
          ))}
        </BindingsList>
      </RouteCard>
    ))}
  </Section>

  {/* Actions */}
  <ButtonGroup>
    <Button onClick={openAddRouteModal}>Add Route</Button>
    <Button onClick={openUpdateSecretModal}>Update Secret</Button>
  </ButtonGroup>
</Drawer>
```

### 4. AddRouteModal (NEW COMPONENT)

**Purpose**: Add a new route to an existing secret

```tsx
<Modal title={`Add Route to "${secret.secret_name}"`} size="medium">
  {/* Show the secret's provider (read-only) */}
  <Alert type="info">
    This secret is configured for <Badge>{secret.provider}</Badge> models only
  </Alert>

  {/* Route Config */}
  <Select
    label="Model"
    options={getFilteredModels(secret.provider)}
    required
    placeholder="Select a model"
    description={`Chat and reasoning models for ${secret.provider}`}
  />
  <Input
    label="Route Name (Optional)"
    placeholder="Staging GPT-4o"
  />
  <Textarea
    label="Route Description (Optional)"
    placeholder="Used for internal testing..."
  />
  <TagInput
    label="Route Tags (Optional)"
    description="Add metadata tags to this route configuration"
    placeholder="Add tag..."
  />

  {/* Initial Binding (Required - GLOBAL only) */}
  <Divider />
  <Typography.Title level={5}>Environment Variable</Typography.Title>
  <Alert type="info">
    This route will be available globally as an environment variable
  </Alert>
  <Input
    label="Field Name"
    required
    placeholder="OPENAI_API_KEY_STAGING"
    description="Environment variable name for this route"
  />
</Modal>
```

**API Call:** `CreateRouteAndBind`

**Note:**
- Model must be compatible with the secret's provider (backend validates)
- Only GLOBAL bindings supported in UI (resource_type fixed)

### 5. UpdateSecretModal (MINOR UPDATE)

**Changes:**

- ✅ Keep: Secret value input with masking
- 🔄 **UPDATE**: Warning message
  - OLD: "This will update X bindings"
  - NEW: "This will update X routes (affecting Y bindings)"
- ✅ Keep: Confirmation flow

### 6. DeleteSecretModal (MINOR UPDATE)

**Changes:**

- 🔄 **UPDATE**: Warning message to show routes + bindings
  - "This will delete 2 routes and 5 bindings across 3 resources"

---

## Model Filtering & Selection

### Provider-Model Mapping

Create a constants file for filtering models by provider and capability:

```typescript
// mlflow/server/js/src/secrets/constants/modelMapping.ts

export type ModelCapability = 'chat' | 'reasoning' | 'completion' | 'embedding' | 'image' | 'audio';

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  capabilities: ModelCapability[];
  deprecated?: boolean;
}

export const PROVIDER_MODELS: Record<string, ModelInfo[]> = {
  openai: [
    // Chat & Reasoning (SHOW in UI)
    { id: 'gpt-4', name: 'GPT-4', provider: 'openai', capabilities: ['chat', 'reasoning'] },
    { id: 'gpt-4-turbo', name: 'GPT-4 Turbo', provider: 'openai', capabilities: ['chat', 'reasoning'] },
    { id: 'gpt-4o', name: 'GPT-4o', provider: 'openai', capabilities: ['chat', 'reasoning'] },
    { id: 'gpt-4o-mini', name: 'GPT-4o Mini', provider: 'openai', capabilities: ['chat', 'reasoning'] },
    { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', provider: 'openai', capabilities: ['chat'] },
    { id: 'o1-preview', name: 'o1 Preview', provider: 'openai', capabilities: ['reasoning'] },
    { id: 'o1-mini', name: 'o1 Mini', provider: 'openai', capabilities: ['reasoning'] },

    // Image/Audio (HIDE from UI - filtered out)
    // { id: 'dall-e-3', name: 'DALL-E 3', provider: 'openai', capabilities: ['image'] },
    // { id: 'whisper-1', name: 'Whisper', provider: 'openai', capabilities: ['audio'] },
    // { id: 'tts-1', name: 'TTS', provider: 'openai', capabilities: ['audio'] },
  ],

  anthropic: [
    { id: 'claude-3-opus-20240229', name: 'Claude 3 Opus', provider: 'anthropic', capabilities: ['chat', 'reasoning'] },
    { id: 'claude-3-sonnet-20240229', name: 'Claude 3 Sonnet', provider: 'anthropic', capabilities: ['chat', 'reasoning'] },
    { id: 'claude-3-haiku-20240307', name: 'Claude 3 Haiku', provider: 'anthropic', capabilities: ['chat'] },
    { id: 'claude-3-5-sonnet-20241022', name: 'Claude 3.5 Sonnet', provider: 'anthropic', capabilities: ['chat', 'reasoning'] },
  ],

  azure: [
    { id: 'gpt-4', name: 'GPT-4 (Azure)', provider: 'azure', capabilities: ['chat', 'reasoning'] },
    { id: 'gpt-4-turbo', name: 'GPT-4 Turbo (Azure)', provider: 'azure', capabilities: ['chat', 'reasoning'] },
    { id: 'gpt-35-turbo', name: 'GPT-3.5 Turbo (Azure)', provider: 'azure', capabilities: ['chat'] },
  ],

  bedrock: [
    { id: 'anthropic.claude-3-opus-20240229-v1:0', name: 'Claude 3 Opus (Bedrock)', provider: 'bedrock', capabilities: ['chat', 'reasoning'] },
    { id: 'anthropic.claude-3-sonnet-20240229-v1:0', name: 'Claude 3 Sonnet (Bedrock)', provider: 'bedrock', capabilities: ['chat', 'reasoning'] },
  ],
};

/**
 * Get filtered models for a provider - only chat & reasoning capabilities
 */
export function getFilteredModels(provider: string): Array<{ value: string; label: string }> {
  const models = PROVIDER_MODELS[provider] || [];

  return models
    .filter(model =>
      // Only show chat and reasoning models (exclude image, audio, embedding)
      (model.capabilities.includes('chat') || model.capabilities.includes('reasoning')) &&
      !model.deprecated
    )
    .map(model => ({
      value: model.id,
      label: model.name,
    }));
}

/**
 * Validate if a model is compatible with a provider
 */
export function isModelCompatibleWithProvider(modelId: string, provider: string): boolean {
  const models = PROVIDER_MODELS[provider] || [];
  return models.some(model => model.id === modelId);
}
```

### Usage in Components

```typescript
// In CreateSecretModal.tsx
import { getFilteredModels } from '../constants/modelMapping';

const [provider, setProvider] = useState('openai');
const [modelName, setModelName] = useState('');

const modelOptions = useMemo(() => getFilteredModels(provider), [provider]);

// Render:
<Select
  label="Model"
  options={modelOptions}
  value={modelName}
  onChange={setModelName}
  placeholder="Select a model"
/>
```

---

## UX Flow Design

### Creation Flow (Happy Path)

```
1. User clicks "New Secret" button
   ↓
2. Modal opens with 3 sections:
   - Secret Details (name, value, provider, tags)
   - Route Configuration (model, name, description, tags)
   - Environment Variable (field name)
   ↓
3. User selects provider (e.g., "openai")
   → Model dropdown auto-filters to OpenAI chat models
   ↓
4. User selects model (e.g., "gpt-4o")
   ↓
5. User enters field name (e.g., "OPENAI_API_KEY")
   ↓
6. User clicks "Create"
   ↓
7. API call: CreateAndBindSecret
   → Creates: Secret + Route + Binding (GLOBAL)
   ↓
8. Success: Modal closes, table refreshes, toast notification
```

### Add Route Flow (Extending Existing Secret)

```
1. User selects secret in table
   ↓
2. Detail drawer opens showing existing routes
   ↓
3. User clicks "Add Route" button
   ↓
4. AddRouteModal opens
   - Shows provider badge (locked to secret's provider)
   - Model dropdown filtered to compatible models
   ↓
5. User selects different model (e.g., "gpt-4-turbo")
   ↓
6. User enters field name (e.g., "OPENAI_API_KEY_TURBO")
   ↓
7. User clicks "Create Route"
   ↓
8. API call: CreateRouteAndBind
   → Creates: Route + Binding (linked to existing secret)
   ↓
9. Success: Drawer updates showing new route
```

### UX Considerations

1. **Provider Lock-In Visual Cue**
   - When adding route, show prominent alert: "This secret uses OpenAI models only"
   - Disable incompatible models in dropdown

2. **Model Suggestion**
   - Show "Recommended" badge on popular models (gpt-4o, claude-3-5-sonnet)
   - Group by capability: "Latest Models", "Legacy Models"

3. **Field Name Validation**
   - Warn if field name already exists for GLOBAL binding
   - Suggest standard naming: `{PROVIDER}_{MODEL}_API_KEY`

4. **Tags UX**
   - Use tag input component similar to experiment tags
   - Suggest common tags: "environment:prod", "team:ml", "cost-center:eng"

5. **Provider Visual Indicator in Secret Selection**
   - When selecting existing secret to add route: Show provider with distinct styling
   - Example dropdown item: `my-openai-key` **`OpenAI`** (provider in different color/weight)
   - Use secondary text color and badge or pill styling for provider label
   - Helps users quickly identify which secrets are compatible with their model choice

6. **Empty States**
   - After creating first secret: Show success message with next steps
   - No routes yet: "Add a route to use this secret with a specific model"
   - **No secrets exist (Add Route clicked)**: Show alert with message and action
     - Message: "No secrets found. Create your first secret to get started."
     - Button: "Create New Secret" → Opens CreateSecretModal
     - UX flow: User clicks "Add Route" → Modal shows empty state → Clicks "Create New Secret" → CreateSecretModal opens → Creates secret+route → Returns to routes view

7. **Error Handling**
   - Provider mismatch: "Model 'claude-3-opus' is not compatible with provider 'openai'"
   - Duplicate field name: "Environment variable 'OPENAI_API_KEY' already exists"
   - Invalid secret value: "Secret value must be at least 8 characters"

---

## API Client Updates

### Old API Client (Stashed)

```typescript
// mlflow/server/js/src/secrets/api/secretsApi.ts
export const secretsApi = {
  createSecret: async (payload: {
    secret_name: string;
    secret_value: string;
    is_shared: boolean;
    resource_type: string;
    resource_id: string;
    field_name: string;
    provider?: string;  // Optional
  }) => {
    return fetch("/ajax-api/2.0/mlflow/secrets/create-and-bind", { ... });
  },

  bindSecret: async (payload: {
    secret_id: string;
    resource_type: string;
    resource_id: string;
    field_name: string;
  }) => {
    return fetch("/ajax-api/2.0/mlflow/secrets/bind", { ... });
  },
};
```

### New API Client (Route-Based)

```typescript
export const secretsApi = {
  createSecret: async (payload: {
    secret_name: string;
    secret_value: string;
    is_shared: boolean;
    resource_type: string;
    resource_id: string;
    field_name: string;
    model_name: string;     // REQUIRED
    provider?: string;      // Optional
    route_name?: string;    // Optional
    route_description?: string;  // Optional
  }) => {
    return fetch("/ajax-api/3.0/mlflow/secrets/create-and-bind", { ... });
  },

  createRouteAndBind: async (payload: {  // NEW
    secret_id: string;
    resource_type: string;
    resource_id: string;
    field_name: string;
    model_name: string;     // REQUIRED
    route_name?: string;
    route_description?: string;
    route_tags?: string;    // JSON array
  }) => {
    return fetch("/ajax-api/3.0/mlflow/secrets/create-route-and-bind", { ... });
  },

  // bindSecret: DEPRECATED - use createRouteAndBind
};
```

---

## Implementation Phases

### Phase 1: Type Definitions (1-2 hours)

1. Update `types.ts` with new interfaces
2. Add `SecretRoute` interface
3. Update `SecretBinding` to include `route_id` and optional `route`

### Phase 2: API Client (2-3 hours)

1. Update `secretsApi.ts` with new endpoints
2. Add `createRouteAndBind` method
3. Update `createSecret` to include `model_name`
4. Update all TypeScript types for payloads

### Phase 3: React Query Hooks (1-2 hours)

1. Update mutation hooks for new API signatures
2. Add `useCreateRouteAndBindMutation` hook
3. Update invalidation logic for routes

### Phase 4: Update Existing Components (4-6 hours)

1. **CreateSecretModal**: Add model_name and route fields
2. **SecretsTable**: Update columns (remove provider, add route count)
3. **SecretDetailDrawer**: Restructure to show routes → bindings hierarchy
4. **UpdateSecretModal**: Update warning messages
5. **DeleteSecretModal**: Update warning messages

### Phase 5: New Components (3-4 hours)

1. **AddRouteModal**: New modal for `CreateRouteAndBind`
2. **RouteCard**: Display route with its bindings
3. **BindingsList**: Nested component for route's bindings

### Phase 6: Testing & Polish (2-3 hours)

1. Manual testing of all flows
2. Fix any TypeScript errors
3. Update loading/error states
4. Add empty states

---

## Migration Strategy

### Option A: Unstash and Update (Recommended)

1. Unstash the UI prototype: `git stash pop stash@{1}`
2. Update types first (foundation)
3. Update API client
4. Update components one by one
5. Test incrementally

### Option B: Fresh Start (More Work)

1. Keep stash as reference
2. Rebuild components from scratch using new architecture
3. Copy reusable parts (styling, layout)

**Recommendation**: Option A - the UI structure is solid, we just need to update the data model.

---

## Key Questions Before Starting

1. **Route Display Priority**: Should we show routes prominently or hide them behind bindings?

   - Proposal: Show routes as expandable cards, bindings nested inside

2. **Model Name Input**: Free text or dropdown?

   - Proposal: Free text with suggestions/validation (common models for the selected provider)

3. **Provider Selection**: Dropdown with common providers or free text?

   - Proposal: Dropdown with common providers ("openai", "anthropic", "azure", "bedrock") + "Other" option for free text

4. **Model Validation**: Should we validate model compatibility with provider on frontend or backend only?

   - Proposal: Basic validation on frontend (helpful error messages), strict validation on backend

5. **Backward Compatibility**: Support old direct binding flow?

   - Proposal: No - force all bindings through routes

6. **Route Tags UI**: Where do we expose route tagging?
   - Proposal: Phase 2 feature - skip for initial implementation

---

## Success Criteria

- ✅ Users can create secrets with model_name
- ✅ Users can add multiple routes to a secret
- ✅ Users can see routes grouped with their bindings
- ✅ Users can unbind resources (same as before)
- ✅ All existing security practices maintained
- ✅ No TypeScript errors
- ✅ Consistent with MLflow design system

---

**Next Steps:**

1. Review this plan
2. Decide on key questions
3. Unstash prototype
4. Start Phase 1 (Type Definitions)
