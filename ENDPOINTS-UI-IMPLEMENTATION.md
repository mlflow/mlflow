# Endpoints UI Implementation Guide

> **Working Document** - Last Updated: 2025-01-XX
> This document tracks the design, architecture, and implementation progress for the MLflow Gateway Endpoints UI feature.

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Implementation Status](#implementation-status)
4. [Component Reference](#component-reference)
5. [Remaining Work](#remaining-work)

---

## Overview

### Goal
Build a full-featured UI for creating and managing LLM gateway endpoints in MLflow, following Databricks Serving Endpoints UX patterns.

### Implemented Pages
1. **Gateway Page** (`/gateway`) - Table of all endpoints with create button
2. **Create Endpoint Page** (`/gateway/create`) - Full-page form with summary sidebar
3. **Endpoint Details Page** (`/gateway/:endpointId`) - View endpoint config, model details, API key info
4. **Edit Endpoint Page** (`/gateway/:endpointId/edit`) - Edit existing endpoint configuration

### Design Decisions

#### Full-Page Forms (Not Modals)
We chose full-page forms over modals for Create and Edit flows because:
- Better UX for complex forms with multiple sections
- Summary sidebar provides real-time feedback on configuration
- More space for provider/model metadata display
- Consistent with Databricks' serving endpoints patterns

#### Single Model Per Endpoint (v1)
The backend supports multiple models per endpoint, but v1 UI keeps it simple with a single model selection. Multi-model support can be added in a future iteration.

#### TypeaheadCombobox for Provider Selection
Provider selection uses `TypeaheadCombobox` with section headers to enable:
- Type-ahead search filtering
- Grouped display (Common Providers / Other Providers)
- Searchable by both display name and internal provider key

---

## Architecture

### Directory Structure

```
mlflow/server/js/src/gateway/
├── pages/
│   ├── GatewayPage.tsx              # Main endpoints list page
│   ├── CreateEndpointPage.tsx       # Full-page create form with sidebar
│   ├── EndpointDetailsPage.tsx      # View endpoint details
│   └── EditEndpointPage.tsx         # Edit existing endpoint
├── components/
│   ├── endpoints/
│   │   ├── CreateEndpointButton.tsx # Button linking to create page
│   │   └── EndpointsList.tsx        # Table of endpoints with actions
│   ├── create-endpoint/
│   │   ├── ProviderSelect.tsx       # TypeaheadCombobox with grouped providers
│   │   └── ModelSelect.tsx          # Model dropdown with metadata display
│   └── secrets/
│       ├── SecretSelector.tsx       # Select existing secret dropdown
│       ├── SecretFormFields.tsx     # Controlled form fields (REUSABLE)
│       ├── CreateSecretForm.tsx     # Complete secret creation form
│       ├── SecretConfigSection.tsx  # Mode toggle + secret selection/creation
│       ├── types.ts                 # Shared types (SecretFormData)
│       └── index.ts                 # Clean exports
├── hooks/
│   ├── useCreateEndpointMutation.tsx
│   ├── useEndpointsQuery.tsx
│   ├── useDeleteEndpointMutation.tsx
│   ├── useProvidersQuery.tsx
│   ├── useModelsQuery.tsx
│   ├── useSecretsQuery.tsx
│   └── useCreateSecretMutation.tsx
├── api.ts                           # API client
├── types.ts                         # TypeScript types
├── routes.ts                        # Route definitions
├── route-defs.ts                    # Route component mapping
└── utils/
    └── providerUtils.ts             # Provider grouping and formatting
```

### Component Hierarchy

```
GatewayPage
├── PageHeader
│   └── CreateEndpointButton → navigates to /gateway/create
└── EndpointsList
    └── Table rows → navigate to /gateway/:endpointId

CreateEndpointPage
├── Breadcrumb
├── Form (React Hook Form + FormProvider)
│   ├── LongFormSection: General
│   │   └── Name input
│   ├── LongFormSection: Model
│   │   ├── ProviderSelect (TypeaheadCombobox)
│   │   └── ModelSelect
│   └── LongFormSection: Authentication
│       └── SecretConfigSection
│           ├── Mode toggle (Existing / New)
│           ├── SecretSelector (if existing)
│           └── SecretFormFields (if new)
├── LongFormSummary (sidebar)
│   ├── Provider tag
│   ├── Model info + capabilities
│   └── Authentication mode
└── Footer (Cancel / Create buttons)

EndpointDetailsPage
├── Breadcrumb
├── Header (Title + Edit button)
├── Active Configuration Card
│   ├── Provider tag
│   ├── ModelCard (name, capabilities, context, costs)
│   └── SecretDetails (name, provider, masked key, timestamps, ID)
└── About Card (Created, Last modified, Created by)

EditEndpointPage
├── Breadcrumb (Gateway → Endpoint Name)
├── Same form structure as CreateEndpointPage
└── Footer (Cancel / Save buttons)
```

---

## Implementation Status

### Phase 1: Foundation ✅
- [x] Directory structure and routing
- [x] API types in `types.ts`
- [x] API client methods in `api.ts`
- [x] All data fetching hooks (providers, models, secrets, endpoints)
- [x] All mutation hooks (create endpoint, create secret, delete endpoint, update endpoint, update model)

### Phase 2: Reusable Components ✅
- [x] `ProviderSelect` - TypeaheadCombobox with grouped providers and search
- [x] `ModelSelect` - Dropdown with model metadata display
- [x] `SecretSelector` - Dropdown filtered by provider
- [x] `SecretFormFields` - Controlled component for standalone or nested use
- [x] `CreateSecretForm` - Complete form with validation
- [x] `SecretConfigSection` - Mode toggle + selector/form integration
- [x] `providerUtils.ts` - Provider grouping (common/other) and display name formatting

### Phase 3: Pages ✅
- [x] `GatewayPage` - Endpoints list with create button and delete actions
- [x] `CreateEndpointPage` - Full-page form with summary sidebar
- [x] `EndpointDetailsPage` - View with ModelCard, SecretDetails, metadata
- [x] `EditEndpointPage` - Edit form pre-populated from existing endpoint

### Phase 4: Polish ✅
- [x] Form validation (required fields, disable submit until complete)
- [x] Disabled button tooltip explaining required fields
- [x] Error handling with Alert components
- [x] Loading states with Spinner
- [x] Browser autocomplete prevention on API key fields
- [x] Provider search with TypeaheadCombobox
- [x] Model capabilities display (Tools, Reasoning, Vision, Caching)
- [x] Cost and context window display

---

## Component Reference

### ProviderSelect
```typescript
interface ProviderSelectProps {
  value: string;
  onChange: (provider: string) => void;
  disabled?: boolean;
  error?: string;
  componentIdPrefix?: string;  // For telemetry reuse
}
```

Uses `TypeaheadCombobox` with `TypeaheadComboboxSectionHeader` for grouped display. Searches both display name and internal provider key.

### SecretFormFields (Reusable)
```typescript
interface SecretFormFieldsProps {
  provider: string;
  value: SecretFormData;
  onChange: (value: SecretFormData) => void;
  errors?: { name?: string; value?: string; authConfig?: Record<string, string> };
  disabled?: boolean;
  componentIdPrefix?: string;
  hideNameField?: boolean;  // For edit scenarios
}
```

Controlled component that works both standalone and nested within React Hook Form.

### SecretConfigSection (Reusable)
```typescript
interface SecretConfigSectionProps {
  provider: string;
  mode: SecretMode;  // 'existing' | 'new'
  onModeChange: (mode: SecretMode) => void;
  selectedSecretId: string;
  onSecretSelect: (secretId: string) => void;
  newSecretFieldPrefix: string;  // For RHF field names
  showModeSelector?: boolean;    // Default true
  label?: string;
  componentIdPrefix?: string;
}
```

### Common Providers (Top of Dropdown)
```typescript
const COMMON_PROVIDERS = [
  'openai',
  'anthropic',
  'bedrock',
  'gemini',
  'azure',
  'groq',
  'databricks'
];
```

---

## Remaining Work

### Phase 5: API Key Management Page (Next PR)

#### Overview
A dedicated page for managing API keys/secrets with full CRUD operations and visibility into which endpoints and entities are using each key.

#### New Pages
1. **API Keys List Page** (`/gateway/keys`) - Table of all secrets with actions
2. **API Key Details Drawer** - Slide-out panel showing:
   - Secret metadata (name, provider, masked value, timestamps)
   - List of endpoints using this secret
   - List of bound entities (experiments, etc.) using endpoints with this secret
   - Unbind/Delete actions

#### Required Components
- [ ] `APIKeysPage` - Main list page with create button
- [ ] `APIKeysList` - Table of secrets with columns: Name, Provider, Masked Value, Created, Actions
- [ ] `APIKeyDetailsDrawer` - Drawer component for viewing/managing a single key
- [ ] `APIKeyEndpointsList` - List of endpoints using the key
- [ ] `APIKeyBoundEntitiesList` - List of bound entities (experiments, etc.)
- [ ] `CreateAPIKeyButton` - Opens create form (can reuse `CreateSecretForm`)

#### Required API Extensions
- [ ] Get endpoints by secret ID (or extend existing list endpoint with filter)
- [ ] List endpoint bindings by endpoint ID
- [ ] Delete endpoint binding API integration

#### Required Hooks
- [ ] `useSecretEndpointsQuery` - Get endpoints using a specific secret
- [ ] `useEndpointBindingsQuery` - Get entities bound to an endpoint
- [ ] `useDeleteEndpointBindingMutation` - Unbind entity from endpoint
- [ ] `useUpdateSecretMutation` - Update secret value
- [ ] `useDeleteSecretMutation` - Delete secret (with validation)

#### UX Considerations
- Prevent deletion of secrets that are in use by endpoints
- Show warning when unbinding will affect active endpoints
- Bulk unbind capability for managing many bindings
- Search/filter by provider, name, bound entity

#### Drawer Design
```
┌─────────────────────────────────────┐
│ API Key: my-openai-key          [X] │
├─────────────────────────────────────┤
│ Provider: OpenAI                    │
│ Masked Key: sk-...abc123            │
│ Created: 2 days ago by user@...     │
│ Last Updated: 1 day ago             │
│ Secret ID: sec_abc123...            │
├─────────────────────────────────────┤
│ ENDPOINTS USING THIS KEY (3)        │
│ ┌─────────────────────────────────┐ │
│ │ my-gpt4-endpoint                │ │
│ │ claude-sonnet-prod              │ │
│ │ test-endpoint                   │ │
│ └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│ BOUND ENTITIES (5)                  │
│ ┌─────────────────────────────────┐ │
│ │ Experiment: ml-pipeline    [x]  │ │
│ │ Experiment: eval-tests     [x]  │ │
│ │ ...                             │ │
│ └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│ [Update Key]  [Delete Key]          │
└─────────────────────────────────────┘
```

### Future Enhancements (Beyond Next PR)
- [ ] Multi-model endpoint support
- [ ] Endpoint usage metrics/analytics
- [ ] Rate limiting configuration
- [ ] Endpoint health status monitoring
- [ ] Bulk operations (delete multiple endpoints, etc.)

---

## Session History

### Session 1
- Created initial working doc
- Explored codebase patterns (prompts, eval datasets, tracing)
- Identified reusable component strategy
- Received `EndpointLongFormContainer.tsx` reference

### Session 2
- Implemented all Phase 1-3 components
- Learned `FormUI.Item` doesn't exist - use `FormUI.Label` + separate components
- Used `SimpleSelectOptionGroup` for grouped provider dropdown

### Session 3 (Current)
- Converted Create Endpoint from modal to full-page form
- Added summary sidebar with model metadata display
- Implemented Endpoint Details page with ModelCard and SecretDetails
- Implemented Edit Endpoint page with form pre-population
- Added TypeaheadCombobox for provider search
- Added form validation with disabled button + tooltip
- Updated provider grouping (swapped vertex_ai for gemini)
- Added browser autocomplete prevention on API key fields
- Added masked_value display for secrets
- Documented remaining work for API Key management page
