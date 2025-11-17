# Routes UI Implementation Plan

## Current Status

**Date**: 2025-01-13
**Branch**: `stack/secrets/ui`
**Status**: 🚧 In Progress - Reimplementing frontend for route-centric architecture

### Problem Statement

The current UI is implementing the OLD secret-centric architecture, but the backend has been updated to support a route-centric architecture following `AI_GATEWAY_ARCHITECTURE.md`. We need to completely reimplement the frontend to match.

**Current Issues**:
1. UI shows "Secrets" page with "Create Secret" button (OLD architecture)
2. Shows "Shared" filter dropdown (not needed in route-centric design)
3. Backend list endpoints (`/ajax-api/3.0/mlflow/secrets/list` and `/api/3.0/mlflow/secrets/list-routes`) returning `{}` instead of proper response format
4. Missing "Add Route" button and routes table

**Expected**:
1. Main page titled "AI Gateway" or "Routes"
2. Two tabs: **Routes** (primary) and **API Keys** (management)
3. Routes table showing: Name, Model, Provider, Bindings
4. "Create Route" button (not "Create Secret")

---

## Architecture Overview

### Backend APIs (✅ Already Implemented)

#### Routes APIs
- `POST /api/3.0/mlflow/secrets/create-route-and-bind` - Create route with new or existing API key
- `GET /api/3.0/mlflow/secrets/list-routes` - List routes with joined secret info
- `POST /api/3.0/mlflow/secrets/update-route` - Update route configuration
- `DELETE /api/3.0/mlflow/secrets/delete-route` - Delete route and bindings

#### Secrets (API Keys) APIs
- `POST /ajax-api/3.0/mlflow/secrets/create` - Create standalone API key
- `GET /ajax-api/3.0/mlflow/secrets/list` - List API keys
- `POST /ajax-api/3.0/mlflow/secrets/update` - Rotate API key (affects all routes)
- `DELETE /ajax-api/3.0/mlflow/secrets/delete` - Delete API key (cascade deletes routes)

#### Bindings APIs
- `GET /api/3.0/mlflow/secrets/list-bindings?route_id=xxx` - List bindings for route
- `POST /api/3.0/mlflow/secrets/bind` - Bind route to resource
- `POST /api/3.0/mlflow/secrets/unbind` - Unbind route from resource

### Frontend Structure (🚧 To Be Implemented)

```
mlflow/server/js/src/secrets/
├── components/
│   ├── AIGatewayPage.tsx           # Main page with tabs (NEW)
│   ├── RoutesTab.tsx               # Routes table view (NEW)
│   ├── ApiKeysTab.tsx              # API Keys table view (NEW)
│   ├── RoutesTable.tsx             # Routes data table (NEW)
│   ├── ApiKeysTable.tsx            # API Keys data table (NEW)
│   ├── CreateRouteModal.tsx        # Modal to create route (REWRITE)
│   ├── UpdateRouteModal.tsx        # Modal to update route config (NEW)
│   ├── DeleteRouteModal.tsx        # Confirmation with binding warnings (NEW)
│   ├── CreateApiKeyModal.tsx       # Modal to create standalone key (NEW)
│   ├── UpdateApiKeyModal.tsx       # Key rotation modal (EXISTS)
│   ├── DeleteApiKeyModal.tsx       # Confirmation with routes affected (NEW)
│   ├── RouteDetailDrawer.tsx       # Show route details + bindings (NEW)
│   ├── ApiKeyDetailDrawer.tsx      # Show key details + routes using (NEW)
│   ├── BindingsTable.tsx           # Table showing bindings (EXISTS)
│   └── ProviderConfigForm.tsx      # Provider-specific auth fields (NEW)
├── hooks/
│   ├── useListRoutes.ts            # List routes (NEW)
│   ├── useCreateRoute.ts           # Create route mutation (NEW)
│   ├── useUpdateRoute.ts           # Update route mutation (NEW)
│   ├── useDeleteRoute.ts           # Delete route mutation (NEW)
│   ├── useListApiKeys.ts           # List API keys (RENAME from useListSecrets)
│   ├── useCreateApiKey.ts          # Create API key mutation (NEW)
│   ├── useUpdateApiKey.ts          # Rotate API key mutation (NEW)
│   ├── useDeleteApiKey.ts          # Delete API key mutation (NEW)
│   ├── useListBindings.ts          # List bindings (EXISTS)
│   ├── useBindRoute.ts             # Bind route mutation (NEW)
│   └── useUnbindRoute.ts           # Unbind route mutation (EXISTS as useUnbindSecretMutation)
├── api/
│   ├── routesApi.ts                # Routes CRUD operations (NEW)
│   ├── apiKeysApi.ts               # API Keys CRUD (RENAME from secretsApi.ts)
│   └── bindingsApi.ts              # Bindings operations (NEW)
├── types.ts                        # TypeScript interfaces (UPDATE)
├── constants.ts                    # Query keys, etc. (UPDATE)
└── providers/                      # Provider configs (NEW)
    └── providerConfigs.ts          # Provider schemas
```

---

## Implementation Tasks

### Phase 1: Fix Backend Response Issue (🚧 IN PROGRESS)

**Issue**: Both `/ajax-api/3.0/mlflow/secrets/list` and `/api/3.0/mlflow/secrets/list-routes` return `{}` instead of `{"secrets": []}` or `{"routes": []}`.

**Tasks**:
- [ ] Debug why `_list_secrets()` handler returns empty response
- [ ] Check if handler is missing `@catch_mlflow_exception` decorator
- [ ] Verify protobuf serialization is working correctly
- [ ] Test that empty list returns `{"secrets": []}` not `{}`
- [ ] Fix `_list_secret_routes()` handler similarly

**Acceptance Criteria**:
- `curl http://localhost:5000/ajax-api/3.0/mlflow/secrets/list` returns `{"secrets": []}`
- `curl http://localhost:5000/api/3.0/mlflow/secrets/list-routes` returns `{"routes": []}`

---

### Phase 2: TypeScript Types & API Client

**Tasks**:
- [ ] Update `types.ts` with Route and provider types
- [ ] Create `routesApi.ts` with CRUD operations
- [ ] Rename `secretsApi.ts` to `apiKeysApi.ts`
- [ ] Create `bindingsApi.ts` for binding operations
- [ ] Add provider config types from AI_GATEWAY_ARCHITECTURE.md

**Files to Update**:
```typescript
// types.ts
export interface Route {
  route_id: string;
  secret_id: string;
  model_name: string;
  name?: string;
  provider: string;
  created_at: number;
  last_updated_at: number;
  bindings_count?: number;
}

export interface ApiKey {
  secret_id: string;
  provider: string;
  is_shared: boolean;
  created_at: number;
  last_updated_at: number;
  routes_count?: number;
}

export interface Provider {
  id: string;
  displayName: string;
  authType: 'simple' | 'complex';
  models: { id: string; displayName: string }[];
}
```

---

### Phase 3: Create Main Page with Tabs

**Tasks**:
- [ ] Create `AIGatewayPage.tsx` with tab navigation
- [ ] Create `RoutesTab.tsx` component
- [ ] Create `ApiKeysTab.tsx` component
- [ ] Update routing to use new page
- [ ] Add "AI Gateway" to sidebar navigation

**Component Structure**:
```tsx
// AIGatewayPage.tsx
<Page>
  <Header title="AI Gateway" />
  <Tabs>
    <Tab label="Routes">
      <RoutesTab />
    </Tab>
    <Tab label="API Keys">
      <ApiKeysTab />
    </Tab>
  </Tabs>
</Page>
```

---

### Phase 4: Routes Table & CRUD

**Tasks**:
- [ ] Create `RoutesTable.tsx` with columns: Name, Model, Provider, Bindings
- [ ] Create `useListRoutes` hook
- [ ] Create `CreateRouteModal.tsx` with provider selection
- [ ] Create `UpdateRouteModal.tsx` for editing route config
- [ ] Create `DeleteRouteModal.tsx` with cascade warnings
- [ ] Create `RouteDetailDrawer.tsx` showing bindings
- [ ] Add search/filter functionality

**Table Columns**:
- **Name**: Route name (clickable)
- **Model**: Model ID (e.g., "gpt-4o", "claude-sonnet-4-5")
- **Provider**: Provider badge (OpenAI, Anthropic, etc.)
- **Bindings**: Count (e.g., "3 resources")
- **Actions**: Edit, Clone, Delete icons

---

### Phase 5: API Keys Table & CRUD

**Tasks**:
- [ ] Create `ApiKeysTable.tsx` with columns: Provider, Created, Routes Using
- [ ] Rename `useListSecrets` to `useListApiKeys`
- [ ] Create `CreateApiKeyModal.tsx` for standalone keys
- [ ] Create `UpdateApiKeyModal.tsx` for key rotation (already exists)
- [ ] Create `DeleteApiKeyModal.tsx` with affected routes warning
- [ ] Create `ApiKeyDetailDrawer.tsx` showing routes using this key

**Table Columns**:
- **Provider**: Provider name with icon
- **Created**: Timestamp
- **Routes Using**: Count of routes using this key
- **Actions**: Rotate, Delete icons

---

### Phase 6: Provider Configuration System

**Tasks**:
- [ ] Create `providerConfigs.ts` with provider schemas
- [ ] Create `ProviderConfigForm.tsx` for dynamic auth forms
- [ ] Add provider-specific validation
- [ ] Support simple providers (OpenAI, Anthropic)
- [ ] Support complex providers (AWS Bedrock, Azure, Vertex AI)

**Provider Config Example**:
```typescript
{
  provider: 'openai',
  displayName: 'OpenAI',
  authType: 'simple',
  models: [
    { id: 'gpt-4o', displayName: 'GPT-4o' },
    { id: 'gpt-4o-mini', displayName: 'GPT-4o Mini' },
  ],
}
```

---

### Phase 7: Bindings Management

**Tasks**:
- [ ] Update `BindingsTable.tsx` to work with routes
- [ ] Create `useBindRoute` mutation hook
- [ ] Update `useUnbindRoute` mutation hook
- [ ] Add "Bind to Resource" modal
- [ ] Show bindings in RouteDetailDrawer

---

### Phase 8: Testing & Polish

**Tasks**:
- [ ] Write unit tests for new components
- [ ] Write integration tests for routes CRUD
- [ ] Test provider config forms
- [ ] Add loading states
- [ ] Add error handling
- [ ] Add empty states
- [ ] Run `yarn check-all` for linting and type checking

---

## Migration Notes

### Files to Delete (OLD Architecture)
- `SecretsPage.tsx` - Replaced by `AIGatewayPage.tsx`
- `SecretsTable.tsx` - Replaced by `RoutesTable.tsx` + `ApiKeysTable.tsx`
- `CreateSecretModal.tsx` - Replaced by `CreateRouteModal.tsx`
- `UpdateSecretModal.tsx` - Replaced by `UpdateRouteModal.tsx`
- `DeleteSecretModal.tsx` - Replaced by `DeleteRouteModal.tsx`
- `SecretDetailDrawer.tsx` - Replaced by `RouteDetailDrawer.tsx`

### Files to Rename
- `secretsApi.ts` → `apiKeysApi.ts`
- `useListSecrets.ts` → `useListApiKeys.ts`
- `useCreateSecretMutation.ts` → `useCreateApiKeyMutation.ts`
- `useUpdateSecretMutation.ts` → `useUpdateApiKeyMutation.ts`
- `useDeleteSecretMutation.ts` → `useDeleteApiKeyMutation.ts`

### Files to Keep
- `BindingsTable.tsx` - Still used for showing bindings
- `useListBindings.ts` - Still used
- `useUnbindSecretMutation.ts` - Rename to `useUnbindRouteMutation.ts`

---

## Testing Strategy

### Backend Testing
- [ ] Test list routes returns empty array, not empty object
- [ ] Test list API keys returns empty array, not empty object
- [ ] Test create route with new API key
- [ ] Test create route with existing API key
- [ ] Test update route config
- [ ] Test delete route cascades to bindings
- [ ] Test API key rotation affects all routes
- [ ] Test delete API key cascades to routes and bindings

### Frontend Testing
- [ ] Test routes table displays correctly
- [ ] Test API keys table displays correctly
- [ ] Test create route modal with provider selection
- [ ] Test update route modal
- [ ] Test delete route confirmation
- [ ] Test API key rotation modal shows affected routes
- [ ] Test bindings table in route detail drawer
- [ ] Test provider config forms for each provider type

---

## Success Criteria

- [ ] Main page shows "AI Gateway" with two tabs
- [ ] Routes tab shows routes table with correct columns
- [ ] API Keys tab shows API keys table with routes count
- [ ] Create Route button opens modal with provider selection
- [ ] Can create route with new API key
- [ ] Can create route with existing API key
- [ ] Can update route configuration
- [ ] Can delete route with binding warnings
- [ ] Can rotate API key with affected routes warning
- [ ] Can view bindings for a route
- [ ] All TypeScript compilation passes
- [ ] All linting passes
- [ ] No console errors in browser

---

## Timeline Estimate

- **Phase 1**: Fix backend responses - 1-2 hours
- **Phase 2**: Types & API client - 2-3 hours
- **Phase 3**: Main page with tabs - 2-3 hours
- **Phase 4**: Routes table & CRUD - 4-5 hours
- **Phase 5**: API Keys table & CRUD - 3-4 hours
- **Phase 6**: Provider config system - 3-4 hours
- **Phase 7**: Bindings management - 2-3 hours
- **Phase 8**: Testing & polish - 3-4 hours

**Total**: 20-28 hours (~3-4 days)

---

## Current Session Progress

### Completed
- ✅ Read AI_GATEWAY_ARCHITECTURE.md to understand route-centric design
- ✅ Identified that current UI is OLD secret-centric architecture
- ✅ Found backend routes APIs are already implemented
- ✅ Created this development plan document

### In Progress
- 🚧 Debugging backend empty response issue

### Next Steps
1. Fix backend `_list_secrets()` handler to return proper response format
2. Fix backend `_list_secret_routes()` handler similarly
3. Start implementing TypeScript types for routes and providers
4. Create AIGatewayPage with tab navigation

---

## Reference Documents

- `AI_GATEWAY_ARCHITECTURE.md` - Route-centric architecture design
- `SECRETS_UI_DEVELOPMENT_PLAN.md` - OLD secret-centric design (deprecated)
- `SECRETS_FEATURE_DEV_GUIDE.md` - Backend implementation guide

---

**Last Updated**: 2025-01-13
**Implemented By**: Claude Code
**Status**: Phase 1 in progress
