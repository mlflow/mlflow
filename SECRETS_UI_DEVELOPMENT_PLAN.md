# MLflow Secrets Management UI - Development Plan

## Overview

This document outlines the design and implementation plan for the MLflow Secrets Management UI. The UI will provide a user-friendly interface for managing secrets (API keys, credentials) and their bindings to MLflow resources.

## Architecture Summary

### Core Technologies

- **Design System**: `@databricks/design-system` (primary component library)
- **UI Framework**: React 18 with TypeScript
- **Routing**: React Router v6 with hash routing
- **State Management**: Redux + React Query (@tanstack/react-query v4)
- **Data Tables**: AG-Grid for complex tables
- **Forms**: React Hook Form
- **Styling**: Emotion CSS-in-JS with theme system

### Key Design Patterns

#### 1. Modal Pattern

Location: `mlflow/server/js/src/experiment-tracking/components/modals/`

- Uses `GenericInputModal` wrapper component
- Form validation via React refs
- Async submission with loading states
- Example: `CreateExperimentModal.tsx`

#### 2. API Integration

Location: `mlflow/server/js/src/shared/web-shared/genai-traces-table/hooks/useMlflowTraces.tsx:496`

- React Query for all data fetching
- Query keys for cache invalidation
- Network + client-side filtering patterns
- Mutation hooks for updates

#### 3. Routing

Location: `mlflow/server/js/src/MlflowRouter.tsx:36`

- Route definitions in separate `route-defs.ts` files
- Lazy loading with code splitting
- Nested routes for tabbed interfaces

#### 4. Sidebar Navigation

Location: `mlflow/server/js/src/common/components/MlflowSidebar.tsx:33`

- Top-level nav items with icons
- Dropdown for "New" actions
- Active state indication

## Backend Secrets API

### Available Endpoints

Based on `mlflow/server/handlers.py:4301`:

- ✅ `CreateAndBindSecret` - Create secret + initial binding atomically
- ✅ `GetSecretInfo` - Get secret metadata by ID
- ✅ `UpdateSecret` - Update secret value
- ✅ `DeleteSecret` - Delete secret
- ✅ `BindSecret` - Bind existing secret to resource
- ✅ `UnbindSecret` - Remove binding
- ✅ `ListSecretBindings` - List bindings (with filters)
- ✅ `ListSecrets` - List all secrets user can access

### Data Models

Based on `mlflow/entities/secret.py:47`

```typescript
interface Secret {
  secret_id: string;
  secret_name: string;
  masked_value: string; // e.g., "sk-...xyz123"
  is_shared: boolean; // Can bind to multiple resources
  created_at: number; // Unix timestamp (ms)
  last_updated_at: number;
  created_by?: string;
  last_updated_by?: string;
}

interface SecretBinding {
  binding_id: string;
  secret_id: string;
  resource_type: string; // "SCORER_JOB" | "GLOBAL"
  resource_id: string;
  field_name: string; // Environment variable name
  created_at: number;
  last_updated_at: number;
  created_by?: string;
  last_updated_by?: string;
}
```

## Development Phases

### Phase 1: Frontend Foundation

#### 1.1 Create Secrets Route Structure

Create new directory: `mlflow/server/js/src/secrets/`

```
mlflow/server/js/src/secrets/
├── routes.ts                    # Route path constants
├── route-defs.ts                # Route definitions
├── components/
│   ├── SecretsPage.tsx         # Main secrets list page
│   ├── SecretsTable.tsx        # Table component
│   ├── SecretDetailPane.tsx    # Right pane for bindings
│   └── modals/
│       ├── CreateSecretModal.tsx
│       ├── UpdateSecretModal.tsx
│       ├── DeleteSecretModal.tsx
│       └── RebindSecretModal.tsx
├── hooks/
│   ├── useSecretsQuery.ts      # React Query for list
│   ├── useSecretMutations.ts   # CRUD mutations
│   └── useSecretBindings.ts    # Bindings query
└── types.ts                     # TypeScript interfaces
```

#### 1.2 Add Route to Router

- Add "Secrets" to `MlflowSidebar.tsx` navigation (`mlflow/server/js/src/common/components/MlflowSidebar.tsx:46`)
- Add route to `MlflowRouter.tsx` (`mlflow/server/js/src/MlflowRouter.tsx:108`)
- Use icon: Consider `KeyIcon` or `LockIcon` from design system

#### 1.3 Create API Client Layer

```typescript
// mlflow/server/js/src/secrets/api/secretsApi.ts
export const secretsApi = {
  listSecrets: async (): Promise<Secret[]> => {
    const response = await fetch("/ajax-api/2.0/mlflow/secrets/list", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    return response.json();
  },

  createSecret: async (payload: CreateSecretPayload) => {
    /* ... */
  },
  updateSecret: async (secretId: string, value: string) => {
    /* ... */
  },
  deleteSecret: async (secretId: string) => {
    /* ... */
  },
  listBindings: async (secretId?: string) => {
    /* ... */
  },
  bindSecret: async (payload: BindSecretPayload) => {
    /* ... */
  },
  unbindSecret: async (payload: UnbindSecretPayload) => {
    /* ... */
  },
};
```

### Phase 2: Core UI Components

#### 2.1 SecretsPage (Main View)

Following pattern from `ExperimentListView.tsx`:

- Split-pane layout: Table (left) + Detail pane (right)
- Search bar for filtering secrets by name
- "New Secret" button (top-right)
- Empty state when no secrets

#### 2.2 SecretsTable Component

Use AG-Grid (like other MLflow tables):

**Columns:**

- Secret Name (clickable, shows in detail pane)
- Masked Value (e.g., `sk-...xyz123`)
- Type (badge: "Shared" / "Private")
- Bindings Count (e.g., "3 resources")
- Created By
- Last Updated
- Actions (Update, Delete)

#### 2.3 SecretDetailPane Component

Appears when secret selected:

- Shows full secret metadata
- **Bindings List:**
  - Resource Type + Resource ID
  - Field Name (env var)
  - Created date
  - "Unbind" action button
- "Bind to Resource" button
- "Update Secret" button

### Phase 3: Modal Components

#### 3.1 CreateSecretModal 🔑 (Most Complex)

Following the Langfuse pattern:

```tsx
<Modal title="Create Secret" visible={isOpen} onOk={handleCreate}>
  <Form>
    {/* Secret Name */}
    <Input label="Secret Name" placeholder="my-openai-key" required />

    {/* Secret Value with Masking */}
    <div>
      <label>Secret Value</label>
      <div style={{ position: "relative" }}>
        <Input
          type={showSecret ? "text" : "password"}
          value={secretValue}
          onChange={(e) => setSecretValue(e.target.value)}
          placeholder="sk-..."
          required
          // IMPORTANT: autoComplete="off" to prevent browser saving
          autoComplete="new-password"
        />
        <IconButton
          icon={showSecret ? <EyeOffIcon /> : <EyeIcon />}
          onClick={() => setShowSecret(!showSecret)}
          style={{ position: "absolute", right: 8, top: 8 }}
          aria-label={showSecret ? "Hide secret" : "Show secret"}
        />
      </div>
      {/* Show last 2 chars when masked */}
      {!showSecret && secretValue && (
        <Typography.Text type="secondary">Last 2 chars: ...{secretValue.slice(-2)}</Typography.Text>
      )}
    </div>

    {/* Shared Toggle */}
    <Switch
      label="Shared Secret"
      description="Allow this secret to be reused across multiple resources"
      checked={isShared}
      onChange={setIsShared}
    />

    {/* Initial Binding (Required) */}
    <Divider />
    <Typography.Title level={5}>Initial Binding</Typography.Title>
    <Select
      label="Resource Type"
      options={[
        { label: "Scorer Job", value: "SCORER_JOB" },
        { label: "Global", value: "GLOBAL" },
      ]}
      required
    />
    <Input label="Resource ID" required />
    <Input label="Field Name" placeholder="OPENAI_API_KEY" required />
  </Form>
</Modal>
```

**Key Security Considerations:**

- ⚠️ Use `autoComplete="new-password"` to prevent browser/extension autofill
- ⚠️ Clear `secretValue` state on modal close
- ⚠️ Never log secret values (not even masked)
- ⚠️ Use `type="password"` by default with toggle
- ✅ Show only last 2 chars when masked

#### 3.2 UpdateSecretModal

Similar to create, but:

- Pre-fill current masked value as placeholder
- Show list of ALL resources using this secret
- Confirmation dialog: "This will update the secret for X resources: [list]. Are you sure?"
- Secondary confirmation with dangerous action styling

#### 3.3 DeleteSecretModal

- Show bindings that will be orphaned
- Warning: "This will remove the secret from X resources"
- Type secret name to confirm (dangerous action pattern)

#### 3.4 RebindSecretModal (For bulk rebinding)

- Multi-select resources (checkboxes)
- Select new secret from dropdown
- Preview changes: "6 resources will switch from 'dev-key' to 'prod-key'"

### Phase 4: React Query Integration

#### 4.1 Queries

```typescript
// hooks/useSecretsQuery.ts
export const useSecretsQuery = () => {
  return useQuery({
    queryKey: ["secrets", "list"],
    queryFn: secretsApi.listSecrets,
    staleTime: 30000, // 30s
  });
};

export const useSecretBindingsQuery = (secretId: string) => {
  return useQuery({
    queryKey: ["secrets", "bindings", secretId],
    queryFn: () => secretsApi.listBindings(secretId),
    enabled: !!secretId,
  });
};
```

#### 4.2 Mutations

```typescript
// hooks/useSecretMutations.ts
export const useCreateSecretMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: secretsApi.createSecret,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["secrets", "list"] });
    },
  });
};

// Similar for update, delete, bind, unbind
```

### Phase 5: State Management & UX Polish

#### 5.1 Selection State

- Track selected secret in URL query param
- Highlight selected row in table
- Show detail pane when selected

#### 5.2 Loading States

- Skeleton loaders while fetching (use `LegacySkeleton` from design system)
- Disable buttons during mutations
- Optimistic updates where appropriate

#### 5.3 Error Handling

- Toast notifications for errors (use design system `Notification`)
- Form validation with helpful messages
- Network error retry logic

#### 5.4 Empty States

- "No secrets yet" with "Create Secret" CTA
- "No bindings" in detail pane

## Design System Components

Based on `mlflow/server/js/package.json:31` and existing patterns:

```typescript
import {
  Modal,
  Button,
  Input,
  Select,
  Switch,
  Table,
  Badge,
  Notification,
  Typography,
  Divider,
  IconButton,
  LegacySkeleton,
  useDesignSystemTheme,
  // Icons
  KeyIcon,
  EyeIcon,
  EyeOffIcon,
  DeleteIcon,
  EditIcon,
  PlusIcon,
} from "@databricks/design-system";
```

### Theme Usage (REQUIRED)

```typescript
const { theme } = useDesignSystemTheme();

// ✅ Use theme.spacing instead of hardcoded pixels
<div
  style={{
    padding: theme.spacing.md,
    gap: theme.spacing.sm,
    marginTop: theme.spacing.lg,
  }}
/>;
```

## Security Best Practices

1. **Never expose full secret values in frontend** - Backend already enforces this with `masked_value`
2. **No logging** - Never log secret values, even partially
3. **Clear state** - Wipe secret input values from memory on modal close
4. **Prevent autofill**: `autoComplete="new-password"` on secret inputs
5. **Confirmation dialogs** - For destructive actions (update, delete)
6. **Audit trail** - Display `created_by`, `last_updated_by` fields

## Implementation Checklist

### Frontend

- [ ] Create `mlflow/server/js/src/secrets/` directory structure
- [ ] Add routes (`routes.ts`, `route-defs.ts`)
- [ ] Create `SecretsPage.tsx`
- [ ] Create `SecretsTable.tsx`
- [ ] Create `SecretDetailPane.tsx`
- [ ] Create modal components (Create, Update, Delete, Rebind)
- [ ] Create API client (`api/secretsApi.ts`)
- [ ] Create React Query hooks
- [ ] Add to `MlflowSidebar.tsx` navigation
- [ ] Add to `MlflowRouter.tsx`
- [ ] Write unit tests

## Implementation Order

1. **Route skeleton**: Basic page with "Coming soon" message
2. **List view**: Table showing secrets (read-only)
3. **Detail pane**: Show bindings for selected secret
4. **Create modal**: With masked input (most complex)
5. **Update modal**: With confirmation for multiple resources
6. **Delete modal**: With safeguards
7. **Rebind functionality**: Bulk operations
8. **Polish**: Loading states, error handling, empty states
9. **Testing**: Unit + integration tests

## Next Steps

Current focus areas:

1. Set up basic route structure
2. Implement the secrets list view
3. Build the create secret modal with proper secret masking
4. Add binding management functionality

## Notes

- All secrets API endpoints use `/ajax-api/2.0/mlflow/secrets/` or `/api/3.0/mlflow/secrets/` prefix
- The `ListSecrets` endpoint is now available (added in auth branch)
- Follow existing MLflow patterns for consistency
- Reference Langfuse UI for inspiration on secret masking UX
