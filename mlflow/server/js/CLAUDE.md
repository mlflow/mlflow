# CLAUDE.md - MLflow Frontend Development

This file provides guidance to Claude Code when working with the MLflow frontend code in this directory.

**For contribution guidelines, code standards, and additional development information not covered here, please refer to [CONTRIBUTING.md](../../../CONTRIBUTING.md).**

## Consistency is Critical

**IMPORTANT**: Always be consistent with the rest of the repository. This is extremely important!

Before implementing any feature:
1. Read through similar files to understand their structure and patterns
2. Do NOT invent new components if they already exist
3. Use existing patterns and conventions found in the codebase
4. Check for similar functionality that already exists

## Development Server

**IMPORTANT**: Always start the development server from the repository root for the best development experience with hot reload:

```bash
# MUST be run from the repository root
nohup uv run bash dev/run-dev-server.sh > /tmp/mlflow-dev-server.log 2>&1 &

# Monitor the logs
tail -f /tmp/mlflow-dev-server.log

# Servers will be available at:
# - MLflow backend: http://localhost:5000
# - React frontend: http://localhost:3000 (with hot reload)
```

This provides fast edit-refresh for UI development - changes to React components will automatically reload in the browser.

## Available Yarn Scripts

When running from the repository root, use this pattern:

```bash
# Example: Run any yarn command from root
pushd mlflow/server/js && yarn <command>; popd
```

Available scripts:

```bash
# Development
yarn start              # Start dev server (port 3000) with hot reload
yarn build              # Build production bundle

# Testing
yarn test               # Run Jest tests
yarn test:watch         # Run tests in watch mode
yarn test:ci            # Run tests with coverage for CI

# Code Quality
yarn lint               # Run ESLint
yarn lint:fix           # Run ESLint with auto-fix
yarn prettier:check     # Check Prettier formatting
yarn prettier:fix       # Fix Prettier formatting
yarn type-check         # Run TypeScript type checking

# Combined Checks
yarn check-all          # Run all checks (lint, prettier, i18n, type-check)

# Other Commands
yarn storybook          # Start Storybook for component development
yarn build-storybook    # Build static Storybook
yarn i18n:check         # Check i18n translations
```

### Before Committing

**IMPORTANT**: Always run these checks and fix any remaining issues before committing:

```bash
# From repository root
pushd mlflow/server/js && yarn check-all; popd

# Fix any issues that are reported
```

## UI Components and Design System

### Use Databricks Design System Components

**Always use components from `@databricks/design-system` when available.** Do not create custom components if they already exist in the design system.

Common components include:

- `Button`, `IconButton` - for actions
- `Input`, `Textarea`, `Select` - for form inputs  
- `Modal`, `Drawer` - for overlays
- `Table`, `TableRow`, `TableCell` - for data tables
- `Tabs`, `TabPane` - for tabbed interfaces
- `Alert`, `Notification` - for feedback
- `Spinner`, `Skeleton` - for loading states
- `Tooltip`, `Popover` - for additional information
- `Card` - for content containers
- `Typography` - for text styling

Example import:

```typescript
import { Button, Modal, Input } from '@databricks/design-system';
```

### Theme Usage

Use the design system theme for consistent styling:

```typescript
import { useDesignSystemTheme } from '@databricks/design-system';

const Component = () => {
  const { theme } = useDesignSystemTheme();
  
  return (
    <div style={{ 
      color: theme.colors.textPrimary,
      padding: theme.spacing.md,
      fontSize: theme.typography.fontSizeBase
    }}>
      Content
    </div>
  );
};
```

### Spacing Guidelines

**ALWAYS use `theme.spacing` values instead of hard-coded pixel widths.** This ensures consistency and maintainability across the application.

```typescript
// ✅ GOOD - Use theme spacing
<div style={{ 
  padding: theme.spacing.md,
  marginBottom: theme.spacing.lg,
  gap: theme.spacing.sm 
}} />

// ❌ BAD - Avoid hard-coded pixels
<div style={{ 
  padding: '16px',
  marginBottom: '24px',
  gap: '8px'
}} />
```

Common spacing values:
- `theme.spacing.xs` - Extra small spacing (4px)
- `theme.spacing.sm` - Small spacing (8px)
- `theme.spacing.md` - Medium spacing (16px)
- `theme.spacing.lg` - Large spacing (24px)
- `theme.spacing.xl` - Extra large spacing (32px)

For custom spacing needs, use the spacing function:
```typescript
// When you need a specific multiple of the base unit
padding: theme.spacing(2.5) // 20px (2.5 * 8px base unit)
```

### Finding the Right Component

When looking for a component:

1. First check `@databricks/design-system` imports in existing code
2. Component names may not be exact (e.g., "dropdown" could be `Select`, `DialogCombobox`, or `DropdownMenu`)
3. Look at similar UI patterns in the codebase for examples
4. If multiple matches exist, choose based on the use case

### Discovering Available Components Dynamically

To see ALL components available in the design system:

```bash
# From mlflow/server/js directory, check what's exported
cat node_modules/@databricks/design-system/dist/design-system/index.d.ts

# This file lists every component as: export * from './ComponentName';
# Each line represents a component you can import
```

This is the definitive source for available components - more reliable than checking folders since it shows only what's publicly exported.

### Viewing Component Documentation in Storybook

You can use Playwright to view the component documentation and examples in Storybook:

```
https://ui-infra.dev.databricks.com/storybook/js/packages/du-bois/index.html?path=/docs/primitives-<component-name>--docs
```

For example:
- Alert: `https://ui-infra.dev.databricks.com/storybook/js/packages/du-bois/index.html?path=/docs/primitives-alert--docs`
- Button: `https://ui-infra.dev.databricks.com/storybook/js/packages/du-bois/index.html?path=/docs/primitives-button--docs`
- Modal: `https://ui-infra.dev.databricks.com/storybook/js/packages/du-bois/index.html?path=/docs/primitives-modal--docs`

Use Playwright MCP to navigate to these URLs and see live examples, props documentation, and usage patterns.

## Browser Testing with Playwright

For testing UI changes in a real browser, Claude Code can use the Playwright MCP (Model Context Protocol) integration.

### Checking Playwright MCP Status

To check if Playwright MCP is available:

- Look for browser testing tools in available MCP functions
- Try using browser navigation or screenshot capabilities

### Installing Playwright MCP

If Playwright MCP is not available and you need to test UI changes, you can install it:

```bash
claude mcp add playwright npx '@playwright/mcp@latest'
```

**Note**: After installation, you must restart Claude Code for the integration to be available.

### Using Playwright MCP

Once installed, you can:

- Navigate to the development server
- Take screenshots of UI components
- Interact with forms and buttons
- Verify UI changes are working correctly

Example workflow:

1. Make changes to React components
2. Wait for hot reload (automatic)
3. Use Playwright to navigate to `http://localhost:3000`
4. Take screenshots or interact with the updated UI
5. Verify the changes work as expected

## Project Structure

```text
mlflow/server/js/
├── src/
│   ├── experiment-tracking/    # Experiment tracking UI
│   ├── model-registry/         # Model registry UI  
│   ├── common/                 # Shared components
│   ├── shared/                 # Shared utilities
│   └── app.tsx                # Main app entry point
├── vendor/                     # Third-party dependencies
├── package.json               # Dependencies and scripts
├── tsconfig.json              # TypeScript configuration
├── webpack.config.js          # Webpack bundler config
└── jest.config.js             # Jest test configuration
```

## Key Technologies

- **React 18**: UI framework
- **TypeScript**: Type safety
- **Redux**: State management
- **Apollo Client**: GraphQL client
- **Ant Design (antd)**: UI component library
- **AG-Grid**: Data table component
- **Jest**: Testing framework
- **React Testing Library**: Component testing
- **Webpack**: Module bundler

## Common Tasks

### Adding a New Component

1. Create component file in appropriate directory
2. Add TypeScript types/interfaces
3. Write component with hooks (functional components preferred)
4. Add unit tests in same directory with `.test.tsx` extension
5. Add to Storybook if it's a reusable component

### Updating GraphQL Queries

1. Modify query in relevant `.graphql` file
2. Run codegen to update TypeScript types (if configured)
3. Update components using the query

### Testing Components

```bash
# Run tests for a specific component
yarn test ComponentName

# Run tests in watch mode for development
yarn test --watch

# Update snapshots if needed
yarn test -u
```

### Debugging

1. Use React Developer Tools browser extension
2. Redux DevTools for state debugging
3. Browser console for network requests
4. Source maps are enabled in development mode

## Code Style

- Use functional components with hooks
- Prefer TypeScript strict mode
- Follow existing patterns in the codebase
- Use meaningful component and variable names
- Add JSDoc comments for complex logic
- Keep components small and focused

## Best Practices

### Data Fetching

**Use React Query** for all API calls and data fetching:

```typescript
// Good: Using React Query
const { data, isLoading, error } = useQuery({
  queryKey: ['experiments', experimentId],
  queryFn: () => fetchExperiment(experimentId),
});

// Avoid: Manual fetch in useEffect
// useEffect(() => { fetch(...) }, [])
```

### State Management

**Avoid useEffect** when possible. Prefer deriving state with `useMemo`:

```typescript
// Good: Derive state with useMemo
const filteredRuns = useMemo(() => {
  return runs.filter(run => run.status === 'active');
}, [runs]);

// Avoid: useEffect to update state
// useEffect(() => {
//   setFilteredRuns(runs.filter(run => run.status === 'active'));
// }, [runs]);
```

Use `useEffect` only for:

- Side effects (DOM manipulation, subscriptions)
- Synchronizing with external systems
- Cleanup operations

## Performance Considerations

- Use React.memo for expensive components
- Implement virtualization for large lists (AG-Grid handles this)
- Lazy load routes and heavy components
