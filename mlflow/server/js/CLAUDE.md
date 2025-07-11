# MLflow Web Development Context

## Development Server

### Starting the Dev Server
**IMPORTANT**: The dev server must always be run from the root of universe, not from mlflow/web/js.

```bash
# Start the dev server in the background with logs written
nohup yarn start --projects mlflow > /tmp/mlflow_dev_server.log 2>&1 &

# Start with a different proxy target (e.g., different workspace)
nohup yarn start --projects mlflow --proxy https://eng-ml-inference-team-us-west-2.cloud.databricks.com > /tmp/mlflow_dev_server.log 2>&1 &
```

### Proxy Configuration
When using the optional `--proxy` flag, ensure the URL format is:
- **Format**: `https://workspace-name.cloud.databricks.com` (https, no trailing slash)
- **Example**: `https://eng-ml-inference-team-us-west-2.cloud.databricks.com`

### Monitoring Dev Server
```bash
# Check server logs
tail -f /tmp/mlflow_dev_server.log

# Check if server is running
ps aux | grep "yarn start --projects mlflow"
```

### Stopping Dev Server
```bash
# Kill only the mlflow dev server (avoids killing other dev processes)
pkill -f "yarn start --projects mlflow"

# Verify no mlflow dev server is running
ps aux | grep "yarn start --projects mlflow" | grep -v grep
```

### Dev Server Status Indicators
- **Ready**: Look for `dev-proxy for monolith,url-proxy started at [URL] 🚀`
- **Starting**: `Waiting for monolith...` logs indicate startup in progress  
- **Startup time**: Can take up to ~2 minutes

### Common Issues

#### Certificate Issues
If the dev server fails with certificate installation errors, ask the user to run `yarn start --projects mlflow` once outside of Claude to handle the interactive sudo prompt, then they can restart with nohup.

## Playwright UI Automation

### Setup Browser Automation

**If Playwright MCP tools are not available:** Run the one-command setup:
```bash
# From mlflow/web/js directory
yarn playwright-jsessionid-cookie refresh
```
Then **restart Claude Code**.

### Available Commands
```bash
# One-command setup/refresh (automatically handles everything)
yarn playwright-jsessionid-cookie refresh

# Check current setup status
yarn playwright-jsessionid-cookie status
```

This automatically:
- Adds Playwright MCP server to Claude
- Extracts your Chrome session (JSESSIONID)
- Configures Playwright with authentication
- Enables Claude to interact with MLflow UI

### Requirements
- Login to https://dev.local:22090 in Chrome first
- MLflow dev server must be running
- Restart Claude Code after setup

### Troubleshooting

#### Login Screen Appears in Playwright
If you see a login screen when using Playwright automation, the JSESSIONID cookie has expired or you're not logged in:

1. **Login in Chrome first:**
   ```bash
   open https://dev.local:22090/?o=6051921418418893
   ```
   Complete the login/signup process in Chrome

2. **Refresh/setup:**
   ```bash
   yarn playwright-jsessionid-cookie refresh
   ```

3. **Restart Claude Code** for the new session to take effect

4. **Try again** - Playwright should now bypass the login screen

## Development Commands (from mlflow/web/js)

### Testing & Quality
```bash
# Test code (supports subfolders/individual files via Jest)
yarn test

# TypeScript type-checking (can be very slow)
yarn type-check

# Lint and fix issues (fix remaining issues manually)
yarn lint:fix

# Extract internationalization messages (run after changing FormattedMessages/intl.formatMessage)
yarn i18n:extract

# Remove unused exports
yarn knip

# Generate code for safex flags (run after changing safex flags)
yarn gocx-codegen

# Fix all issues (much slower comprehensive script)
yarn fix-all
```

### Before Committing
**IMPORTANT**: Always run `yarn fix-all` and fix any remaining issues before committing or sending a pull request.

## UI Components and Design System
- The Dubois design system components are located at: `@databricks/design-system`
- Implementation source code for design system components can be found at: `design-system/src/design-system`
- All UI components used in this directory should import from `@databricks/design-system`
- Theme structure can be found at: `design-system/src/theme/index.ts`
  - Colors: `theme.colors` (e.g., `textValidationDanger` for error text)
  - Typography: `theme.typography` (e.g., `fontSizeSm`, `fontSizeBase`)
  - Spacing: `theme.spacing` (e.g., `sm`, `md`, `lg`)

### Available Design System Components
When adding UI components, first check `js/packages/du-bois/src/primitives` to see if there's a matching component available. Note that component names may not exactly match what you're looking for (e.g., a "dropdown" might be called `Select`, `DialogCombobox`, or `DropdownMenu` depending on the use case). If there are multiple matches, ask the user which they mean.

Each component folder contains:
- `.mdx` files with documentation and best practices
- `.stories.tsx` files with usage examples
- `stories/` folder with detailed implementation examples

You can view component documentation in the Dubois docs using this URL pattern:
`https://ui-infra.dev.databricks.com/storybook/js/packages/du-bois/index.html?path=/docs/primitives-<component-name>--docs`

For example, to see the Accordion component: https://ui-infra.dev.databricks.com/storybook/js/packages/du-bois/index.html?path=/docs/primitives-accordion--docs

## Project Structure
- **Location**: `/mlflow/web/js/`
- **Dev Proxy**: Uses `@databricks/dev-proxy` package
- **Package Manager**: Yarn
- **Playwright**: UI automation via `scripts/playwright-jsessionid-cookie.js`