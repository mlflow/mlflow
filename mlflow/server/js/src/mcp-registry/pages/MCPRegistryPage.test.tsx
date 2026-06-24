import { describe, it, expect } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import { setupServer } from '../../common/utils/setup-msw';
import MCPRegistryPage from './MCPRegistryPage';
import { getMockedSearchMCPServersResponse } from '../test-utils';

describe('MCPRegistryPage', () => {
  const server = setupServer(getMockedSearchMCPServersResponse([]));

  const renderPage = (initialEntries = ['/']) => {
    const queryClient = new QueryClient();
    render(<MCPRegistryPage />, {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <TestRouter
            routes={[
              testRoute(
                <DesignSystemProvider>
                  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
                </DesignSystemProvider>,
                '/',
              ),
              testRoute(<div />, '*'),
            ]}
            initialEntries={initialEntries}
          />
        </IntlProvider>
      ),
    });
  };

  it('renders page title', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('MCP Registry')).toBeInTheDocument();
    });
  });

  it('renders both tabs', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Servers')).toBeInTheDocument();
      expect(screen.getByText('Access Bindings')).toBeInTheDocument();
    });
  });

  it('defaults to servers tab with search input', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search MCP servers by name')).toBeInTheDocument();
    });
  });

  it('switches to access bindings tab', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('MCP Registry')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Access Bindings'));
    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search access bindings')).toBeInTheDocument();
    });
  });

  it('shows empty state on servers tab when no servers exist', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Create and manage MCP servers using MLflow.')).toBeInTheDocument();
    });
  });

  it('shows empty state on access bindings tab when no servers exist', async () => {
    renderPage(['/?tab=bindings']);
    await waitFor(() => {
      expect(screen.getByText('Create and manage access bindings for your MCP servers.')).toBeInTheDocument();
    });
  });
});
