import { describe, it, expect } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import { setupServer } from '../../common/utils/setup-msw';
import MCPRegistryPage from './MCPRegistryPage';
import { rest } from 'msw';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import {
  createMockMCPServer,
  getMockedSearchMCPServersResponse,
  getMockedSearchMCPServersErrorResponse,
} from '../test-utils';

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

  it('renders empty state when no servers exist', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Create and manage MCP servers using MLflow.')).toBeInTheDocument();
    });
  });

  it('renders server cards when data is available', async () => {
    const servers = [createMockMCPServer({ name: 'server-1' }), createMockMCPServer({ name: 'server-2' })];
    server.use(getMockedSearchMCPServersResponse(servers));
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('server-1')).toBeInTheDocument();
      expect(screen.getByText('server-2')).toBeInTheDocument();
    });
  });

  it('shows Create MCP server button when servers exist', async () => {
    const servers = [createMockMCPServer({ name: 'server-1' })];
    server.use(getMockedSearchMCPServersResponse(servers));
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('server-1')).toBeInTheDocument();
    });
    expect(screen.getAllByText('Create MCP server').length).toBeGreaterThanOrEqual(1);
  });

  it('renders error alert when API fails', async () => {
    server.use(getMockedSearchMCPServersErrorResponse(500, 'Something broke'));
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Something broke')).toBeInTheDocument();
    });
  });

  it('does not show header create button in empty state', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Create and manage MCP servers using MLflow.')).toBeInTheDocument();
    });
    // The empty state has a title and a CTA button both saying "Create MCP server" (2 elements).
    // The header create button is a third one that should NOT exist in empty state.
    const createTexts = screen.getAllByText('Create MCP server');
    expect(createTexts).toHaveLength(2);
  });

  it('shows empty state in list view when no servers exist', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('MCP Registry')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByLabelText('List view'));

    await waitFor(() => {
      expect(screen.getByText('Create and manage MCP servers using MLflow.')).toBeInTheDocument();
    });
  });

  it('filters servers via search input', async () => {
    let capturedFilter: string | null = null;
    server.use(
      rest.get(getAjaxUrl('ajax-api/3.0/mlflow/mcp-servers'), (req, res, ctx) => {
        capturedFilter = req.url.searchParams.get('filter_string');
        return res(ctx.json({ mcp_servers: [], next_page_token: undefined }));
      }),
    );
    renderPage();

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search MCP servers by name')).toBeInTheDocument();
    });

    await userEvent.type(screen.getByPlaceholderText('Search MCP servers by name'), 'github');

    await waitFor(() => {
      expect(capturedFilter).toBe("name ILIKE '%github%'");
    });
  });

  it('switches between grid and list views', async () => {
    const servers = [createMockMCPServer({ name: 'server-1' })];
    server.use(getMockedSearchMCPServersResponse(servers));
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('server-1')).toBeInTheDocument();
    });

    // Default is grid, switch to list
    await userEvent.click(screen.getByLabelText('List view'));

    // Table headers should appear in list view
    await waitFor(() => {
      expect(screen.getByText('Latest version')).toBeInTheDocument();
      expect(screen.getByText('Last modified')).toBeInTheDocument();
    });

    // Switch back to grid
    await userEvent.click(screen.getByLabelText('Grid view'));

    await waitFor(() => {
      expect(screen.queryByText('Latest version')).not.toBeInTheDocument();
    });
  });
});
