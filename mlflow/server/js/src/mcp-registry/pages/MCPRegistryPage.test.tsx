import { describe, it, expect } from '@jest/globals';
import { MCPStatus, MCPServerAction } from '../types';
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
  getMockedCurrentUserResponse,
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

  describe('Available/All toggle visibility', () => {
    const activeServer = (overrides: Partial<ReturnType<typeof createMockMCPServer>> = {}) =>
      createMockMCPServer({
        name: 'server-1',
        status: MCPStatus.ACTIVE,
        access_endpoints: [
          {
            id: 1,
            server_name: 'server-1',
            url: 'https://example.com',
            transport_type: 'streamable-http',
          } as any,
        ],
        ...overrides,
      });

    it('hides toggle when user_has_manage is false', async () => {
      server.use(
        getMockedSearchMCPServersResponse([activeServer({ allowed_actions: [] })], { userHasManage: false }),
        getMockedCurrentUserResponse({ isAdmin: false }),
      );
      renderPage();
      await waitFor(() => {
        expect(screen.getByText('server-1')).toBeInTheDocument();
      });
      expect(screen.queryByTestId('mcp-registry-availability-filter')).not.toBeInTheDocument();
    });

    it('shows toggle when user_has_manage is true', async () => {
      server.use(
        getMockedSearchMCPServersResponse([activeServer()], { userHasManage: true }),
        getMockedCurrentUserResponse({ isAdmin: false }),
      );
      renderPage();
      await waitFor(() => {
        expect(screen.getByText('server-1')).toBeInTheDocument();
      });
      await waitFor(() => {
        expect(screen.getByTestId('mcp-registry-availability-filter')).toBeVisible();
      });
    });

    it('shows toggle for admin users (user_has_manage undefined)', async () => {
      server.use(getMockedSearchMCPServersResponse([activeServer()]), getMockedCurrentUserResponse({ isAdmin: true }));
      renderPage();
      await waitFor(() => {
        expect(screen.getByText('server-1')).toBeInTheDocument();
      });
      await waitFor(() => {
        expect(screen.getByTestId('mcp-registry-availability-filter')).toBeVisible();
      });
    });

    it('hides dimmed servers without MANAGE in All mode', async () => {
      const managedServer = activeServer({
        name: 'my-server',
        allowed_actions: [MCPServerAction.USE, MCPServerAction.UPDATE, MCPServerAction.DELETE, MCPServerAction.MANAGE],
      });
      const dimmedReadOnly = createMockMCPServer({
        name: 'other-server',
        status: MCPStatus.DRAFT,
        access_endpoints: [],
        allowed_actions: [],
      });
      server.use(
        getMockedSearchMCPServersResponse([managedServer, dimmedReadOnly], { userHasManage: true }),
        getMockedCurrentUserResponse({ isAdmin: false }),
      );
      renderPage();
      await waitFor(() => {
        expect(screen.getByText('my-server')).toBeInTheDocument();
      });
      await waitFor(() => {
        expect(screen.getByTestId('mcp-registry-availability-filter')).toBeVisible();
      });

      await userEvent.click(screen.getByText('All'));

      await waitFor(() => {
        expect(screen.getByText('my-server')).toBeInTheDocument();
        expect(screen.queryByText('other-server')).not.toBeInTheDocument();
      });
    });

    it('shows dimmed servers with MANAGE in All mode', async () => {
      const dimmedManaged = createMockMCPServer({
        name: 'my-dimmed-server',
        status: MCPStatus.DRAFT,
        access_endpoints: [],
        allowed_actions: [MCPServerAction.USE, MCPServerAction.UPDATE, MCPServerAction.DELETE, MCPServerAction.MANAGE],
      });
      server.use(
        getMockedSearchMCPServersResponse([dimmedManaged], { userHasManage: true }),
        getMockedCurrentUserResponse({ isAdmin: false }),
      );
      renderPage();
      await waitFor(() => {
        expect(screen.getByTestId('mcp-registry-availability-filter')).toBeVisible();
      });

      await userEvent.click(screen.getByText('All'));

      await waitFor(() => {
        expect(screen.getByText('my-dimmed-server')).toBeInTheDocument();
      });
    });
  });
});
