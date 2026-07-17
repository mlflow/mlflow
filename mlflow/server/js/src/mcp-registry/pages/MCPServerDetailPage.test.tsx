import { describe, it, expect, jest } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import { setupServer } from '../../common/utils/setup-msw';
import MCPServerDetailPage from './MCPServerDetailPage';
import { TransportType, MCPStatus, MCPServerAction } from '../types';
import {
  createMockMCPServer,
  createMockMCPServerVersion,
  getMockedGetMCPServerResponse,
  getMockedGetMCPServerErrorResponse,
  getMockedSearchMCPServerVersionsResponse,
  getMockedDeleteMCPServerVersionResponse,
  getMockedDeleteMCPServerResponse,
  getMockedGetLatestMCPServerVersionResponse,
  getMockedUpdateMCPServerResponse,
  getMockedSetMCPServerTagResponse,
  getMockedDeleteMCPServerTagResponse,
  getMockedCurrentUserResponse,
} from '../test-utils';

// Monaco does not render in jsdom; stand the editor in with a labelled textarea.
jest.mock('../../experiment-tracking/pages/experiment-evaluation-datasets-v2/components/LazyJsonRecordEditor', () => ({
  LazyJsonRecordEditor: ({
    ariaLabel,
    value,
    onChange,
  }: {
    ariaLabel: string;
    value: string;
    onChange: (next: string) => void;
  }) => <textarea aria-label={ariaLabel} value={value} onChange={(e) => onChange(e.target.value)} />,
}));

const mockServer = createMockMCPServer({
  name: 'dev.mainline/mcp',
  display_name: 'Mainline',
  description: 'A test server',
});
const mockVersion = createMockMCPServerVersion({
  name: 'dev.mainline/mcp',
  version: '1',
  status: MCPStatus.ACTIVE,
  server_json: {
    name: 'dev.mainline/mcp',
    version: '1.0.0',
    title: 'Mainline',
    description: 'Gives your AI agent your story map.',
    packages: [
      {
        registryType: 'npm',
        identifier: '@mainline/mcp-server',
        version: '1.0.0',
        runtimeHint: 'npx',
        transport: { type: TransportType.STDIO },
        environmentVariables: [
          { name: 'API_KEY', description: 'API key for authentication', isRequired: true, isSecret: true },
          { name: 'LOG_LEVEL', description: 'Logging verbosity' },
        ],
      },
      {
        registryType: 'pypi',
        identifier: 'mainline-mcp-server',
        transport: { type: TransportType.STDIO },
      },
    ],
    remotes: [{ type: TransportType.STREAMABLE_HTTP, url: 'https://api.mainline.dev/mcp' }],
  },
});

const defaultHandlers = [
  getMockedGetLatestMCPServerVersionResponse(mockVersion),
  getMockedGetMCPServerResponse(mockServer),
  getMockedSearchMCPServerVersionsResponse([mockVersion]),
  getMockedDeleteMCPServerVersionResponse(),
  getMockedDeleteMCPServerResponse(),
  getMockedCurrentUserResponse({ isAdmin: true }),
];

describe('MCPServerDetailPage', () => {
  const server = setupServer(...defaultHandlers);

  const renderPage = (initialEntries = ['/mcp-registry/dev.mainline%2Fmcp']) => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<MCPServerDetailPage />, {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <TestRouter
            routes={[
              testRoute(
                <DesignSystemProvider>
                  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
                </DesignSystemProvider>,
                '/mcp-registry/:serverName',
              ),
              testRoute(<div data-testid="mcp-registry-list" />, '/mcp-registry'),
              testRoute(<div />, '*'),
            ]}
            initialEntries={initialEntries}
          />
        </IntlProvider>
      ),
    });
  };

  it('renders breadcrumb and server name', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getAllByText('Mainline').length).toBeGreaterThanOrEqual(1);
    });
    expect(screen.getAllByText('MCP Registry').length).toBeGreaterThanOrEqual(1);
  });

  it('renders version list with status badge', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('1')).toBeInTheDocument();
    });
    expect(screen.getAllByText('active').length).toBeGreaterThanOrEqual(1);
  });

  it('renders version detail metadata', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
    });
    expect(screen.getAllByText('dev.mainline/mcp').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Gives your AI agent your story map.')).toBeInTheDocument();
  });

  it('renders error state when server not found', async () => {
    server.use(getMockedGetMCPServerErrorResponse(404, 'Server not found'));
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Failed to load MCP server')).toBeInTheDocument();
    });
  });

  it('shows packages in version detail', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText('Run locally')).toBeInTheDocument();
    });
    expect(screen.getByText('npm')).toBeInTheDocument();
    expect(screen.getByText('pypi')).toBeInTheDocument();
  });

  it('shows remotes in version detail', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText('Official endpoints')).toBeInTheDocument();
    });
    expect(screen.getByText('streamable-http')).toBeInTheDocument();
    expect(screen.getByText('https://api.mainline.dev/mcp')).toBeInTheDocument();
  });

  it('expands a package row to show environment variables', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText('npm')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('button', { name: /Expand package @mainline\/mcp-server/ }));
    await waitFor(() => {
      expect(screen.getByText('View details')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('View details'));
    await waitFor(() => {
      expect(screen.getByText('Environment Variables (2)')).toBeInTheDocument();
    });
    expect(screen.getByText('API_KEY')).toBeInTheDocument();
    expect(screen.getByText('required')).toBeInTheDocument();
    expect(screen.getByText('secret')).toBeInTheDocument();
    expect(screen.getByText('API key for authentication')).toBeInTheDocument();
    expect(screen.getByText('LOG_LEVEL')).toBeInTheDocument();
  });

  it('toggles raw server.json view', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText('View raw server.json')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('View raw server.json'));
    await waitFor(() => {
      expect(screen.getByText(/"name": "dev.mainline\/mcp"/)).toBeInTheDocument();
    });
    expect(screen.getByText('Hide raw server.json')).toBeInTheDocument();

    await userEvent.click(screen.getByText('Hide raw server.json'));
    await waitFor(() => {
      expect(screen.queryByText(/"name": "dev.mainline\/mcp"/)).not.toBeInTheDocument();
    });
  });

  it('opens edit version modal with status select when Edit is clicked', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
    });

    const editBtn = document.querySelector(
      '[data-component-id="mlflow.mcp_registry.detail.edit_version"]',
    ) as HTMLElement;
    await userEvent.click(editBtn);
    await waitFor(() => {
      expect(screen.getByText('Edit version details')).toBeInTheDocument();
      expect(screen.getByText('Status')).toBeInTheDocument();
    });
  });

  it('selects different version when multiple exist', async () => {
    const version2 = createMockMCPServerVersion({
      name: 'dev.mainline/mcp',
      version: '2',
      status: MCPStatus.DRAFT,
      server_json: {
        name: 'dev.mainline/mcp',
        version: '2.0.0',
        title: 'Mainline v2',
        description: 'Updated version.',
      },
    });
    server.use(getMockedSearchMCPServerVersionsResponse([mockVersion, version2]));

    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('2'));
    await waitFor(() => {
      expect(screen.getByText('Viewing version 2')).toBeInTheDocument();
    });
  });

  it('opens delete version confirmation modal', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('button', { name: /Delete version/ }));
    await waitFor(() => {
      expect(screen.getByText(/Are you sure you want to delete version/)).toBeInTheDocument();
    });
  });

  it('opens delete server confirmation modal from overflow menu', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getAllByText('Mainline').length).toBeGreaterThanOrEqual(1);
    });

    await userEvent.click(screen.getByRole('button', { name: 'More actions' }));
    const menuItems = await screen.findAllByRole('menuitem');
    const deleteItem = menuItems.find((item) => item.textContent === 'Delete');
    expect(deleteItem).toBeDefined();
    await userEvent.click(deleteItem!);
    await waitFor(() => {
      expect(screen.getByText(/Are you sure you want to delete this MCP server/)).toBeInTheDocument();
    });
  });

  it('selects first version by default when multiple exist', async () => {
    const version2 = createMockMCPServerVersion({
      name: 'dev.mainline/mcp',
      version: '2',
      status: MCPStatus.DRAFT,
      server_json: {
        name: 'dev.mainline/mcp',
        version: '2.0.0',
        title: 'Mainline v2',
        description: 'Updated version.',
      },
    });
    server.use(getMockedSearchMCPServerVersionsResponse([mockVersion, version2]));

    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
    });
  });

  it('does not rewrite URL when version param is not in loaded page', async () => {
    renderPage(['/mcp-registry/dev.mainline%2Fmcp?version=nonexistent']);
    await waitFor(() => {
      expect(screen.getByText('Select a version to view details.')).toBeInTheDocument();
    });
  });

  it('persists selected version across clicks', async () => {
    const version2 = createMockMCPServerVersion({
      name: 'dev.mainline/mcp',
      version: '2',
      status: MCPStatus.DRAFT,
      server_json: {
        name: 'dev.mainline/mcp',
        version: '2.0.0',
        title: 'Mainline v2',
        description: 'Updated version.',
      },
    });
    server.use(getMockedSearchMCPServerVersionsResponse([mockVersion, version2]));

    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('2'));
    await waitFor(() => {
      expect(screen.getByText('Viewing version 2')).toBeInTheDocument();
    });
  });

  it('disables all status transitions for deleted version in edit modal', async () => {
    const deletedVersion = createMockMCPServerVersion({
      name: 'dev.mainline/mcp',
      version: '1',
      status: MCPStatus.DELETED,
      server_json: {
        name: 'dev.mainline/mcp',
        version: '1.0.0',
        title: 'Mainline',
        description: 'Deleted version.',
      },
    });
    server.use(getMockedSearchMCPServerVersionsResponse([deletedVersion]));
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
    });

    const editBtn = document.querySelector(
      '[data-component-id="mlflow.mcp_registry.detail.edit_version"]',
    ) as HTMLElement;
    await userEvent.click(editBtn);
    await waitFor(() => {
      expect(screen.getByText('Edit version details')).toBeInTheDocument();
    });
    expect(screen.getByText('Status')).toBeInTheDocument();
  });

  it('displays server description as read-only', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Gives your AI agent your story map.')).toBeInTheDocument();
    });

    expect(
      document.querySelector('[data-component-id="mlflow.mcp_registry.detail.version.edit_description"]'),
    ).not.toBeInTheDocument();
  });

  describe('server display name editing', () => {
    it('opens edit display name modal from overflow menu', async () => {
      server.use(getMockedUpdateMCPServerResponse());
      renderPage();
      await waitFor(() => {
        expect(screen.getAllByText('Mainline').length).toBeGreaterThanOrEqual(1);
      });

      await userEvent.click(screen.getByRole('button', { name: 'More actions' }));
      const menuItems = await screen.findAllByRole('menuitem');
      const editItem = menuItems.find((item) => item.textContent === 'Edit display name');
      expect(editItem).toBeDefined();
      await userEvent.click(editItem!);
      await waitFor(() => {
        expect(screen.getByText('Edit display name')).toBeInTheDocument();
      });
    });
  });

  describe('server tags', () => {
    it('shows "Add tags" button when server has no tags', async () => {
      server.use(getMockedSetMCPServerTagResponse(), getMockedDeleteMCPServerTagResponse());
      renderPage();
      await waitFor(() => {
        expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
      });

      expect(screen.getByText('Add tags')).toBeInTheDocument();
    });

    it('shows tags when server has tags', async () => {
      const taggedServer = createMockMCPServer({
        name: 'dev.mainline/mcp',
        display_name: 'Mainline',
        description: 'A test server',
        tags: { env: 'production', team: 'platform' },
      });
      server.use(
        getMockedGetMCPServerResponse(taggedServer),
        getMockedSetMCPServerTagResponse(),
        getMockedDeleteMCPServerTagResponse(),
      );
      renderPage();
      await waitFor(() => {
        expect(screen.getByText('env')).toBeInTheDocument();
        expect(screen.getByText('team')).toBeInTheDocument();
      });
    });
  });

  describe('permission gating via allowed_actions', () => {
    const setupWithPermissions = (allowed_actions?: MCPServerAction[]) => {
      const permServer = createMockMCPServer({
        name: 'dev.mainline/mcp',
        display_name: 'Mainline',
        description: 'A test server',
        allowed_actions,
      });
      server.use(getMockedGetMCPServerResponse(permServer), getMockedCurrentUserResponse({ isAdmin: false }));
    };

    it('hides all action buttons for READ-only user', async () => {
      setupWithPermissions([]);
      renderPage();
      await waitFor(() => {
        expect(screen.getAllByText('Mainline').length).toBeGreaterThanOrEqual(1);
      });
      expect(screen.queryByText('Create MCP server version')).not.toBeInTheDocument();
      expect(screen.queryByText('Edit')).not.toBeInTheDocument();
      expect(screen.queryByText('Delete version')).not.toBeInTheDocument();
      expect(screen.queryByLabelText('More actions')).not.toBeInTheDocument();
    });

    it('shows edit buttons for EDIT user', async () => {
      setupWithPermissions([MCPServerAction.USE, MCPServerAction.UPDATE]);
      renderPage();
      await waitFor(() => {
        expect(screen.getByText('Create MCP server version')).toBeInTheDocument();
      });
      expect(screen.getByText('Edit')).toBeInTheDocument();
      expect(screen.getByLabelText('More actions')).toBeInTheDocument();
      expect(screen.queryByText('Delete version')).not.toBeInTheDocument();
    });

    it('shows all buttons for MANAGE user', async () => {
      setupWithPermissions([
        MCPServerAction.USE,
        MCPServerAction.UPDATE,
        MCPServerAction.DELETE,
        MCPServerAction.MANAGE,
      ]);
      renderPage();
      await waitFor(() => {
        expect(screen.getByText('Create MCP server version')).toBeInTheDocument();
      });
      expect(screen.getByText('Edit')).toBeInTheDocument();
      expect(screen.getByText('Delete version')).toBeInTheDocument();
      expect(screen.getByLabelText('More actions')).toBeInTheDocument();
    });

    it('shows all buttons when allowed_actions is undefined (no auth)', async () => {
      setupWithPermissions(undefined);
      renderPage();
      await waitFor(() => {
        expect(screen.getByText('Create MCP server version')).toBeInTheDocument();
      });
      expect(screen.getByText('Edit')).toBeInTheDocument();
      expect(screen.getByText('Delete version')).toBeInTheDocument();
    });

    it('shows Unavailable tag when auth is available and server has no bindings', async () => {
      setupWithPermissions([]);
      renderPage();
      await waitFor(() => {
        expect(screen.getByText('Unavailable')).toBeInTheDocument();
      });
    });
  });
});
