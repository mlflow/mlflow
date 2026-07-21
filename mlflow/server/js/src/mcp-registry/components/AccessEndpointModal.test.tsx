import { describe, it, expect, jest, beforeAll, afterAll, afterEach } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { getAjaxUrl } from '../../common/utils/FetchUtils';
import { setupServer } from '../../common/utils/setup-msw';
import { AccessEndpointModal } from './AccessEndpointModal';
import { TransportType } from '../types';
import {
  createMockAccessEndpoint,
  createMockMCPServer,
  createMockMCPServerVersion,
  getMockedGetMCPServerResponse,
  getMockedSearchMCPServersResponse,
  getMockedSearchMCPServerVersionsResponse,
} from '../test-utils';

const BASE_URL = 'ajax-api/3.0/mlflow/mcp-servers';

const server = setupServer();
beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

const mockServer = createMockMCPServer({
  name: 'com.test/my-server',
  aliases: [{ alias: 'stable', version: '1.0.0' }],
});
const mockVersion = createMockMCPServerVersion({ version: '1.0.0' });

const renderModal = (props = {}) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  server.use(
    getMockedSearchMCPServersResponse([mockServer]),
    getMockedGetMCPServerResponse(mockServer),
    getMockedSearchMCPServerVersionsResponse([mockVersion]),
  );
  const defaultProps = {
    visible: true,
    onCancel: jest.fn(),
    onSuccess: jest.fn(),
    ...props,
  };
  return {
    ...render(
      <QueryClientProvider client={queryClient}>
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <AccessEndpointModal {...defaultProps} />
          </DesignSystemProvider>
        </IntlProvider>
      </QueryClientProvider>,
    ),
    onCancel: defaultProps.onCancel,
    onSuccess: defaultProps.onSuccess,
  };
};

describe('AccessEndpointModal', () => {
  describe('create mode', () => {
    it('renders create title by default', () => {
      renderModal();
      expect(screen.getByText('Create access endpoint')).toBeInTheDocument();
    });

    it('renders custom create title when provided', () => {
      renderModal({ createTitle: <span>Custom Title</span> });
      expect(screen.getByText('Custom Title')).toBeInTheDocument();
    });

    it('shows server selector when no lockedServer', async () => {
      renderModal();
      await waitFor(() => {
        expect(screen.getByText('Select an MCP server')).toBeInTheDocument();
      });
    });

    it('locks server when lockedServer is provided', () => {
      renderModal({ lockedServer: 'com.test/my-server' });
      expect(screen.getByText('com.test/my-server')).toBeInTheDocument();
      expect(screen.queryByText('Select an MCP server')).not.toBeInTheDocument();
    });

    it('disables Create button when form is empty', () => {
      renderModal();
      const okButton = screen.getByRole('button', { name: /Create/i });
      expect(okButton).toBeDisabled();
    });

    it('shows validation error for invalid URL', async () => {
      renderModal({ lockedServer: 'com.test/my-server' });
      const urlInput = screen.getByPlaceholderText('https://mcp.example.com/server');
      await userEvent.type(urlInput, 'not-a-url');
      await waitFor(() => {
        expect(screen.getByText(/valid HTTP or HTTPS URL/i)).toBeInTheDocument();
      });
    });

    it('submits create form successfully', async () => {
      const requestBody = jest.fn();
      server.use(
        rest.post(getAjaxUrl(`${BASE_URL}/:name/endpoints`), async (req, res, ctx) => {
          requestBody(await req.json());
          return res(ctx.json(createMockAccessEndpoint()));
        }),
      );
      const { onCancel, onSuccess } = renderModal({ lockedServer: 'com.test/my-server' });
      const urlInput = screen.getByPlaceholderText('https://mcp.example.com/server');
      await userEvent.type(urlInput, 'https://example.com/mcp');
      await userEvent.click(screen.getByRole('button', { name: /Create/i }));
      await waitFor(() => {
        expect(onCancel).toHaveBeenCalled();
        expect(onSuccess).toHaveBeenCalled();
        expect(requestBody).toHaveBeenCalledWith(expect.objectContaining({ url: 'https://example.com/mcp' }));
      });
    });
  });

  describe('edit mode', () => {
    const editEndpoint = createMockAccessEndpoint({
      server_name: 'com.test/my-server',
      url: 'https://old.example.com/mcp',
      server_alias: 'stable',
      transport_type: TransportType.STREAMABLE_HTTP,
    });

    it('renders edit title', () => {
      renderModal({ editEndpoint });
      expect(screen.getByText('Edit access endpoint')).toBeInTheDocument();
    });

    it('pre-populates URL field', () => {
      renderModal({ editEndpoint });
      expect(screen.getByDisplayValue('https://old.example.com/mcp')).toBeInTheDocument();
    });

    it('shows Save button instead of Create', () => {
      renderModal({ editEndpoint });
      expect(screen.getByRole('button', { name: /Save/i })).toBeInTheDocument();
    });

    it('locks server name in edit mode', () => {
      renderModal({ editEndpoint });
      expect(screen.queryByText('Select an MCP server')).not.toBeInTheDocument();
    });

    it('submits update successfully', async () => {
      const requestBody = jest.fn();
      server.use(
        rest.patch(getAjaxUrl(`${BASE_URL}/:name/endpoints/:endpointId`), async (req, res, ctx) => {
          requestBody(await req.json());
          return res(ctx.json(createMockAccessEndpoint()));
        }),
      );
      const { onCancel, onSuccess } = renderModal({ editEndpoint });
      const urlInput = screen.getByDisplayValue('https://old.example.com/mcp');
      await userEvent.clear(urlInput);
      await userEvent.type(urlInput, 'https://new.example.com/mcp');
      await userEvent.click(screen.getByRole('button', { name: /Save/i }));
      await waitFor(() => {
        expect(onCancel).toHaveBeenCalled();
        expect(onSuccess).toHaveBeenCalled();
        expect(requestBody).toHaveBeenCalledWith(expect.objectContaining({ url: 'https://new.example.com/mcp' }));
      });
    });
  });

  describe('EndpointTargetSelector (scoped vs unscoped)', () => {
    it('shows scoped version when scopedVersion is provided', () => {
      renderModal({ lockedServer: 'com.test/my-server', scopedVersion: '1.0.0' });
      expect(screen.getByText('1.0.0')).toBeInTheDocument();
    });

    it('shows aliases group in unscoped mode', async () => {
      renderModal({ lockedServer: 'com.test/my-server' });
      await waitFor(() => {
        expect(screen.getByText('@latest')).toBeInTheDocument();
      });
    });
  });

  it('returns null when not visible', () => {
    const { container } = renderModal({ visible: false });
    expect(container.querySelector('[class*="modal"]')).not.toBeInTheDocument();
  });
});
