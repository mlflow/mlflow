import { describe, it, expect, jest } from '@jest/globals';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { rest } from 'msw';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import { setupServer } from '../../common/utils/setup-msw';
import {
  getMockedSearchMCPServersResponse,
  getMockedCreateMCPServerVersionResponse,
  getMockedCreateMCPServerVersionErrorResponse,
  getMockedUpdateMCPServerResponse,
  createMockMCPServerVersion,
} from '../test-utils';
import { useCreateMCPServerVersionModal } from './useCreateMCPServerVersionModal';

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

const VALID_SERVER_JSON = JSON.stringify({
  name: 'io.github.test/server',
  version: '1.0.0',
  description: 'Test server',
});

const setTextareaValue = (element: HTMLElement, value: string) => {
  fireEvent.change(element, { target: { value } });
};

const TestComponent = ({ onSuccess }: { onSuccess?: (result: { name: string; version: string }) => void }) => {
  const { CreateMCPServerVersionModal, openModal } = useCreateMCPServerVersionModal({ onSuccess });
  return (
    <>
      <button onClick={openModal}>Open</button>
      {CreateMCPServerVersionModal}
    </>
  );
};

const VersionModeTestComponent = ({
  onSuccess,
  latestVersion,
}: {
  onSuccess?: (result: { name: string; version: string }) => void;
  latestVersion: ReturnType<typeof createMockMCPServerVersion>;
}) => {
  const { CreateMCPServerVersionModal, openModal } = useCreateMCPServerVersionModal({
    onSuccess,
    serverName: latestVersion.name,
    latestVersion,
  });
  return (
    <>
      <button onClick={openModal}>Open</button>
      {CreateMCPServerVersionModal}
    </>
  );
};

describe('useCreateMCPServerVersionModal', () => {
  const mswServer = setupServer(
    getMockedSearchMCPServersResponse([]),
    getMockedCreateMCPServerVersionResponse(),
    getMockedUpdateMCPServerResponse(),
  );

  const renderModal = (onSuccess?: (result: { name: string; version: string }) => void) => {
    const queryClient = new QueryClient();
    render(<TestComponent onSuccess={onSuccess} />, {
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
            ]}
            initialEntries={['/']}
          />
        </IntlProvider>
      ),
    });
  };

  const openModal = async () => {
    await userEvent.click(screen.getByText('Open'));
    await waitFor(() => {
      expect(screen.getByText('Create MCP server')).toBeInTheDocument();
    });
  };

  it('opens modal with all form fields', async () => {
    renderModal();
    await openModal();

    expect(screen.getByText('Display name:')).toBeInTheDocument();
    expect(screen.getByText(/server\.json:/)).toBeInTheDocument();
    expect(screen.getByText('Status:')).toBeInTheDocument();
    expect(screen.getByText('Source:')).toBeInTheDocument();
    expect(screen.getByText('Tools:')).toBeInTheDocument();
    expect(screen.getByText('Cancel')).toBeInTheDocument();
    expect(screen.getByText('Create')).toBeInTheDocument();
  });

  it('disables Create button when server.json is empty', async () => {
    renderModal();
    await openModal();

    const createButton = screen.getByRole('button', { name: 'Create' });
    expect(createButton).toBeDisabled();
  });

  it('shows validation error for invalid JSON', async () => {
    renderModal();
    await openModal();

    const textarea = screen.getByLabelText('server.json editor');
    setTextareaValue(textarea, '{invalid json');
    await userEvent.click(screen.getByText('Create'));

    await waitFor(() => {
      expect(screen.getByText('Invalid JSON format in server configuration')).toBeInTheDocument();
    });
  });

  it('shows validation error when name is missing from server.json', async () => {
    renderModal();
    await openModal();

    const textarea = screen.getByLabelText('server.json editor');
    setTextareaValue(textarea, '{"version": "1.0.0"}');
    await userEvent.click(screen.getByText('Create'));

    await waitFor(() => {
      expect(screen.getByText('Server configuration must include a "name" field')).toBeInTheDocument();
    });
  });

  it('shows validation error when version is missing from server.json', async () => {
    renderModal();
    await openModal();

    const textarea = screen.getByLabelText('server.json editor');
    setTextareaValue(textarea, '{"name": "test-server"}');
    await userEvent.click(screen.getByText('Create'));

    await waitFor(() => {
      expect(screen.getByText('Server configuration must include a "version" field')).toBeInTheDocument();
    });
  });

  it('submits successfully with valid server.json and calls onSuccess', async () => {
    const onSuccess = jest.fn();
    const mockVersion = createMockMCPServerVersion({
      name: 'io.github.test/server',
      version: '1.0.0',
    });
    mswServer.use(getMockedCreateMCPServerVersionResponse(mockVersion));

    renderModal(onSuccess);
    await openModal();

    const textarea = screen.getByLabelText('server.json editor');
    setTextareaValue(textarea, VALID_SERVER_JSON);
    await userEvent.click(screen.getByText('Create'));

    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalledWith({
        name: 'io.github.test/server',
        version: '1.0.0',
      });
    });
  });

  it('displays API error when creation fails', async () => {
    mswServer.use(getMockedCreateMCPServerVersionErrorResponse(409, 'Version already exists'));

    renderModal();
    await openModal();

    const textarea = screen.getByLabelText('server.json editor');
    setTextareaValue(textarea, VALID_SERVER_JSON);
    await userEvent.click(screen.getByText('Create'));

    await waitFor(() => {
      expect(screen.getByText('Version already exists')).toBeInTheDocument();
    });
  });

  it('closes modal on cancel', async () => {
    renderModal();
    await openModal();

    await userEvent.click(screen.getByText('Cancel'));

    await waitFor(() => {
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });
  });

  it('clears validation errors when form fields change', async () => {
    renderModal();
    await openModal();

    // Enter invalid JSON and submit to trigger a validation error
    const textarea = screen.getByLabelText('server.json editor');
    setTextareaValue(textarea, '{bad json');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() => {
      expect(screen.getByText('Invalid JSON format in server configuration')).toBeInTheDocument();
    });

    // Change the textarea value to clear the error
    setTextareaValue(textarea, 'something else');

    await waitFor(() => {
      expect(screen.queryByText('Invalid JSON format in server configuration')).not.toBeInTheDocument();
    });
  });

  it('resets form state when reopened', async () => {
    renderModal();
    await openModal();

    // Type something in display name
    const displayNameInput = screen.getByPlaceholderText('Human-readable label for this server');
    await userEvent.type(displayNameInput, 'My Server');

    // Close and reopen
    await userEvent.click(screen.getByText('Cancel'));
    await waitFor(() => {
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    });

    await openModal();

    // Display name should be empty
    const freshInput = screen.getByPlaceholderText('Human-readable label for this server');
    expect(freshInput).toHaveValue('');
  });

  it('carries over tools from latest version when creating a new version', async () => {
    const tools = [{ name: 'search', description: 'Search the web' }];
    const latestVersion = createMockMCPServerVersion({ tools });
    const capturedBody = jest.fn();

    mswServer.use(
      rest.post(getAjaxUrl('ajax-api/3.0/mlflow/mcp-servers/:name/versions'), async (req, res, ctx) => {
        capturedBody(await req.json());
        return res(ctx.json(createMockMCPServerVersion({ version: '2' })));
      }),
    );

    const queryClient = new QueryClient();
    render(<VersionModeTestComponent latestVersion={latestVersion} />, {
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
            ]}
            initialEntries={['/']}
          />
        </IntlProvider>
      ),
    });

    await userEvent.click(screen.getByText('Open'));
    await waitFor(() => {
      expect(screen.getByText('Create MCP server version')).toBeInTheDocument();
    });

    expect(screen.queryByText('Tools:')).not.toBeInTheDocument();

    await userEvent.click(screen.getByText('Create'));

    await waitFor(() => {
      expect(capturedBody).toHaveBeenCalled();
    });

    expect(capturedBody.mock.calls[0][0]).toMatchObject({ tools });
  });
});
