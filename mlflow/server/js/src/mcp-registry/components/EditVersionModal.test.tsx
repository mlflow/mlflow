import { describe, it, expect, jest, beforeAll, afterAll, afterEach } from '@jest/globals';
import { MCPStatus } from '../types';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { setupServer } from '../../common/utils/setup-msw';
import { EditVersionModal } from './EditVersionModal';
import {
  createMockMCPServer,
  createMockMCPServerVersion,
  getMockedUpdateMCPServerVersionResponse,
} from '../test-utils';

const server = setupServer();
beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

const renderModal = (props: Partial<React.ComponentProps<typeof EditVersionModal>> = {}) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const mockServer = createMockMCPServer();
  const mockVersion = createMockMCPServerVersion({
    status: MCPStatus.DRAFT,
    tools: [{ name: 'test_tool', description: 'A test tool' }],
  });
  const defaultProps = {
    visible: true,
    server: mockServer,
    version: mockVersion,
    aliasesByVersion: {},
    onClose: jest.fn(),
    ...props,
  };
  return {
    ...render(
      <QueryClientProvider client={queryClient}>
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <EditVersionModal {...defaultProps} />
          </DesignSystemProvider>
        </IntlProvider>
      </QueryClientProvider>,
    ),
    onClose: defaultProps.onClose,
  };
};

describe('EditVersionModal', () => {
  it('renders with status and aliases fields', () => {
    renderModal();
    expect(screen.getByText('Edit version details')).toBeInTheDocument();
    expect(screen.getByText('Status')).toBeInTheDocument();
    expect(screen.getByText('Aliases')).toBeInTheDocument();
    expect(screen.queryByText('Display name')).not.toBeInTheDocument();
  });

  it('renders status selector with current status', () => {
    renderModal();
    expect(screen.getByText('Draft')).toBeInTheDocument();
  });

  it('does not call onClose without interaction', () => {
    const { onClose } = renderModal();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('submits successfully and calls onClose', async () => {
    const updatedVersion = createMockMCPServerVersion({ status: MCPStatus.ACTIVE });
    server.use(getMockedUpdateMCPServerVersionResponse(updatedVersion));

    const { onClose } = renderModal();
    await userEvent.click(screen.getByText('Save'));

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled();
    });
  });
});
