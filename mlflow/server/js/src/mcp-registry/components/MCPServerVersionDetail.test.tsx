import { beforeEach, describe, it, expect, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

import { MCPServerVersionDetail } from './MCPServerVersionDetail';
import { createMockMCPServer, createMockMCPServerVersion } from '../test-utils';
import { MCPStatus, type TransportType } from '../types';

jest.mock('../hooks/useServerState', () => ({
  useServerState: jest.fn(),
}));

jest.mock('../hooks/useAddAccessEndpointModal', () => ({
  useAddAccessEndpointModal: () => ({ AddAccessEndpointModal: null, openAddEndpoint: jest.fn() }),
}));

jest.mock('../hooks/useEditAccessEndpointModal', () => ({
  useEditAccessEndpointModal: () => ({ EditAccessEndpointModal: null, openEditEndpoint: jest.fn() }),
}));

jest.mock('../hooks/useDeleteAccessEndpointModal', () => ({
  useDeleteAccessEndpointModal: () => ({ DeleteAccessEndpointModal: null, openDeleteEndpoint: jest.fn() }),
}));

jest.mock('../hooks/useDeleteVersionModal', () => ({
  useDeleteVersionModal: () => ({ DeleteVersionModal: null, openDeleteVersionModal: jest.fn() }),
}));

jest.mock('../hooks/useMCPServerVersionMutations', () => ({
  useUpdateMCPServerVersion: jest.fn(),
}));

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { useServerState } = require('../hooks/useServerState') as {
  useServerState: jest.Mock;
};

// eslint-disable-next-line @typescript-eslint/no-require-imports
const { useUpdateMCPServerVersion } = require('../hooks/useMCPServerVersionMutations') as {
  useUpdateMCPServerVersion: jest.Mock;
};

const mockPermissions = ({ canUpdate = false, canDelete = false } = {}) => {
  useServerState.mockReturnValue({
    canUpdate,
    canDelete,
    canManage: false,
    isDimmed: false,
    showVisibilityControls: false,
    isAuthAvailable: true,
  });
};

const mockUpdateVersionMutation = (overrides = {}) => {
  const mutation = {
    mutate: jest.fn(),
    reset: jest.fn(),
    isLoading: false,
    error: null,
    ...overrides,
  };
  useUpdateMCPServerVersion.mockReturnValue(mutation);
  return mutation;
};

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>
    <IntlProvider locale="en">
      <DesignSystemProvider>{children}</DesignSystemProvider>
    </IntlProvider>
  </QueryClientProvider>
);

const serverWithRemotes = createMockMCPServer({ name: 'io.github.test/server' });
const versionWithRemotes = createMockMCPServerVersion({
  server_json: {
    name: 'io.github.test/server',
    version: '1.0.0',
    remotes: [{ type: 'streamable-http' as TransportType, url: 'https://mcp.example.com' }],
  },
});

const versionWithoutRemotes = createMockMCPServerVersion({
  server_json: {
    name: 'io.github.test/server',
    version: '1.0.0',
  },
});

const versionTwo = createMockMCPServerVersion({
  version: '2',
  server_json: {
    name: 'io.github.test/server',
    version: '2.0.0',
  },
});

const renderDetail = (props: Partial<React.ComponentProps<typeof MCPServerVersionDetail>> = {}) =>
  render(
    <Wrapper>
      <MCPServerVersionDetail
        server={serverWithRemotes}
        version={versionWithRemotes}
        aliasesByVersion={{}}
        {...props}
      />
    </Wrapper>,
  );

beforeEach(() => {
  jest.clearAllMocks();
  mockUpdateVersionMutation();
});

describe('Auto-discover tools button', () => {
  const clickToolsTab = async () => {
    await userEvent.click(screen.getByRole('tab', { name: /tools/i }));
  };

  it('visible when remotes exist and no auth (canUpdate defaults true)', async () => {
    mockPermissions({ canUpdate: true });
    renderDetail();
    await clickToolsTab();
    expect(screen.getByText('Auto-discover tools')).toBeInTheDocument();
  });

  it('visible when remotes exist and user has UPDATE permission', async () => {
    mockPermissions({ canUpdate: true });
    renderDetail();
    await clickToolsTab();
    expect(screen.getByText('Auto-discover tools')).toBeInTheDocument();
  });

  it('hidden when remotes exist but user lacks UPDATE permission', async () => {
    mockPermissions({ canUpdate: false });
    renderDetail();
    await clickToolsTab();
    expect(screen.queryByText('Auto-discover tools')).toBeNull();
  });

  it('hidden when no remotes', async () => {
    mockPermissions({ canUpdate: true });
    renderDetail({ version: versionWithoutRemotes });
    await clickToolsTab();
    expect(screen.queryByText('Auto-discover tools')).toBeNull();
  });
});

describe('Status editor', () => {
  it('keeps optimistic status until the selected version status refetches', async () => {
    mockPermissions({ canUpdate: true });
    const mutate = jest.fn((_payload, options: { onSuccess?: () => void; onError?: () => void }) => {
      options.onSuccess?.();
    });
    mockUpdateVersionMutation({ mutate });
    const activeVersion = createMockMCPServerVersion({ status: MCPStatus.ACTIVE });
    const deprecatedVersion = createMockMCPServerVersion({ status: MCPStatus.DEPRECATED });
    const { rerender } = renderDetail({ version: activeVersion });

    await userEvent.click(screen.getByLabelText('Edit version status'));
    await userEvent.click(await screen.findByRole('option', { name: 'Deprecated' }));

    expect(mutate).toHaveBeenCalledWith(
      { version: activeVersion.version, status: MCPStatus.DEPRECATED },
      { onError: expect.any(Function) },
    );
    expect(screen.getByText(MCPStatus.DEPRECATED)).toBeInTheDocument();

    rerender(
      <Wrapper>
        <MCPServerVersionDetail server={serverWithRemotes} version={activeVersion} aliasesByVersion={{}} />
      </Wrapper>,
    );
    expect(screen.getByText(MCPStatus.DEPRECATED)).toBeInTheDocument();

    rerender(
      <Wrapper>
        <MCPServerVersionDetail server={serverWithRemotes} version={deprecatedVersion} aliasesByVersion={{}} />
      </Wrapper>,
    );
    expect(screen.getByText(MCPStatus.DEPRECATED)).toBeInTheDocument();
  });

  it('closes when the selected version changes', async () => {
    mockPermissions({ canUpdate: true });
    const { rerender } = renderDetail();

    await userEvent.click(screen.getByLabelText('Edit version status'));
    expect(screen.getByRole('combobox', { name: 'Version status' })).toBeInTheDocument();

    rerender(
      <Wrapper>
        <MCPServerVersionDetail server={serverWithRemotes} version={versionTwo} aliasesByVersion={{}} />
      </Wrapper>,
    );

    expect(screen.queryByRole('combobox', { name: 'Version status' })).not.toBeInTheDocument();
    expect(screen.getByLabelText('Edit version status')).toBeInTheDocument();
  });
});
