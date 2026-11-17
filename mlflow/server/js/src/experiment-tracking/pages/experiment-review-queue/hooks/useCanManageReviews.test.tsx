import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { NotFoundError } from '@databricks/web-shared/errors';
import { AccountApi } from '../../../../account/api';
import { useCanEditReviews, useCanManageReviews } from './useCanManageReviews';

let mockUser: any;
let mockActiveWorkspace: string | null = 'ws-a';
let mockWorkspacesEnabled = true;
let mockWorkspacesLoading = false;
jest.mock('../../../../account/api', () => ({ AccountApi: { getUserPermission: jest.fn() } }));
jest.mock('../../../../account/hooks', () => ({ useCurrentUserQuery: () => mockUser }));
jest.mock('../../../../workspaces/utils/WorkspaceUtils', () => ({ useActiveWorkspace: () => mockActiveWorkspace }));
jest.mock('../../../hooks/useServerInfo', () => ({
  useWorkspacesEnabled: () => ({ workspacesEnabled: mockWorkspacesEnabled, loading: mockWorkspacesLoading }),
}));

const renderPermission = (levels: 'edit' | 'manage' = 'manage') => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const usePermission = levels === 'manage' ? useCanManageReviews : useCanEditReviews;
  return renderHook(() => usePermission('exp-1'), {
    wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>,
  });
};

beforeEach(() => {
  mockUser = { data: { user: { username: 'reviewer', is_admin: false } }, isError: false };
  mockActiveWorkspace = 'ws-a';
  mockWorkspacesEnabled = true;
  mockWorkspacesLoading = false;
  jest.mocked(AccountApi.getUserPermission).mockReset().mockResolvedValue({ allowed: true, permission: 'MANAGE' });
});

it('uses the server effective permission, including role/default/admin resolution', async () => {
  const { result } = renderPermission();
  await waitFor(() => expect(result.current).toBe(true));
  expect(AccountApi.getUserPermission).toHaveBeenCalledWith('reviewer', 'experiment', 'exp-1');
});

it('grants on a missing auth endpoint only', () => {
  mockUser = { isError: true, error: new NotFoundError({}) };
  expect(renderPermission().result.current).toBe(true);
  expect(AccountApi.getUserPermission).not.toHaveBeenCalled();
});

it.each([new Error('403'), new Error('500'), new Error('network')])('auth lookup failures deny: %s', (error) => {
  mockUser = { isError: true, error };
  expect(renderPermission().result.current).toBe(false);
});

it('denies while identity is pending', () => {
  mockUser = { isLoading: true };
  expect(renderPermission().result.current).toBe(false);
  expect(AccountApi.getUserPermission).not.toHaveBeenCalled();
});

it('denies until effective permissions resolve', async () => {
  let resolve!: (value: { allowed: boolean; permission: string }) => void;
  jest.mocked(AccountApi.getUserPermission).mockReturnValue(
    new Promise((done) => {
      resolve = done;
    }),
  );
  const { result } = renderPermission();
  expect(result.current).toBe(false);
  resolve({ allowed: true, permission: 'MANAGE' });
  await waitFor(() => expect(result.current).toBe(true));
});

it.each([new Error('403'), new Error('500'), new Error('network'), new NotFoundError({})])(
  'permission lookup failures deny: %s',
  async (error) => {
    jest.mocked(AccountApi.getUserPermission).mockRejectedValue(error);
    const { result } = renderPermission();
    await waitFor(() => expect(AccountApi.getUserPermission).toHaveBeenCalled());
    expect(result.current).toBe(false);
  },
);

it.each(['READ', 'USE', 'NO_PERMISSIONS'])(
  'insufficient effective permission %s denies editing',
  async (permission) => {
    jest.mocked(AccountApi.getUserPermission).mockResolvedValue({ allowed: permission === 'USE', permission });
    const { result } = renderPermission('edit');
    await waitFor(() => expect(AccountApi.getUserPermission).toHaveBeenCalled());
    expect(result.current).toBe(false);
  },
);

it('EDIT permits editing but not management', async () => {
  jest.mocked(AccountApi.getUserPermission).mockResolvedValue({ allowed: true, permission: 'EDIT' });
  const edit = renderPermission('edit');
  await waitFor(() => expect(edit.result.current).toBe(true));
  const manage = renderPermission();
  await waitFor(() => expect(AccountApi.getUserPermission).toHaveBeenCalledTimes(2));
  expect(manage.result.current).toBe(false);
});

it('does not reuse a grant after switching workspaces', async () => {
  const { result, rerender } = renderPermission();
  await waitFor(() => expect(result.current).toBe(true));
  jest.mocked(AccountApi.getUserPermission).mockResolvedValue({ allowed: false, permission: 'NO_PERMISSIONS' });
  mockActiveWorkspace = 'ws-other';
  rerender();
  expect(result.current).toBe(false);
  await waitFor(() => expect(AccountApi.getUserPermission).toHaveBeenCalledTimes(2));
  expect(result.current).toBe(false);
});

it.each([true, false])('denies unresolved workspace (enabled=%s)', (enabled) => {
  mockWorkspacesEnabled = enabled;
  mockWorkspacesLoading = !enabled;
  mockActiveWorkspace = null;
  expect(renderPermission().result.current).toBe(false);
  expect(AccountApi.getUserPermission).not.toHaveBeenCalled();
});

it('queries normally when workspaces are definitively disabled', async () => {
  mockWorkspacesEnabled = false;
  mockActiveWorkspace = null;
  const { result } = renderPermission();
  await waitFor(() => expect(result.current).toBe(true));
});
