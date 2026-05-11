import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { AccountApi } from './api';
import {
  useCurrentUserAdminWorkspaces,
  useCurrentUserIsWorkspaceAdmin,
  useCurrentUserQuery,
  useIsAuthAvailable,
  useIsBasicAuth,
  useMyPermissionsQuery,
  useUserRolesQuery,
} from './hooks';

jest.mock('./api', () => ({
  AccountApi: {
    getCurrentUser: jest.fn(),
    listUserRoles: jest.fn(),
    listMyPermissions: jest.fn(),
    updatePassword: jest.fn(),
  },
}));

const mockedApi = AccountApi as jest.Mocked<typeof AccountApi>;

const makeWrapper = () => {
  // Each test gets a fresh QueryClient so cache state doesn't bleed between
  // assertions on hook-state transitions.
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
};

beforeEach(() => {
  jest.clearAllMocks();
});

describe('useCurrentUserQuery', () => {
  it('resolves with the backend payload', async () => {
    mockedApi.getCurrentUser.mockResolvedValueOnce({
      user: { id: 1, username: 'pat', is_admin: false },
      is_basic_auth: true,
    });

    const { result } = renderHook(() => useCurrentUserQuery(), { wrapper: makeWrapper() });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.user.username).toBe('pat');
    expect(result.current.data?.is_basic_auth).toBe(true);
    expect(mockedApi.getCurrentUser).toHaveBeenCalledTimes(1);
  });
});

describe('useIsAuthAvailable', () => {
  it('returns true while loading - avoids flicker on auth-gated UI', () => {
    // Pending promise: query stays in loading.
    mockedApi.getCurrentUser.mockReturnValueOnce(new Promise(() => {}));
    const { result } = renderHook(() => useIsAuthAvailable(), { wrapper: makeWrapper() });
    expect(result.current).toBe(true);
  });

  it('returns true when /users/current resolves with a user', async () => {
    mockedApi.getCurrentUser.mockResolvedValueOnce({
      user: { id: 1, username: 'pat', is_admin: false },
    });
    const { result } = renderHook(() => useIsAuthAvailable(), { wrapper: makeWrapper() });
    await waitFor(() => expect(result.current).toBe(true));
  });

  it('returns false when /users/current errors (auth not configured)', async () => {
    mockedApi.getCurrentUser.mockRejectedValueOnce(new Error('404'));
    const { result } = renderHook(() => useIsAuthAvailable(), { wrapper: makeWrapper() });
    await waitFor(() => expect(result.current).toBe(false));
  });
});

describe('useIsBasicAuth', () => {
  it('returns true only after the response says is_basic_auth: true', async () => {
    mockedApi.getCurrentUser.mockResolvedValueOnce({
      user: { id: 1, username: 'pat', is_admin: false },
      is_basic_auth: true,
    });
    const { result } = renderHook(() => useIsBasicAuth(), { wrapper: makeWrapper() });
    expect(result.current).toBe(false); // pending
    await waitFor(() => expect(result.current).toBe(true));
  });

  it('returns false when is_basic_auth is missing or false (custom auth plugin)', async () => {
    mockedApi.getCurrentUser.mockResolvedValueOnce({
      user: { id: 1, username: 'pat', is_admin: false },
      is_basic_auth: false,
    });
    const { result } = renderHook(() => useIsBasicAuth(), { wrapper: makeWrapper() });
    await waitFor(() => expect(result.current).toBe(false));
    expect(mockedApi.listMyPermissions).not.toHaveBeenCalled();
  });
});

describe('useMyPermissionsQuery (gated on known username)', () => {
  it('does not fire until the current-user query resolves with a username', async () => {
    mockedApi.getCurrentUser.mockReturnValueOnce(new Promise(() => {}));
    renderHook(() => useMyPermissionsQuery(), { wrapper: makeWrapper() });
    // No call while username is unknown.
    expect(mockedApi.listMyPermissions).not.toHaveBeenCalled();
  });

  it('fires once the current-user query yields a username', async () => {
    mockedApi.getCurrentUser.mockResolvedValueOnce({
      user: { id: 1, username: 'pat', is_admin: false },
    });
    mockedApi.listMyPermissions.mockResolvedValueOnce({ permissions: [] });

    const { result } = renderHook(() => useMyPermissionsQuery(), { wrapper: makeWrapper() });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(mockedApi.listMyPermissions).toHaveBeenCalledTimes(1);
  });
});

describe('useUserRolesQuery (gated on truthy username)', () => {
  it('does not fire when username is empty', () => {
    renderHook(() => useUserRolesQuery(''), { wrapper: makeWrapper() });
    expect(mockedApi.listUserRoles).not.toHaveBeenCalled();
  });

  it('fires when username is provided', async () => {
    mockedApi.listUserRoles.mockResolvedValueOnce({ roles: [] });
    const { result } = renderHook(() => useUserRolesQuery('pat'), { wrapper: makeWrapper() });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(mockedApi.listUserRoles).toHaveBeenCalledWith('pat');
  });
});

describe('useCurrentUserIsWorkspaceAdmin', () => {
  // (workspace, *, MANAGE) → workspace-admin role.
  const wsAdminRole = {
    id: 1,
    name: 'wp-admin-foo',
    workspace: 'foo',
    description: '',
    permissions: [{ id: 10, role_id: 1, resource_type: 'workspace', resource_pattern: '*', permission: 'MANAGE' }],
  };
  const memberRole = {
    id: 2,
    name: 'foo-member',
    workspace: 'foo',
    description: '',
    permissions: [{ id: 11, role_id: 2, resource_type: 'experiment', resource_pattern: '*', permission: 'READ' }],
  };

  it('returns false while the queries are still loading', () => {
    mockedApi.getCurrentUser.mockReturnValueOnce(new Promise(() => {}));
    const { result } = renderHook(() => useCurrentUserIsWorkspaceAdmin(), {
      wrapper: makeWrapper(),
    });
    expect(result.current).toBe(false);
    expect(mockedApi.listUserRoles).not.toHaveBeenCalled();
  });

  it('returns true when the user has at least one workspace-admin role', async () => {
    mockedApi.getCurrentUser.mockResolvedValueOnce({
      user: { id: 1, username: 'pat', is_admin: false },
      is_basic_auth: true,
    });
    mockedApi.listUserRoles.mockResolvedValueOnce({ roles: [memberRole, wsAdminRole] });

    const { result } = renderHook(() => useCurrentUserIsWorkspaceAdmin(), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current).toBe(true));
    expect(mockedApi.listUserRoles).toHaveBeenCalledWith('pat');
  });

  it('returns false when the user holds only non-admin roles', async () => {
    mockedApi.getCurrentUser.mockResolvedValueOnce({
      user: { id: 1, username: 'pat', is_admin: false },
      is_basic_auth: true,
    });
    mockedApi.listUserRoles.mockResolvedValueOnce({ roles: [memberRole] });

    const { result } = renderHook(() => useCurrentUserIsWorkspaceAdmin(), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(mockedApi.listUserRoles).toHaveBeenCalledTimes(1));
    expect(result.current).toBe(false);
  });

  it('returns false when the user has no roles at all', async () => {
    mockedApi.getCurrentUser.mockResolvedValueOnce({
      user: { id: 1, username: 'pat', is_admin: false },
      is_basic_auth: true,
    });
    mockedApi.listUserRoles.mockResolvedValueOnce({ roles: [] });

    const { result } = renderHook(() => useCurrentUserIsWorkspaceAdmin(), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(mockedApi.listUserRoles).toHaveBeenCalledTimes(1));
    expect(result.current).toBe(false);
  });
});

describe('useCurrentUserAdminWorkspaces', () => {
  const adminFooRole = {
    id: 1,
    name: 'admin-foo',
    workspace: 'foo',
    description: '',
    permissions: [{ id: 10, role_id: 1, resource_type: 'workspace', resource_pattern: '*', permission: 'MANAGE' }],
  };
  const adminBarRole = {
    id: 2,
    name: 'admin-bar',
    workspace: 'bar',
    description: '',
    permissions: [{ id: 20, role_id: 2, resource_type: 'workspace', resource_pattern: '*', permission: 'MANAGE' }],
  };
  const memberRole = {
    id: 3,
    name: 'foo-member',
    workspace: 'foo',
    description: '',
    permissions: [{ id: 30, role_id: 3, resource_type: 'experiment', resource_pattern: '*', permission: 'READ' }],
  };

  it('returns the set of workspaces where the user is a workspace admin', async () => {
    mockedApi.getCurrentUser.mockResolvedValueOnce({
      user: { id: 1, username: 'pat', is_admin: false },
      is_basic_auth: true,
    });
    mockedApi.listUserRoles.mockResolvedValueOnce({
      roles: [adminFooRole, memberRole, adminBarRole],
    });

    const { result } = renderHook(() => useCurrentUserAdminWorkspaces(), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.size).toBe(2));
    expect(result.current.has('foo')).toBe(true);
    expect(result.current.has('bar')).toBe(true);
  });

  it('returns an empty set when the user has only non-admin roles', async () => {
    mockedApi.getCurrentUser.mockResolvedValueOnce({
      user: { id: 1, username: 'pat', is_admin: false },
      is_basic_auth: true,
    });
    mockedApi.listUserRoles.mockResolvedValueOnce({ roles: [memberRole] });

    const { result } = renderHook(() => useCurrentUserAdminWorkspaces(), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(mockedApi.listUserRoles).toHaveBeenCalledTimes(1));
    expect(result.current.size).toBe(0);
  });

  it('skips listUserRoles for platform admins (they short-circuit elsewhere)', async () => {
    mockedApi.getCurrentUser.mockResolvedValueOnce({
      user: { id: 1, username: 'admin', is_admin: true },
      is_basic_auth: true,
    });

    const { result } = renderHook(() => useCurrentUserAdminWorkspaces(), {
      wrapper: makeWrapper(),
    });
    // Wait for currentUserQuery to resolve, then assert no roles fetch fired.
    await waitFor(() => expect(mockedApi.getCurrentUser).toHaveBeenCalled());
    expect(mockedApi.listUserRoles).not.toHaveBeenCalled();
    expect(result.current.size).toBe(0);
  });

  it('returns an empty set while the queries are still loading', () => {
    mockedApi.getCurrentUser.mockReturnValueOnce(new Promise(() => {}));
    const { result } = renderHook(() => useCurrentUserAdminWorkspaces(), {
      wrapper: makeWrapper(),
    });
    expect(result.current.size).toBe(0);
  });
});
