import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { AdminApi } from './api';
import {
  AdminQueryKeys,
  useAssignRole,
  useGrantUserPermission,
  useRevokeUserPermission,
  useRolesQuery,
  useUnassignRole,
} from './hooks';
import { AccountQueryKeys } from '../account/hooks';

jest.mock('./api', () => ({
  AdminApi: {
    listRoles: jest.fn(),
    assignRole: jest.fn(),
    unassignRole: jest.fn(),
    grantUserPermission: jest.fn(),
    revokeUserPermission: jest.fn(),
  },
}));

const mockedApi = AdminApi as jest.Mocked<typeof AdminApi>;

const makeWrapper = (client: QueryClient) => {
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
};

const freshClient = () => new QueryClient({ defaultOptions: { queries: { retry: false } } });

beforeEach(() => {
  jest.clearAllMocks();
});

describe('useRolesQuery cache-key normalization', () => {
  // The cache key is ``[...AdminQueryKeys.roles, normalized]`` — for a list,
  // ``normalized`` is the input sorted ascending, so callers passing the same
  // set in different order share one cache entry.

  const findRolesKeys = (client: QueryClient): unknown[][] => {
    return client
      .getQueryCache()
      .getAll()
      .map((q) => q.queryKey as unknown[])
      .filter((k) => Array.isArray(k) && k[0] === AdminQueryKeys.roles[0]);
  };

  it('uses ``undefined`` as the trailing segment when called with no argument', async () => {
    mockedApi.listRoles.mockResolvedValueOnce({ roles: [] });
    const client = freshClient();
    const { result } = renderHook(() => useRolesQuery(), { wrapper: makeWrapper(client) });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(findRolesKeys(client)).toEqual([['admin_roles', undefined]]);
  });

  it('preserves a string workspace literal in the key', async () => {
    mockedApi.listRoles.mockResolvedValueOnce({ roles: [] });
    const client = freshClient();
    const { result } = renderHook(() => useRolesQuery('foo'), { wrapper: makeWrapper(client) });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(findRolesKeys(client)).toEqual([['admin_roles', 'foo']]);
  });

  it('canonicalizes a list to sorted form so input order does not affect the cache key', async () => {
    mockedApi.listRoles.mockResolvedValue({ roles: [] });
    const client = freshClient();

    // Mount two readers with the same set in different order — they should
    // share one cache entry under the sorted key.
    const { result: r1 } = renderHook(() => useRolesQuery(['b', 'a']), {
      wrapper: makeWrapper(client),
    });
    await waitFor(() => expect(r1.current.isSuccess).toBe(true));

    const { result: r2 } = renderHook(() => useRolesQuery(['a', 'b']), {
      wrapper: makeWrapper(client),
    });
    await waitFor(() => expect(r2.current.isSuccess).toBe(true));

    // Single cache entry, normalized to sorted form. (Refetch behavior is
    // governed by staleTime, which is unrelated to the cache-key contract.)
    expect(findRolesKeys(client)).toEqual([['admin_roles', ['a', 'b']]]);
  });

  it('skips the fetch when explicitly disabled via options', () => {
    const client = freshClient();
    renderHook(() => useRolesQuery('foo', { enabled: false }), { wrapper: makeWrapper(client) });
    expect(mockedApi.listRoles).not.toHaveBeenCalled();
  });
});

describe('mutation invalidation contract', () => {
  // These hooks fan invalidations to multiple query keys so the admin Users
  // tab (eager-loaded ``users[].roles``), the per-user roles query, and
  // role-detail views all refresh after a single mutation. Pinning the
  // contract here guards the eager-load N+1 fix from regressing.

  const setupSpy = (client: QueryClient) => jest.spyOn(client, 'invalidateQueries');

  it('useAssignRole invalidates roleUsers, per-user userRoles, and the bulk users query', async () => {
    mockedApi.assignRole.mockResolvedValueOnce({} as any);
    const client = freshClient();
    const spy = setupSpy(client);
    const { result } = renderHook(() => useAssignRole(42), { wrapper: makeWrapper(client) });

    await result.current.mutateAsync('alice');

    const keys = spy.mock.calls.map((c) => c[0] as { queryKey?: unknown } | undefined).map((arg) => arg?.queryKey);
    expect(keys).toEqual(
      expect.arrayContaining([AdminQueryKeys.roleUsers(42), AccountQueryKeys.userRoles('alice'), AdminQueryKeys.users]),
    );
  });

  it('useUnassignRole invalidates the same three query sets', async () => {
    mockedApi.unassignRole.mockResolvedValueOnce(undefined);
    const client = freshClient();
    const spy = setupSpy(client);
    const { result } = renderHook(() => useUnassignRole(42), { wrapper: makeWrapper(client) });

    await result.current.mutateAsync('alice');

    const keys = spy.mock.calls.map((c) => c[0] as { queryKey?: unknown } | undefined).map((arg) => arg?.queryKey);
    expect(keys).toEqual(
      expect.arrayContaining([AdminQueryKeys.roleUsers(42), AccountQueryKeys.userRoles('alice'), AdminQueryKeys.users]),
    );
  });

  it('useGrantUserPermission invalidates per-user userRoles and userPermissions', async () => {
    mockedApi.grantUserPermission.mockResolvedValueOnce(undefined);
    const client = freshClient();
    const spy = setupSpy(client);
    const { result } = renderHook(() => useGrantUserPermission(), { wrapper: makeWrapper(client) });

    await result.current.mutateAsync({
      resource_type: 'experiment',
      resource_id: '42',
      username: 'alice',
      permission: 'EDIT',
    });

    const keys = spy.mock.calls.map((c) => c[0] as { queryKey?: unknown } | undefined).map((arg) => arg?.queryKey);
    expect(keys).toEqual(
      expect.arrayContaining([AccountQueryKeys.userRoles('alice'), AdminQueryKeys.userPermissions('alice')]),
    );
  });

  it('useRevokeUserPermission invalidates per-user userRoles and userPermissions', async () => {
    mockedApi.revokeUserPermission.mockResolvedValueOnce(undefined);
    const client = freshClient();
    const spy = setupSpy(client);
    const { result } = renderHook(() => useRevokeUserPermission(), { wrapper: makeWrapper(client) });

    await result.current.mutateAsync({
      resource_type: 'experiment',
      resource_id: '42',
      username: 'alice',
    });

    const keys = spy.mock.calls.map((c) => c[0] as { queryKey?: unknown } | undefined).map((arg) => arg?.queryKey);
    expect(keys).toEqual(
      expect.arrayContaining([AccountQueryKeys.userRoles('alice'), AdminQueryKeys.userPermissions('alice')]),
    );
  });
});
