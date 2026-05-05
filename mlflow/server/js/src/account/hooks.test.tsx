import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { AccountApi } from './api';
import {
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
