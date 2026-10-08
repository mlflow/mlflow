import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { AccountApi } from '../../account/api';
import { useCanEditServerSettings } from './useCanEditServerSettings';

jest.mock('../../account/api', () => ({
  AccountApi: {
    getCurrentUser: jest.fn(),
  },
}));

const mockedApi = AccountApi as jest.Mocked<typeof AccountApi>;

const makeWrapper = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
};

const renderForUser = (isLocalServer: boolean) =>
  renderHook(() => useCanEditServerSettings(isLocalServer), { wrapper: makeWrapper() });

beforeEach(() => {
  jest.clearAllMocks();
});

describe('useCanEditServerSettings', () => {
  it('is false while the current user is loading', () => {
    mockedApi.getCurrentUser.mockReturnValueOnce(new Promise(() => {}));

    const { result } = renderForUser(true);

    expect(result.current).toBe(false);
  });

  it('is true for an admin on the server host', async () => {
    mockedApi.getCurrentUser.mockResolvedValueOnce({
      user: { id: 1, username: 'admin', is_admin: true },
      is_basic_auth: true,
    });

    const { result } = renderForUser(true);

    await waitFor(() => expect(result.current).toBe(true));
  });

  it('is false for a non-admin on the server host', async () => {
    mockedApi.getCurrentUser.mockResolvedValueOnce({
      user: { id: 2, username: 'pat', is_admin: false },
      is_basic_auth: true,
    });

    const { result } = renderForUser(true);

    await waitFor(() => expect(mockedApi.getCurrentUser).toHaveBeenCalled());
    await waitFor(() => expect(result.current).toBe(false));
  });

  it('is true on the server host when the server has no auth', async () => {
    // Without auth, the current-user endpoint fails.
    mockedApi.getCurrentUser.mockRejectedValueOnce(new Error('Not found'));

    const { result } = renderForUser(true);

    await waitFor(() => expect(result.current).toBe(true));
  });

  it('is false for a remote caller, even an admin', async () => {
    mockedApi.getCurrentUser.mockResolvedValueOnce({
      user: { id: 1, username: 'admin', is_admin: true },
      is_basic_auth: true,
    });

    const { result } = renderForUser(false);

    await waitFor(() => expect(mockedApi.getCurrentUser).toHaveBeenCalled());
    expect(result.current).toBe(false);
  });
});
