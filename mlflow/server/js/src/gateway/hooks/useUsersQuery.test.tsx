import { describe, jest, test, expect } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { useUsersQuery } from './useUsersQuery';
import { GatewayApi } from '../api';
import { setActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';

jest.mock('../api', () => ({
  GatewayApi: { listUsers: jest.fn() },
}));

const mockedListUsers = jest.mocked(GatewayApi.listUsers);

const createWrapper = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useUsersQuery', () => {
  test('re-fetches the roster when the active workspace changes', async () => {
    mockedListUsers.mockResolvedValueOnce({ users: [{ id: 1, username: 'pat' }] });
    mockedListUsers.mockResolvedValueOnce({ users: [{ id: 2, username: 'alex' }] });
    setActiveWorkspace('wsA');
    try {
      const { result } = renderHook(() => useUsersQuery(), { wrapper: createWrapper() });
      await waitFor(() => expect(result.current.data[0]?.username).toBe('pat'));

      act(() => setActiveWorkspace('wsB'));
      await waitFor(() => expect(result.current.data[0]?.username).toBe('alex'));
      expect(mockedListUsers.mock.calls).toEqual([['wsA'], ['wsB']]);
    } finally {
      setActiveWorkspace(null);
    }
  });
});
