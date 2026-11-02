import { describe, jest, test, expect } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { useAssignableUsersQuery } from './useAssignableUsersQuery';
import { AdminApi } from '../../../../admin/api';
import { setActiveWorkspace } from '../../../../workspaces/utils/WorkspaceUtils';

jest.mock('../../../../admin/api', () => ({
  AdminApi: { listUsers: jest.fn() },
}));

const mockedListUsers = jest.mocked(AdminApi.listUsers);

const createWrapper = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('useAssignableUsersQuery', () => {
  test('re-fetches the roster when the active workspace changes', async () => {
    mockedListUsers.mockResolvedValueOnce({ users: [{ id: 1, username: 'pat', is_admin: false, roles: [] }] });
    mockedListUsers.mockResolvedValueOnce({ users: [{ id: 2, username: 'alex', is_admin: false, roles: [] }] });
    setActiveWorkspace('wsA');
    try {
      const { result } = renderHook(() => useAssignableUsersQuery({ enabled: true }), {
        wrapper: createWrapper(),
      });
      await waitFor(() => expect(result.current.users[0]?.username).toBe('pat'));

      act(() => setActiveWorkspace('wsB'));
      await waitFor(() => expect(result.current.users[0]?.username).toBe('alex'));
      expect(mockedListUsers.mock.calls).toEqual([['wsA'], ['wsB']]);
    } finally {
      setActiveWorkspace(null);
    }
  });
});
