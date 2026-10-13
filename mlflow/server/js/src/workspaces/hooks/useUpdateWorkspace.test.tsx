import { describe, beforeEach, expect, jest, test } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { fetchAPI } from '../../common/utils/FetchUtils';
import { useUpdateWorkspace } from './useUpdateWorkspace';

jest.mock('../../common/utils/FetchUtils', () => ({
  fetchAPI: jest.fn(),
  getAjaxUrl: jest.fn((url: string) => url),
  HTTPMethods: { GET: 'GET', POST: 'POST', PATCH: 'PATCH', DELETE: 'DELETE' },
}));

const fetchAPIMock = jest.mocked(fetchAPI);

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
      mutations: {
        retry: false,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe('useUpdateWorkspace', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    fetchAPIMock.mockResolvedValue({
      workspace: {
        name: 'team-a',
      },
    } as any);
  });

  test('serializes trace archival config in patch request', async () => {
    const { result } = renderHook(() => useUpdateWorkspace(), { wrapper: createWrapper() });

    await result.current.mutateAsync({
      name: 'team-a',
      description: 'Team A',
      default_artifact_root: 's3://artifacts/team-a',
      trace_archival_config: {
        location: 's3://archive/team-a',
        retention: '30d',
      },
    });

    expect(fetchAPIMock).toHaveBeenCalledWith('ajax-api/3.0/mlflow/workspaces/team-a', {
      method: 'PATCH',
      body: JSON.stringify({
        description: 'Team A',
        default_artifact_root: 's3://artifacts/team-a',
        trace_archival_config: {
          location: 's3://archive/team-a',
          retention: '30d',
        },
      }),
      headers: { 'X-MLFLOW-WORKSPACE': '' },
    });
  });

  test('sends empty strings to clear trace archival overrides', async () => {
    const { result } = renderHook(() => useUpdateWorkspace(), { wrapper: createWrapper() });

    await result.current.mutateAsync({
      name: 'team-a',
      trace_archival_config: {
        location: '',
        retention: '',
      },
    });

    expect(fetchAPIMock).toHaveBeenCalledWith('ajax-api/3.0/mlflow/workspaces/team-a', {
      method: 'PATCH',
      body: JSON.stringify({
        trace_archival_config: {
          location: '',
          retention: '',
        },
      }),
      headers: { 'X-MLFLOW-WORKSPACE': '' },
    });
  });
});
