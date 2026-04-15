import { describe, jest, beforeEach, it, expect } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';

import { QueryClientProvider, QueryClient } from '@databricks/web-shared/query-client';

import { useWorkspaces } from './useWorkspaces';
import { fetchAPI } from '../../common/utils/FetchUtils';

jest.mock('../../common/utils/FetchUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/FetchUtils')>('../../common/utils/FetchUtils'),
  fetchAPI: jest.fn(),
}));

const fetchAPIMock = jest.mocked(fetchAPI);

describe('useWorkspaces', () => {
  const wrapper = ({ children }: { children: React.ReactNode }) => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('returns empty workspaces array when disabled', () => {
    const { result } = renderHook(() => useWorkspaces(false), { wrapper });

    // When disabled, query doesn't run
    expect(result.current.workspaces).toEqual([]);
    expect(result.current.isError).toBe(false);
    // Verify fetchAPI was never called since enabled=false
    expect(fetchAPIMock).not.toHaveBeenCalled();
  });

  it('fetches workspaces from API and returns them when enabled', async () => {
    fetchAPIMock.mockResolvedValue({
      workspaces: [
        { name: 'default', description: 'Default workspace' },
        { name: 'team-a', description: 'Team A workspace' },
        { name: 'team-b', description: null },
      ],
    });

    const { result } = renderHook(() => useWorkspaces(true), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.workspaces).toHaveLength(3);
    expect(result.current.workspaces[0]).toEqual({
      name: 'default',
      description: 'Default workspace',
      default_artifact_root: null,
    });
    expect(result.current.workspaces[1]).toEqual({
      name: 'team-a',
      description: 'Team A workspace',
      default_artifact_root: null,
    });
    expect(result.current.workspaces[2]).toEqual({ name: 'team-b', description: null, default_artifact_root: null });
    expect(result.current.isError).toBe(false);
  });

  it('handles loading state correctly', async () => {
    fetchAPIMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          setTimeout(() => resolve({ workspaces: [{ name: 'default' }] }), 100);
        }),
    );

    const { result } = renderHook(() => useWorkspaces(true), { wrapper });

    expect(result.current.isLoading).toBe(true);
    expect(result.current.workspaces).toEqual([]);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.workspaces).toHaveLength(1);
  });

  it('handles API error gracefully', async () => {
    fetchAPIMock.mockRejectedValue(new Error('Internal server error'));

    const { result } = renderHook(() => useWorkspaces(true), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.isError).toBe(true);
    expect(result.current.workspaces).toEqual([]);
  });

  it('filters out invalid workspace objects from response', async () => {
    fetchAPIMock.mockResolvedValue({
      workspaces: [
        { name: 'valid-workspace', description: 'Valid' },
        { notAName: 'invalid' }, // Missing 'name' field
        { name: 123 }, // Invalid type for 'name'
        null, // Null item
        { name: 'another-valid', description: 'Also valid' },
      ],
    });

    const { result } = renderHook(() => useWorkspaces(true), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.workspaces).toHaveLength(2);
    expect(result.current.workspaces[0].name).toBe('valid-workspace');
    expect(result.current.workspaces[1].name).toBe('another-valid');
  });

  it('does not send X-MLFLOW-WORKSPACE header in request', async () => {
    fetchAPIMock.mockResolvedValue({ workspaces: [{ name: 'default' }] });

    const { result } = renderHook(() => useWorkspaces(true), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Verify fetchAPI was called with header that sets X-MLFLOW-WORKSPACE to empty string
    expect(fetchAPIMock).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({
        headers: expect.objectContaining({
          'X-MLFLOW-WORKSPACE': '',
        }),
      }),
    );
  });

  it('allows refetching workspaces', async () => {
    let callCount = 0;
    fetchAPIMock.mockImplementation(() => {
      callCount++;
      if (callCount === 1) {
        return Promise.resolve({ workspaces: [{ name: 'initial' }] });
      }
      return Promise.resolve({ workspaces: [{ name: 'updated' }] });
    });

    const { result } = renderHook(() => useWorkspaces(true), { wrapper });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.workspaces[0].name).toBe('initial');

    // Trigger refetch
    result.current.refetch();

    await waitFor(() => {
      expect(result.current.workspaces[0].name).toBe('updated');
    });

    expect(callCount).toBe(2);
  });
});
