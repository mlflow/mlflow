import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { useExperimentListQuery } from './useExperimentListQuery';
import { MlflowService } from '../../../sdk/MlflowService';
import type { SearchExperimentsApiResponse } from '../../../types';
import type { TagFilter } from './useTagsFilter';

// Mock MlflowService
jest.mock('../../../sdk/MlflowService', () => ({
  MlflowService: {
    searchExperiments: jest.fn(),
  },
}));

// Mock localStorage for useLocalStorage hook
const localStorageMock = (() => {
  let store: Record<string, string> = {};

  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

const mockSearchExperiments = MlflowService.searchExperiments as jest.MockedFunction<
  typeof MlflowService.searchExperiments
>;

describe('useExperimentListQuery', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    // Create a fresh QueryClient for each test
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    // Clear localStorage before each test
    localStorageMock.clear();

    // Reset mock
    mockSearchExperiments.mockReset();
  });

  const createWrapper = () => {
    // eslint-disable-next-line react/display-name
    return ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };

  const createMockResponse = (experimentIds: string[], nextPageToken?: string): SearchExperimentsApiResponse => ({
    experiments: experimentIds.map((id) => ({
      experimentId: id,
      name: `Experiment ${id}`,
      artifactLocation: `/artifacts/${id}`,
      lifecycleStage: 'active',
      lastUpdateTime: Date.now(),
      creationTime: Date.now(),
      tags: [],
      allowedActions: [],
    })),
    next_page_token: nextPageToken,
  });

  describe('pagination reset on filter changes', () => {
    it('resets pagination when searchFilter changes', async () => {
      // Mock responses for all possible query variations
      mockSearchExperiments.mockImplementation(async (params) => {
        // Extract the filter to determine which response to return
        const filterParam = params.find((p: any) => p?.[0] === 'filter')?.[1];
        const pageToken = params.find((p: any) => p?.[0] === 'page_token')?.[1];

        if (filterParam?.includes('test')) {
          // After filter change - return different data
          return createMockResponse(['7'], undefined);
        } else if (pageToken === 'page_2_token') {
          // Second page
          return createMockResponse(['4', '5', '6'], 'page_3_token');
        } else {
          // First page
          return createMockResponse(['1', '2', '3'], 'page_2_token');
        }
      });

      // Initial render without search filter
      const { result, rerender } = renderHook(
        ({ searchFilter }: { searchFilter?: string }) => useExperimentListQuery({ searchFilter }),
        {
          wrapper: createWrapper(),
          initialProps: { searchFilter: undefined as string | undefined },
        },
      );

      // Wait for initial query to complete
      await waitFor(() => expect(result.current.isLoading).toBe(false));

      // Verify we have the first page of experiments
      expect(result.current.data).toHaveLength(3);
      expect(result.current.hasNextPage).toBe(true);
      expect(result.current.hasPreviousPage).toBe(false);

      // Navigate to next page
      act(() => {
        result.current.onNextPage();
      });

      // Wait for next page to load
      await waitFor(() => {
        expect(result.current.data?.[0].experimentId).toBe('4');
      });
      expect(result.current.hasPreviousPage).toBe(true);

      // Now change the search filter - this should reset pagination
      rerender({ searchFilter: 'test' });

      // Wait for the query to complete with new filter
      await waitFor(() => {
        expect(result.current.data?.[0].experimentId).toBe('7');
      });

      // Verify pagination was reset
      expect(result.current.data).toHaveLength(1);
      expect(result.current.hasPreviousPage).toBe(false); // Should be back to first page
      expect(result.current.hasNextPage).toBe(false);
    });

    it('resets pagination when tagsFilter changes', async () => {
      // Mock responses based on tags filter
      mockSearchExperiments.mockImplementation(async (params) => {
        const filterParam = params.find((p: any) => p?.[0] === 'filter')?.[1];
        const pageToken = params.find((p: any) => p?.[0] === 'page_token')?.[1];

        if (filterParam?.includes('tags')) {
          // After tags filter change
          return createMockResponse(['7'], undefined);
        } else if (pageToken === 'page_2_token') {
          // Second page
          return createMockResponse(['4', '5', '6'], 'page_3_token');
        } else {
          // First page
          return createMockResponse(['1', '2', '3'], 'page_2_token');
        }
      });

      const initialTagsFilter: TagFilter[] | undefined = [];
      const updatedTagsFilter: TagFilter[] = [{ key: 'env', operator: 'IS', value: 'prod' }];

      // Initial render without tags filter
      const { result, rerender } = renderHook(
        ({ tagsFilter }: { tagsFilter?: TagFilter[] }) => useExperimentListQuery({ tagsFilter }),
        {
          wrapper: createWrapper(),
          initialProps: { tagsFilter: initialTagsFilter as TagFilter[] | undefined },
        },
      );

      // Wait for initial query to complete
      await waitFor(() => expect(result.current.isLoading).toBe(false));

      // Navigate to next page
      act(() => {
        result.current.onNextPage();
      });

      // Wait for next page to load
      await waitFor(() => {
        expect(result.current.data?.[0].experimentId).toBe('4');
      });

      // Now change the tags filter - this should reset pagination
      rerender({ tagsFilter: updatedTagsFilter });

      // Wait for the query to complete with new filter
      await waitFor(() => {
        expect(result.current.data?.[0].experimentId).toBe('7');
      });

      // Verify pagination was reset
      expect(result.current.hasPreviousPage).toBe(false); // Should be back to first page
    });

    it('resets pagination when sorting changes', async () => {
      // Mock responses based on sorting order
      mockSearchExperiments.mockImplementation(async (params) => {
        const orderBy = params.find((p: any) => p?.[0] === 'order_by')?.[1];
        const pageToken = params.find((p: any) => p?.[0] === 'page_token')?.[1];

        if (orderBy?.includes('name ASC')) {
          // After sorting change
          return createMockResponse(['10', '11', '12'], undefined);
        } else if (pageToken === 'page_2_token') {
          // Second page
          return createMockResponse(['4', '5', '6'], 'page_3_token');
        } else {
          // First page
          return createMockResponse(['1', '2', '3'], 'page_2_token');
        }
      });

      // Initial render
      const { result } = renderHook(() => useExperimentListQuery(), {
        wrapper: createWrapper(),
      });

      // Wait for initial query to complete
      await waitFor(() => expect(result.current.isLoading).toBe(false));

      // Navigate to next page
      act(() => {
        result.current.onNextPage();
      });

      // Wait for next page to load
      await waitFor(() => {
        expect(result.current.data?.[0].experimentId).toBe('4');
      });

      // Now change the sorting - this should reset pagination
      act(() => {
        result.current.setSorting([{ id: 'name', desc: false }]);
      });

      // Wait for the query to complete with new sorting
      await waitFor(() => {
        expect(result.current.data?.[0].experimentId).toBe('10');
      });

      // Verify pagination was reset
      expect(result.current.hasPreviousPage).toBe(false); // Should be back to first page
    });
  });

  describe('pagination controls', () => {
    it('allows forward and backward pagination', async () => {
      mockSearchExperiments
        .mockResolvedValueOnce(createMockResponse(['1', '2', '3'], 'page_2_token'))
        .mockResolvedValueOnce(createMockResponse(['4', '5', '6'], 'page_3_token'))
        .mockResolvedValueOnce(createMockResponse(['1', '2', '3'], 'page_2_token'));

      const { result } = renderHook(() => useExperimentListQuery(), {
        wrapper: createWrapper(),
      });

      // Wait for initial query
      await waitFor(() => expect(result.current.isLoading).toBe(false));

      // First page
      expect(result.current.data?.[0].experimentId).toBe('1');
      expect(result.current.hasNextPage).toBe(true);
      expect(result.current.hasPreviousPage).toBe(false);

      // Go to next page
      act(() => {
        result.current.onNextPage();
      });

      await waitFor(() => expect(result.current.data?.[0].experimentId).toBe('4'));
      expect(result.current.hasNextPage).toBe(true);
      expect(result.current.hasPreviousPage).toBe(true);

      // Go back to previous page
      act(() => {
        result.current.onPreviousPage();
      });

      await waitFor(() => expect(result.current.data?.[0].experimentId).toBe('1'));
      expect(result.current.hasPreviousPage).toBe(false);
    });

    it('resets pagination when page size changes', async () => {
      mockSearchExperiments
        .mockResolvedValueOnce(createMockResponse(['1', '2', '3'], 'page_2_token'))
        .mockResolvedValueOnce(createMockResponse(['4', '5', '6'], undefined))
        .mockResolvedValueOnce(createMockResponse(['1', '2', '3', '4', '5'], undefined));

      const { result } = renderHook(() => useExperimentListQuery(), {
        wrapper: createWrapper(),
      });

      // Wait for initial query
      await waitFor(() => expect(result.current.isLoading).toBe(false));

      // Go to next page
      act(() => {
        result.current.onNextPage();
      });

      await waitFor(() => expect(result.current.data?.[0].experimentId).toBe('4'));
      expect(result.current.hasPreviousPage).toBe(true);

      // Change page size - should reset pagination
      act(() => {
        result.current.pageSizeSelect.onChange(50);
      });

      await waitFor(() => expect(result.current.data).toHaveLength(5));
      expect(result.current.hasPreviousPage).toBe(false); // Should be back to first page
    });
  });

  describe('API query construction', () => {
    it('includes search filter in API call', async () => {
      mockSearchExperiments.mockResolvedValueOnce(createMockResponse(['1'], undefined));

      renderHook(() => useExperimentListQuery({ searchFilter: 'my-experiment' }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => expect(mockSearchExperiments).toHaveBeenCalled());

      const apiCallData = mockSearchExperiments.mock.calls[0][0];
      const filterParam = apiCallData.find((param: [string, string]) => param?.[0] === 'filter');

      expect(filterParam).toBeDefined();
      expect(filterParam?.[1]).toContain("name ILIKE '%my-experiment%'");
    });

    it('includes tag filters in API call', async () => {
      mockSearchExperiments.mockResolvedValueOnce(createMockResponse(['1'], undefined));

      const tagsFilter: TagFilter[] = [
        { key: 'env', operator: 'IS', value: 'prod' },
        { key: 'team', operator: 'CONTAINS', value: 'ml' },
      ];

      renderHook(() => useExperimentListQuery({ tagsFilter }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => expect(mockSearchExperiments).toHaveBeenCalled());

      const apiCallData = mockSearchExperiments.mock.calls[0][0];
      const filterParam = apiCallData.find((param: [string, string]) => param?.[0] === 'filter');

      expect(filterParam).toBeDefined();
      expect(filterParam?.[1]).toContain("tags.`env` = 'prod'");
      expect(filterParam?.[1]).toContain("tags.`team` ILIKE '%ml%'");
    });

    it('includes page token in API call when paginating', async () => {
      mockSearchExperiments
        .mockResolvedValueOnce(createMockResponse(['1', '2', '3'], 'page_2_token'))
        .mockResolvedValueOnce(createMockResponse(['4', '5', '6'], undefined));

      const { result } = renderHook(() => useExperimentListQuery(), {
        wrapper: createWrapper(),
      });

      // Wait for initial query to complete
      await waitFor(() => expect(result.current.isLoading).toBe(false));

      // First call should not have page token
      let apiCallData = mockSearchExperiments.mock.calls[0][0];
      let pageTokenParam = apiCallData.find((param: [string, string]) => param?.[0] === 'page_token');
      expect(pageTokenParam).toBeUndefined();

      // Go to next page
      act(() => {
        result.current.onNextPage();
      });

      // Wait for second query to complete
      await waitFor(() => {
        expect(result.current.data?.[0].experimentId).toBe('4');
      });

      // Second call should have page token
      expect(mockSearchExperiments).toHaveBeenCalledTimes(2);
      apiCallData = mockSearchExperiments.mock.calls[1][0];
      pageTokenParam = apiCallData.find((param: [string, string]) => param?.[0] === 'page_token');
      expect(pageTokenParam).toBeDefined();
      expect(pageTokenParam?.[1]).toBe('page_2_token');
    });
  });
});
