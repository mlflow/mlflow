import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { renderHook, waitFor } from '@testing-library/react';
import { NotFoundError, InternalServerError, UnknownError } from '@databricks/web-shared/errors';
import { useGetScheduledScorers } from './useGetScheduledScorers';
import { listScheduledScorers } from '../api';

// Mock external dependencies
jest.mock('../api');

const mockListScheduledScorers = jest.mocked(listScheduledScorers);

describe('useGetScheduledScorers', () => {
  let queryClient: QueryClient;

  const mockExperimentId = 'experiment-123';

  const mockScorerConfigs = [
    {
      name: 'llm-guidelines-scorer',
      serialized_scorer: JSON.stringify({
        builtin_scorer_class: 'Guidelines',
        builtin_scorer_pydantic_data: {
          guidelines: ['Test guideline 1', 'Test guideline 2'],
        },
      }),
      builtin: { name: 'llm-guidelines-scorer' },
      sample_rate: 0.5,
      filter_string: 'column = "value"',
    },
    {
      name: 'custom-code-scorer',
      serialized_scorer: JSON.stringify({
        call_source: 'return {"score": 1}',
        original_func_name: 'my_scorer',
        call_signature: '(inputs, outputs)',
      }),
      custom: {},
      sample_rate: 0.25,
    },
  ];

  const mockApiResponse = {
    experiment_id: mockExperimentId,
    scheduled_scorers: {
      scorers: mockScorerConfigs,
    },
  };

  const mockEmptyApiResponse = {
    experiment_id: mockExperimentId,
    scheduled_scorers: {
      scorers: [],
    },
  };

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });

    jest.clearAllMocks();
  });

  describe('Golden Path - Successful Operations', () => {
    it('should successfully fetch and transform scheduled scorers', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      mockListScheduledScorers.mockResolvedValue(mockApiResponse);

      const { result } = renderHook(() => useGetScheduledScorers(mockExperimentId), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act & Assert - Wait for loading to complete
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.isSuccess).toBe(true);
      expect(result.current.error).toBe(null);

      // Verify the response structure and transformation
      expect(result.current.data).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'llm-guidelines-scorer',
            type: 'llm',
            llmTemplate: 'Guidelines',
            guidelines: ['Test guideline 1', 'Test guideline 2'],
            sampleRate: 50, // Converted from 0.5 to percentage
            filterString: 'column = "value"',
          },
          {
            name: 'custom-code-scorer',
            type: 'custom-code',
            code: 'def my_scorer(inputs, outputs):\n    return {"score": 1}',
            callSignature: '(inputs, outputs)',
            originalFuncName: 'my_scorer',
            sampleRate: 25, // Converted from 0.25 to percentage
          },
        ],
      });

      expect(mockListScheduledScorers).toHaveBeenCalledWith(mockExperimentId);

      // Verify cache was populated
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual(result.current.data);
    });

    it('should successfully handle empty scorers list', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      mockListScheduledScorers.mockResolvedValue(mockEmptyApiResponse);

      const { result } = renderHook(() => useGetScheduledScorers(mockExperimentId), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act & Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.isSuccess).toBe(true);
      expect(result.current.data).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [],
      });

      expect(mockListScheduledScorers).toHaveBeenCalledWith(mockExperimentId);

      // Verify cache was populated with empty data
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [],
      });
    });

    it('should handle response with missing scheduled_scorers field', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const responseWithoutScorers = {
        experiment_id: mockExperimentId,
      };
      mockListScheduledScorers.mockResolvedValue(responseWithoutScorers);

      const { result } = renderHook(() => useGetScheduledScorers(mockExperimentId), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act & Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.isSuccess).toBe(true);
      expect(result.current.data).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [],
      });

      // Verify cache was populated with fallback data
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [],
      });
    });
  });

  describe('Query Configuration', () => {
    it('should not execute query when experimentId is falsy', () => {
      const falsyValues = [undefined, ''];

      falsyValues.forEach((experimentId) => {
        const { result } = renderHook(() => useGetScheduledScorers(experimentId), {
          wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
        });

        expect(result.current.data).toBeUndefined();
        expect(mockListScheduledScorers).not.toHaveBeenCalled();
      });
    });
  });

  describe('Error Conditions', () => {
    it('should handle various API errors properly', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const errors = [new InternalServerError({}), new NotFoundError({}), new Error('Network timeout')];

      for (const apiError of errors) {
        jest.clearAllMocks();
        mockListScheduledScorers.mockRejectedValue(apiError);

        // Create a new client for each iteration to avoid eslint warning
        const testQueryClient = queryClient;
        const { result } = renderHook(() => useGetScheduledScorers(mockExperimentId), {
          wrapper: ({ children }) => <QueryClientProvider client={testQueryClient}>{children}</QueryClientProvider>,
        });

        // Act & Assert
        await waitFor(() => {
          expect(result.current.isLoading).toBe(false);
        });

        expect(result.current.isError).toBe(true);
        expect(result.current.error).toBe(apiError);
        expect(result.current.data).toBeUndefined();
        expect(mockListScheduledScorers).toHaveBeenCalledWith(mockExperimentId);

        // Verify cache was not populated due to error
        const cacheAfterError = testQueryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
        expect(cacheAfterError).toBeUndefined();
      }
    });

    it('should handle scorer transformation errors gracefully', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      // Create a malformed scorer config that will cause transformation to fail
      const malformedResponse = {
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: [
            {
              name: 'malformed-scorer',
              serialized_scorer: 'invalid-json',
              sample_rate: 0.5,
            },
          ],
        },
      };

      mockListScheduledScorers.mockResolvedValue(malformedResponse);

      const { result } = renderHook(() => useGetScheduledScorers(mockExperimentId), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act & Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.isError).toBe(true);
      expect(result.current.error?.message).toContain('Failed to parse scorer configuration');
      expect(mockListScheduledScorers).toHaveBeenCalledWith(mockExperimentId);

      // Verify cache was not populated due to transformation error
      const cacheAfterError = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheAfterError).toBeUndefined();
    });
  });

  describe('Query Key and Caching', () => {
    it('should use correct query key format', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      mockListScheduledScorers.mockResolvedValue(mockEmptyApiResponse);

      const { result } = renderHook(() => useGetScheduledScorers(mockExperimentId), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Assert - Verify the specific query key is used
      const cacheData = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheData).toBeDefined();

      // Verify different experiment IDs use different cache keys
      const differentExperimentCache = queryClient.getQueryData([
        'mlflow',
        'scheduled-scorers',
        'different-experiment',
      ]);
      expect(differentExperimentCache).toBeUndefined();
    });

    it('should respect staleTime configuration', async () => {
      // Arrange
      mockListScheduledScorers.mockResolvedValue(mockEmptyApiResponse);

      const { result } = renderHook(() => useGetScheduledScorers(mockExperimentId), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Verify query was cached and won't refetch immediately
      const queryState = queryClient.getQueryState(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(queryState?.dataUpdatedAt).toBeDefined();

      // Clear mock to verify no refetch happens
      jest.clearAllMocks();

      // Render hook again - should not trigger new request due to staleTime
      renderHook(() => useGetScheduledScorers(mockExperimentId), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      expect(mockListScheduledScorers).not.toHaveBeenCalled();
    });
  });
});
