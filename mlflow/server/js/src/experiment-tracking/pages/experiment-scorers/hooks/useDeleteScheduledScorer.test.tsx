import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { renderHook, waitFor } from '@testing-library/react';
import { InternalServerError } from '@databricks/web-shared/errors';
import { useDeleteScheduledScorerMutation } from './useDeleteScheduledScorer';
import { deleteScheduledScorers } from '../api';

// Mock external dependencies
jest.mock('../api');

const mockDeleteScheduledScorers = jest.mocked(deleteScheduledScorers);

describe('useDeleteScheduledScorerMutation', () => {
  let queryClient: QueryClient;

  const mockExperimentId = 'experiment-123';

  const mockScorerConfig1 = {
    scorer_name: 'test-scorer-1',
    serialized_scorer: '{"name": "test-scorer-1"}',
    sample_rate: 0.5,
    filter_string: 'column = "value1"',
    scorer_version: 1,
  };

  const mockScorerConfig2 = {
    scorer_name: 'test-scorer-2',
    serialized_scorer: '{"name": "test-scorer-2"}',
    sample_rate: 0.25,
    filter_string: 'column = "value2"',
    scorer_version: 1,
  };

  const mockDeleteAllResponse = {
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

    // Reset all mocks
    jest.clearAllMocks();
  });

  describe('Golden Path - Successful Operations', () => {
    it('should successfully delete all scorers when no scorerNames provided', async () => {
      // Arrange - Set up initial cache with scorers
      queryClient.setQueryData(['mlflow', 'scheduled-scorers', mockExperimentId], {
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'test-scorer-1',
            type: 'custom-code',
            sampleRate: 50,
            filterString: 'column = "value1"',
            code: '',
            version: 1,
          },
          {
            name: 'test-scorer-2',
            type: 'custom-code',
            sampleRate: 25,
            filterString: 'column = "value2"',
            code: '',
            version: 1,
          },
        ],
      });

      mockDeleteScheduledScorers.mockResolvedValue(mockDeleteAllResponse);

      const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
      });

      // Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      const response = await mutationPromise;
      expect(response).toEqual(mockDeleteAllResponse);
      expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId, undefined);

      // Verify cache was invalidated (not set to empty array)
      // The cache should be marked as stale for refetching
      const cacheState = queryClient.getQueryState(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheState?.isInvalidated).toBe(true);
    });

    it('should successfully delete all scorers when empty scorerNames array provided', async () => {
      // Arrange - Set up initial cache with scorers
      queryClient.setQueryData(['mlflow', 'scheduled-scorers', mockExperimentId], {
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'test-scorer-1',
            type: 'custom-code',
            sampleRate: 50,
            filterString: 'column = "value1"',
            code: '',
            version: 1,
          },
        ],
      });

      mockDeleteScheduledScorers.mockResolvedValue(mockDeleteAllResponse);

      const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
        scorerNames: [],
      });

      // Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      const response = await mutationPromise;
      expect(response).toEqual(mockDeleteAllResponse);
      expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId, []);

      // Verify cache was invalidated
      const cacheState = queryClient.getQueryState(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheState?.isInvalidated).toBe(true);
    });

    it('should successfully delete specific scorers and update cache', async () => {
      // Arrange - Set up initial cache with two scorers
      queryClient.setQueryData(['mlflow', 'scheduled-scorers', mockExperimentId], {
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'test-scorer-1',
            type: 'custom-code',
            sampleRate: 50,
            filterString: 'column = "value1"',
            code: '',
            version: 1,
          },
          {
            name: 'test-scorer-2',
            type: 'custom-code',
            sampleRate: 25,
            filterString: 'column = "value2"',
            code: '',
            version: 1,
          },
        ],
      });

      const expectedDeleteResponse = {
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: [mockScorerConfig2],
        },
      };

      mockDeleteScheduledScorers.mockResolvedValue(expectedDeleteResponse);

      const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
        scorerNames: ['test-scorer-1'],
      });

      // Assert
      const response = await mutationPromise;
      expect(response).toEqual(expectedDeleteResponse);
      expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId, ['test-scorer-1']);

      // Verify cache was updated with only the remaining scorer
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'test-scorer-2',
            type: 'custom-code',
            sampleRate: 25,
            filterString: 'column = "value2"',
            code: '',
            version: 1,
          },
        ],
      });
    });

    it('should delete multiple specific scorers and update cache', async () => {
      // Arrange - Set up initial cache with three scorers
      queryClient.setQueryData(['mlflow', 'scheduled-scorers', mockExperimentId], {
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'test-scorer-1',
            type: 'custom-code',
            sampleRate: 50,
            filterString: 'column = "value1"',
            code: '',
            version: 1,
          },
          {
            name: 'test-scorer-2',
            type: 'custom-code',
            sampleRate: 25,
            filterString: 'column = "value2"',
            code: '',
            version: 1,
          },
          {
            name: 'test-scorer-3',
            type: 'custom-code',
            sampleRate: 75,
            filterString: 'column = "value3"',
            code: '',
            version: 1,
          },
        ],
      });

      const expectedDeleteResponse = {
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: [mockScorerConfig2],
        },
      };

      mockDeleteScheduledScorers.mockResolvedValue(expectedDeleteResponse);

      const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act - Delete scorer-1 and scorer-3, leaving scorer-2
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
        scorerNames: ['test-scorer-1', 'test-scorer-3'],
      });

      // Assert
      const response = await mutationPromise;
      expect(response).toEqual(expectedDeleteResponse);
      expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId, ['test-scorer-1', 'test-scorer-3']);

      // Verify cache was updated with only scorer-2
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'test-scorer-2',
            type: 'custom-code',
            sampleRate: 25,
            filterString: 'column = "value2"',
            code: '',
            version: 1,
          },
        ],
      });
    });

    it('should handle deletion when cache does not exist', async () => {
      // Arrange - No initial cache set
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      mockDeleteScheduledScorers.mockResolvedValue(mockDeleteAllResponse);

      const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
        scorerNames: ['test-scorer-1'],
      });

      // Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      const response = await mutationPromise;
      expect(response).toEqual(mockDeleteAllResponse);
      expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId, ['test-scorer-1']);

      // When there's no existing cache, removeScheduledScorersFromCache invalidates the query
      // but invalidation on a non-existent cache doesn't create a cache entry
      // Just verify that the operation completed successfully without error
      const cacheAfter = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheAfter).toBeUndefined();
    });

    it('should handle deletion when scorer names do not match existing scorers', async () => {
      // Arrange - Set up initial cache with two scorers
      queryClient.setQueryData(['mlflow', 'scheduled-scorers', mockExperimentId], {
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'test-scorer-1',
            type: 'custom-code',
            sampleRate: 50,
            filterString: 'column = "value1"',
            code: '',
            version: 1,
          },
          {
            name: 'test-scorer-2',
            type: 'custom-code',
            sampleRate: 25,
            filterString: 'column = "value2"',
            code: '',
            version: 1,
          },
        ],
      });

      const mockExistingScorersResponse = {
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: [mockScorerConfig1, mockScorerConfig2],
        },
      };

      // Backend returns all scorers unchanged when deleting non-existent scorer
      mockDeleteScheduledScorers.mockResolvedValue(mockExistingScorersResponse);

      const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
        scorerNames: ['non-existent-scorer'],
      });

      // Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      const response = await mutationPromise;
      expect(response).toEqual(mockExistingScorersResponse);
      expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId, ['non-existent-scorer']);

      // Verify cache still has both scorers (nothing was deleted)
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'test-scorer-1',
            type: 'custom-code',
            sampleRate: 50,
            filterString: 'column = "value1"',
            code: '',
            version: 1,
          },
          {
            name: 'test-scorer-2',
            type: 'custom-code',
            sampleRate: 25,
            filterString: 'column = "value2"',
            code: '',
            version: 1,
          },
        ],
      });
    });
  });

  describe('Edge Cases and Error Conditions', () => {
    it('should handle deleteScheduledScorers API failure when deleting specific scorers', async () => {
      // Arrange - Set up initial cache
      queryClient.setQueryData(['mlflow', 'scheduled-scorers', mockExperimentId], {
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'test-scorer-1',
            type: 'custom-code',
            sampleRate: 50,
            filterString: 'column = "value1"',
            code: '',
            version: 1,
          },
        ],
      });

      const networkError = new InternalServerError({});
      mockDeleteScheduledScorers.mockRejectedValue(networkError);

      const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act & Assert
      await expect(
        result.current.mutateAsync({
          experimentId: mockExperimentId,
          scorerNames: ['test-scorer-1'],
        }),
      ).rejects.toThrow(InternalServerError);

      expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId, ['test-scorer-1']);

      // Verify cache was not modified due to error
      const cacheAfterError = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheAfterError).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'test-scorer-1',
            type: 'custom-code',
            sampleRate: 50,
            filterString: 'column = "value1"',
            code: '',
            version: 1,
          },
        ],
      });
    });

    it('should handle deleteScheduledScorers API failure when deleting all scorers', async () => {
      // Arrange - Set up initial cache
      queryClient.setQueryData(['mlflow', 'scheduled-scorers', mockExperimentId], {
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'test-scorer-1',
            type: 'custom-code',
            sampleRate: 50,
            filterString: 'column = "value1"',
            code: '',
            version: 1,
          },
        ],
      });

      const deleteError = new InternalServerError({});
      mockDeleteScheduledScorers.mockRejectedValue(deleteError);

      const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act & Assert
      await expect(
        result.current.mutateAsync({
          experimentId: mockExperimentId,
        }),
      ).rejects.toThrow(InternalServerError);

      expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId, undefined);

      // Verify cache was not modified due to error
      const cacheAfterError = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheAfterError).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'test-scorer-1',
            type: 'custom-code',
            sampleRate: 50,
            filterString: 'column = "value1"',
            code: '',
            version: 1,
          },
        ],
      });
    });

    it('should return response correctly', async () => {
      // Arrange
      const customResponse = {
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: [mockScorerConfig1],
        },
      };

      mockDeleteScheduledScorers.mockResolvedValue(customResponse);

      const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const response = await result.current.mutateAsync({
        experimentId: mockExperimentId,
        scorerNames: ['test-scorer-2'],
      });

      // Assert - verify the response is returned as-is
      expect(response).toEqual(customResponse);
      expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId, ['test-scorer-2']);
    });
  });
});
