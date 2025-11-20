import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { renderHook, waitFor } from '@testing-library/react';
import { NotFoundError, InternalServerError } from '@databricks/web-shared/errors';
import { useDeleteScheduledScorerMutation } from './useDeleteScheduledScorer';
import { listScheduledScorers, updateScheduledScorers, deleteScheduledScorers } from '../api';

// Mock external dependencies
jest.mock('../api');

const mockListScheduledScorers = jest.mocked(listScheduledScorers);
const mockUpdateScheduledScorers = jest.mocked(updateScheduledScorers);
const mockDeleteScheduledScorers = jest.mocked(deleteScheduledScorers);

describe('useDeleteScheduledScorerMutation', () => {
  let queryClient: QueryClient;

  const mockExperimentId = 'experiment-123';

  const mockScorerConfig1 = {
    name: 'test-scorer-1',
    serialized_scorer: '{"name": "test-scorer-1"}',
    sample_rate: 0.5,
    filter_string: 'column = "value1"',
  };

  const mockScorerConfig2 = {
    name: 'test-scorer-2',
    serialized_scorer: '{"name": "test-scorer-2"}',
    sample_rate: 0.25,
    filter_string: 'column = "value2"',
  };

  const mockExistingScorersResponse = {
    experiment_id: mockExperimentId,
    scheduled_scorers: {
      scorers: [mockScorerConfig1, mockScorerConfig2],
    },
  };

  const mockSingleScorerResponse = {
    experiment_id: mockExperimentId,
    scheduled_scorers: {
      scorers: [mockScorerConfig1],
    },
  };

  const mockEmptyResponse = {
    experiment_id: mockExperimentId,
    scheduled_scorers: {
      scorers: [],
    },
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
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

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
      expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
      expect(mockListScheduledScorers).not.toHaveBeenCalled();
      expect(mockUpdateScheduledScorers).not.toHaveBeenCalled();

      // Verify cache was updated
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [],
      });
    });

    it('should successfully delete all scorers when empty scorerNames array provided', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

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
      expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
      expect(mockListScheduledScorers).not.toHaveBeenCalled();
      expect(mockUpdateScheduledScorers).not.toHaveBeenCalled();
    });

    it('should successfully delete specific scorers and update with remaining ones', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const expectedUpdateResponse = {
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: [mockScorerConfig2],
        },
      };

      mockListScheduledScorers.mockResolvedValue(mockExistingScorersResponse);
      mockUpdateScheduledScorers.mockResolvedValue(expectedUpdateResponse);

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
      expect(response).toEqual(expectedUpdateResponse);
      expect(mockListScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
      expect(mockUpdateScheduledScorers).toHaveBeenCalledWith(mockExperimentId, {
        scorers: [mockScorerConfig2],
      });
      expect(mockDeleteScheduledScorers).not.toHaveBeenCalled();

      // Verify cache was updated with remaining scorers
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
          },
        ],
      });
    });

    it('should delete all scorers using DELETE endpoint when all specific scorers are selected', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      mockListScheduledScorers.mockResolvedValue(mockExistingScorersResponse);
      mockDeleteScheduledScorers.mockResolvedValue(mockDeleteAllResponse);

      const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
        scorerNames: ['test-scorer-1', 'test-scorer-2'],
      });

      // Assert
      const response = await mutationPromise;
      expect(response).toEqual(mockDeleteAllResponse);
      expect(mockListScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
      expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
      expect(mockUpdateScheduledScorers).not.toHaveBeenCalled();

      // Verify cache was updated with empty scorers array
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [],
      });
    });

    it('should delete all scorers using DELETE endpoint when remaining scorers array is empty after filtering', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      mockListScheduledScorers.mockResolvedValue(mockSingleScorerResponse);
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
      const response = await mutationPromise;
      expect(response).toEqual(mockDeleteAllResponse);
      expect(mockListScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
      expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
      expect(mockUpdateScheduledScorers).not.toHaveBeenCalled();

      // Verify cache was updated with empty scorers array
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [],
      });
    });

    it('should handle deletion when scorer names do not match existing scorers', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      mockListScheduledScorers.mockResolvedValue(mockExistingScorersResponse);
      mockUpdateScheduledScorers.mockResolvedValue(mockExistingScorersResponse);

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
      expect(mockListScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
      // All scorers remain since none matched for deletion
      expect(mockUpdateScheduledScorers).toHaveBeenCalledWith(mockExperimentId, {
        scorers: [mockScorerConfig1, mockScorerConfig2],
      });
      expect(mockDeleteScheduledScorers).not.toHaveBeenCalled();
    });
  });

  describe('Edge Cases and Error Conditions', () => {
    it('should handle listScheduledScorers API failure', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const networkError = new InternalServerError({});
      mockListScheduledScorers.mockRejectedValue(networkError);

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

      expect(mockListScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
      expect(mockUpdateScheduledScorers).not.toHaveBeenCalled();
      expect(mockDeleteScheduledScorers).not.toHaveBeenCalled();

      // Verify cache was not modified due to error
      const cacheAfterError = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheAfterError).toBeUndefined();
    });

    it('should handle deleteScheduledScorers API failure', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

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

      expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId);

      // Verify cache was not modified due to error
      const cacheAfterError = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheAfterError).toBeUndefined();
    });

    it('should handle updateScheduledScorers API failure', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const updateError = new InternalServerError({});
      mockListScheduledScorers.mockResolvedValue(mockExistingScorersResponse);
      mockUpdateScheduledScorers.mockRejectedValue(updateError);

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

      expect(mockListScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
      expect(mockUpdateScheduledScorers).toHaveBeenCalledWith(mockExperimentId, {
        scorers: [mockScorerConfig2],
      });
      expect(mockDeleteScheduledScorers).not.toHaveBeenCalled();

      // Verify cache was not modified due to error
      const cacheAfterError = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheAfterError).toBeUndefined();
    });

    it('should handle empty existing scorers response', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      mockListScheduledScorers.mockResolvedValue(mockEmptyResponse);
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
      const response = await mutationPromise;
      expect(response).toEqual(mockDeleteAllResponse);
      expect(mockListScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
      expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
      expect(mockUpdateScheduledScorers).not.toHaveBeenCalled();

      // Verify cache was updated with empty scorers array
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [],
      });
    });
  });
});
