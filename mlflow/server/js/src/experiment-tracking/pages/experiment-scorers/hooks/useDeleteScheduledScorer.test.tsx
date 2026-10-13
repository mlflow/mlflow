import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { renderHook, waitFor } from '@testing-library/react';
import { InternalServerError } from '@databricks/web-shared/errors';

import { useDeleteScheduledScorerMutation } from './useDeleteScheduledScorer';
import { deleteScheduledScorers, listScheduledScorers, updateScheduledScorers } from '../api';
import { shouldPaginateScorers } from '../../../../common/utils/FeatureUtils';

// Mock external dependencies
jest.mock('../api');

const mockListAllScheduledScorers = jest.mocked(listScheduledScorers);
const mockDeleteScheduledScorers = jest.mocked(deleteScheduledScorers);
const mockUpdateScheduledScorers = jest.mocked(updateScheduledScorers);
const paginationEnabled = shouldPaginateScorers();

describe('useDeleteScheduledScorerMutation', () => {
  let queryClient: QueryClient;
  const mockExperimentId = 'experiment-123';

  const mockScorerConfig1 = {
    name: 'test-scorer-1',
    scorer_name: 'test-scorer-1',
    serialized_scorer: '{"name": "test-scorer-1"}',
    sample_rate: 0.5,
    filter_string: 'column = "value1"',
    scorer_version: 1,
  };

  const mockScorerConfig2 = {
    name: 'test-scorer-2',
    scorer_name: 'test-scorer-2',
    serialized_scorer: '{"name": "test-scorer-2"}',
    sample_rate: 0.25,
    filter_string: 'column = "value2"',
    scorer_version: 1,
  };

  const mockScorerConfig3 = {
    name: 'test-scorer-3',
    scorer_name: 'test-scorer-3',
    serialized_scorer: '{"name": "test-scorer-3"}',
    sample_rate: 0.75,
    filter_string: 'column = "value3"',
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
      // Arrange
      queryClient.setQueryData(['mlflow', 'scheduled-scorers', mockExperimentId], {
        experimentId: mockExperimentId,
        scheduledScorers: [{ name: 'test-scorer-1' }, { name: 'test-scorer-2' }],
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
      if (paginationEnabled) {
        expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
        expect(mockListAllScheduledScorers).not.toHaveBeenCalled();
      } else {
        expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId, undefined);
      }

      // Verify cache was invalidated
      const cacheState = queryClient.getQueryState(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheState?.isInvalidated).toBe(true);
    });

    it('should successfully delete all scorers when empty scorerNames array provided', async () => {
      // Arrange
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
      if (paginationEnabled) {
        expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
        expect(mockListAllScheduledScorers).not.toHaveBeenCalled();
      } else {
        expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId, []);
      }
    });

    it('should delete specific scorers', async () => {
      if (paginationEnabled) {
        // Pagination path: fetch-filter-PATCH
        mockListAllScheduledScorers.mockResolvedValue([mockScorerConfig1, mockScorerConfig2] as any);

        const expectedUpdateResponse = {
          experiment_id: mockExperimentId,
          scheduled_scorers: {
            scorers: [mockScorerConfig2],
          },
        };
        mockUpdateScheduledScorers.mockResolvedValue(expectedUpdateResponse);

        const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
          wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
        });

        const response = await result.current.mutateAsync({
          experimentId: mockExperimentId,
          scorerNames: ['test-scorer-1'],
        });

        expect(mockListAllScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
        expect(mockUpdateScheduledScorers).toHaveBeenCalledWith(mockExperimentId, {
          scorers: [mockScorerConfig2],
        });
        expect(mockDeleteScheduledScorers).not.toHaveBeenCalled();
        expect(response).toEqual(expectedUpdateResponse);
      } else {
        // OSS path: pass scorer names directly to delete API
        mockDeleteScheduledScorers.mockResolvedValue(mockDeleteAllResponse);

        const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
          wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
        });

        const response = await result.current.mutateAsync({
          experimentId: mockExperimentId,
          scorerNames: ['test-scorer-1'],
        });

        expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId, ['test-scorer-1']);
        expect(response).toEqual(mockDeleteAllResponse);
      }
    });

    it('should delete multiple specific scorers', async () => {
      if (paginationEnabled) {
        // Pagination path: fetch-filter-PATCH
        mockListAllScheduledScorers.mockResolvedValue([mockScorerConfig1, mockScorerConfig2, mockScorerConfig3] as any);

        const expectedUpdateResponse = {
          experiment_id: mockExperimentId,
          scheduled_scorers: {
            scorers: [mockScorerConfig2],
          },
        };
        mockUpdateScheduledScorers.mockResolvedValue(expectedUpdateResponse);

        const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
          wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
        });

        const response = await result.current.mutateAsync({
          experimentId: mockExperimentId,
          scorerNames: ['test-scorer-1', 'test-scorer-3'],
        });

        expect(mockUpdateScheduledScorers).toHaveBeenCalledWith(mockExperimentId, {
          scorers: [mockScorerConfig2],
        });
        expect(response).toEqual(expectedUpdateResponse);
      } else {
        // OSS path: pass scorer names directly
        mockDeleteScheduledScorers.mockResolvedValue(mockDeleteAllResponse);

        const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
          wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
        });

        const response = await result.current.mutateAsync({
          experimentId: mockExperimentId,
          scorerNames: ['test-scorer-1', 'test-scorer-3'],
        });

        expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId, ['test-scorer-1', 'test-scorer-3']);
        expect(response).toEqual(mockDeleteAllResponse);
      }
    });

    it('should fall back to deleteAll when all scorers are being removed', async () => {
      mockListAllScheduledScorers.mockResolvedValue([mockScorerConfig1, mockScorerConfig2] as any);
      mockDeleteScheduledScorers.mockResolvedValue(mockDeleteAllResponse);

      const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      const response = await result.current.mutateAsync({
        experimentId: mockExperimentId,
        scorerNames: ['test-scorer-1', 'test-scorer-2'],
      });

      if (paginationEnabled) {
        expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
      } else {
        expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId, ['test-scorer-1', 'test-scorer-2']);
      }
      expect(mockUpdateScheduledScorers).not.toHaveBeenCalled();
      expect(response).toEqual(mockDeleteAllResponse);
    });

    it('should handle deletion when scorer names do not match existing scorers', async () => {
      // Arrange
      mockListAllScheduledScorers.mockResolvedValue([mockScorerConfig1, mockScorerConfig2] as any);

      const expectedUpdateResponse = {
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: [mockScorerConfig1, mockScorerConfig2],
        },
      };
      mockUpdateScheduledScorers.mockResolvedValue(expectedUpdateResponse);

      const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const response = await result.current.mutateAsync({
        experimentId: mockExperimentId,
        scorerNames: ['non-existent-scorer'],
      });

      if (paginationEnabled) {
        // Assert - no scorers removed, PATCHes back the same list
        expect(mockUpdateScheduledScorers).toHaveBeenCalledWith(mockExperimentId, {
          scorers: [mockScorerConfig1, mockScorerConfig2],
        });
        expect(response).toEqual(expectedUpdateResponse);
      } else {
        expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId, ['non-existent-scorer']);
        expect(response).toEqual(mockDeleteAllResponse);
      }
    });
  });

  describe('Edge Cases and Error Conditions', () => {
    // Pagination-only: listAll failure during selective delete
    if (paginationEnabled) {
      it('should handle listAllScheduledScorers API failure when deleting specific scorers', async () => {
        const networkError = new InternalServerError({});
        mockListAllScheduledScorers.mockRejectedValue(networkError);

        const { result } = renderHook(() => useDeleteScheduledScorerMutation(), {
          wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
        });

        await expect(
          result.current.mutateAsync({
            experimentId: mockExperimentId,
            scorerNames: ['test-scorer-1'],
          }),
        ).rejects.toThrow(InternalServerError);

        expect(mockListAllScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
        expect(mockUpdateScheduledScorers).not.toHaveBeenCalled();
      });
    }

    it('should handle deleteScheduledScorers API failure when deleting all scorers', async () => {
      // Arrange
      queryClient.setQueryData(['mlflow', 'scheduled-scorers', mockExperimentId], {
        experimentId: mockExperimentId,
        scheduledScorers: [{ name: 'test-scorer-1' }],
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

      if (paginationEnabled) {
        expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
      } else {
        expect(mockDeleteScheduledScorers).toHaveBeenCalledWith(mockExperimentId, undefined);
      }

      // Verify cache was not modified due to error
      const cacheAfterError = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheAfterError).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [{ name: 'test-scorer-1' }],
      });
    });

    // Pagination-only: updateScheduledScorers failure during selective delete
    if (paginationEnabled) {
      it('should handle updateScheduledScorers API failure during selective delete', async () => {
        // Arrange
        mockListAllScheduledScorers.mockResolvedValue([mockScorerConfig1, mockScorerConfig2] as any);
        const updateError = new InternalServerError({});
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

        expect(mockUpdateScheduledScorers).toHaveBeenCalledWith(mockExperimentId, {
          scorers: [mockScorerConfig2],
        });
      });
    }
  });
});
