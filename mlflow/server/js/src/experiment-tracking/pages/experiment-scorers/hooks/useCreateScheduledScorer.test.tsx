import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { renderHook, waitFor } from '@testing-library/react';
import { NotFoundError, InternalServerError } from '@databricks/web-shared/errors';
import { useCreateScheduledScorerMutation } from './useCreateScheduledScorer';
import { listScheduledScorers, updateScheduledScorers, createScheduledScorers } from '../api';
import { transformScheduledScorer } from '../utils/scorerTransformUtils';

// Mock external dependencies
jest.mock('../api');

const mockListScheduledScorers = jest.mocked(listScheduledScorers);
const mockUpdateScheduledScorers = jest.mocked(updateScheduledScorers);
const mockCreateScheduledScorers = jest.mocked(createScheduledScorers);

describe('useCreateScheduledScorerMutation', () => {
  let queryClient: QueryClient;
  const mockExperimentId = 'experiment-123';

  const mockLLMScorer = {
    name: 'test-llm-scorer',
    type: 'llm' as const,
    llmTemplate: 'Guidelines' as const,
    guidelines: ['Test guideline 1', 'Test guideline 2'],
    sampleRate: 50,
    filterString: 'column = "value"',
  };

  const mockCustomCodeScorer = {
    name: 'test-custom-scorer',
    type: 'custom-code' as const,
    code: 'def my_scorer(inputs, outputs):\n    return {"score": 1}',
    sampleRate: 25,
    callSignature: '',
    originalFuncName: '',
  };

  const mockScorerConfig = {
    name: 'test-scorer-config',
    serialized_scorer: '{"name": "test"}',
    sample_rate: 0.5,
    filter_string: 'column = "value"',
  };

  const mockExistingScorersResponse = {
    experiment_id: mockExperimentId,
    scheduled_scorers: {
      scorers: [mockScorerConfig],
    },
  };

  const mockEmptyScorersResponse = {
    experiment_id: mockExperimentId,
    scheduled_scorers: {
      scorers: [],
    },
  };

  const mockCreateResponse = {
    experiment_id: mockExperimentId,
    scheduled_scorers: {
      scorers: [mockScorerConfig],
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
    it('should successfully create a new scorer when no existing scorers', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      mockListScheduledScorers.mockResolvedValue(mockEmptyScorersResponse);
      mockUpdateScheduledScorers.mockRejectedValue(new NotFoundError({}));
      mockCreateScheduledScorers.mockResolvedValue(mockCreateResponse);

      const { result } = renderHook(() => useCreateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
        scheduledScorer: mockLLMScorer,
      });

      // Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      const response = await mutationPromise;
      expect(response).toEqual(mockCreateResponse);

      // Verify the response structure
      expect(response).toMatchObject({
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: expect.arrayContaining([mockScorerConfig]),
        },
      });

      expect(mockListScheduledScorers).toHaveBeenCalledWith(mockExperimentId);

      // Verify the real transformation was used
      const expectedTransformedConfig = transformScheduledScorer(mockLLMScorer);
      expect(mockUpdateScheduledScorers).toHaveBeenCalledWith(mockExperimentId, {
        scorers: [expectedTransformedConfig],
      });
      expect(mockCreateScheduledScorers).toHaveBeenCalledWith(mockExperimentId, {
        scorers: [expectedTransformedConfig],
      });
      // Verify cache was updated with the transformed data
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: expect.arrayContaining([
          expect.objectContaining({
            name: 'test-scorer-config',
            type: 'custom-code',
            sampleRate: 50,
            filterString: 'column = "value"',
          }),
        ]),
      });
    });

    it('should successfully add scorer to existing list', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      mockListScheduledScorers.mockResolvedValue(mockExistingScorersResponse);
      mockUpdateScheduledScorers.mockResolvedValue(mockCreateResponse);

      const { result } = renderHook(() => useCreateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
        scheduledScorer: mockCustomCodeScorer,
      });

      // Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      const response = await mutationPromise;
      expect(response).toEqual(mockCreateResponse);
      expect(mockListScheduledScorers).toHaveBeenCalledWith(mockExperimentId);

      // Verify the real transformation was used
      const expectedTransformedConfig = transformScheduledScorer(mockCustomCodeScorer);
      expect(mockUpdateScheduledScorers).toHaveBeenCalledWith(mockExperimentId, {
        scorers: [mockScorerConfig, expectedTransformedConfig],
      });
      expect(mockCreateScheduledScorers).not.toHaveBeenCalled();
      // Verify cache was updated with the transformed data
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: expect.arrayContaining([
          expect.objectContaining({
            name: 'test-scorer-config',
            type: 'custom-code',
            sampleRate: 50,
            filterString: 'column = "value"',
          }),
        ]),
      });
    });

    it('should successfully create scorer when update succeeds on first try (empty list case)', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      mockListScheduledScorers.mockResolvedValue(mockEmptyScorersResponse);
      mockUpdateScheduledScorers.mockResolvedValue(mockCreateResponse);

      const { result } = renderHook(() => useCreateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
        scheduledScorer: mockLLMScorer,
      });

      // Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      const response = await mutationPromise;
      expect(response).toEqual(mockCreateResponse);

      // Verify the real transformation was used
      const expectedTransformedConfig = transformScheduledScorer(mockLLMScorer);
      expect(mockUpdateScheduledScorers).toHaveBeenCalledWith(mockExperimentId, {
        scorers: [expectedTransformedConfig],
      });
      expect(mockCreateScheduledScorers).not.toHaveBeenCalled();
      // Verify cache was updated with the transformed data
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: expect.arrayContaining([
          expect.objectContaining({
            name: 'test-scorer-config',
            type: 'custom-code',
            sampleRate: 50,
            filterString: 'column = "value"',
          }),
        ]),
      });
    });
  });

  describe('Edge Cases and Error Conditions', () => {
    it('should throw error when scorer name already exists', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const duplicateScorer = { ...mockLLMScorer, name: 'test-scorer-config' }; // Same name as existing
      mockListScheduledScorers.mockResolvedValue(mockExistingScorersResponse);

      const { result } = renderHook(() => useCreateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act & Assert
      await expect(
        result.current.mutateAsync({
          experimentId: mockExperimentId,
          scheduledScorer: duplicateScorer,
        }),
      ).rejects.toThrow(
        "A scorer with name 'test-scorer-config' has already been registered. " +
          'Update the existing scorer or choose a different name.',
      );

      expect(mockListScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
      expect(mockUpdateScheduledScorers).not.toHaveBeenCalled();
      expect(mockCreateScheduledScorers).not.toHaveBeenCalled();
      // Verify cache was not modified due to validation error
      const cacheAfterError = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheAfterError).toBeUndefined();
    });

    it('should re-throw non-404 update errors', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const serverError = new InternalServerError({});
      mockListScheduledScorers.mockResolvedValue(mockEmptyScorersResponse);
      mockUpdateScheduledScorers.mockRejectedValue(serverError);

      const { result } = renderHook(() => useCreateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act & Assert
      await expect(
        result.current.mutateAsync({
          experimentId: mockExperimentId,
          scheduledScorer: mockLLMScorer,
        }),
      ).rejects.toThrow(InternalServerError);

      expect(mockCreateScheduledScorers).not.toHaveBeenCalled();
      // Verify cache was not modified due to error
      const cacheAfterError = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheAfterError).toBeUndefined();
    });

    it('should handle listScheduledScorers API failure', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const networkError = new InternalServerError({});
      mockListScheduledScorers.mockRejectedValue(networkError);

      const { result } = renderHook(() => useCreateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act & Assert
      await expect(
        result.current.mutateAsync({
          experimentId: mockExperimentId,
          scheduledScorer: mockLLMScorer,
        }),
      ).rejects.toThrow(InternalServerError);

      expect(mockUpdateScheduledScorers).not.toHaveBeenCalled();
      expect(mockCreateScheduledScorers).not.toHaveBeenCalled();
      // Verify cache was not modified due to error
      const cacheAfterError = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheAfterError).toBeUndefined();
    });

    it('should handle createScheduledScorers API failure after update 404', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const createError = new InternalServerError({});
      mockListScheduledScorers.mockResolvedValue(mockEmptyScorersResponse);
      mockUpdateScheduledScorers.mockRejectedValue(new NotFoundError({}));
      mockCreateScheduledScorers.mockRejectedValue(createError);

      const { result } = renderHook(() => useCreateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act & Assert
      await expect(
        result.current.mutateAsync({
          experimentId: mockExperimentId,
          scheduledScorer: mockLLMScorer,
        }),
      ).rejects.toThrow(InternalServerError);

      // Verify cache was not modified due to error
      const cacheAfterError = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheAfterError).toBeUndefined();
    });
  });
});
