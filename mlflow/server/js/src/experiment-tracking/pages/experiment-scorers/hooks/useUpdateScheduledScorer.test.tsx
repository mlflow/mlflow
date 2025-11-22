import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { renderHook, waitFor } from '@testing-library/react';
import { InternalServerError, NotFoundError } from '@databricks/web-shared/errors';
import { useUpdateScheduledScorerMutation } from './useUpdateScheduledScorer';
import { listScheduledScorers, updateScheduledScorers } from '../api';

// Mock external dependencies
jest.mock('../api');

const mockListScheduledScorers = jest.mocked(listScheduledScorers);
const mockUpdateScheduledScorers = jest.mocked(updateScheduledScorers);

describe('useUpdateScheduledScorerMutation', () => {
  let queryClient: QueryClient;

  const mockExperimentId = 'experiment-123';

  const mockUpdatedLLMScorer = {
    name: 'original-scorer', // Same name as existing scorer to test update in place
    type: 'llm' as const,
    llmTemplate: 'Guidelines' as const,
    guidelines: ['Updated guideline 1', 'Updated guideline 2'],
    sampleRate: 75,
    filterString: 'updated_column = "value"',
  };

  const mockUpdatedCustomCodeScorer = {
    name: 'updated-custom-scorer',
    type: 'custom-code' as const,
    code: 'def updated_scorer(inputs, outputs):\n    return {"score": 0.8}',
    sampleRate: 60,
    filterString: 'status = "success"',
    callSignature: '',
    originalFuncName: '',
  };

  const mockOriginalScorerConfig = {
    name: 'original-scorer',
    serialized_scorer: '{"name": "original"}',
    sample_rate: 0.5,
    filter_string: 'original_filter = "value"',
  };

  const mockUpdatedScorerConfig = {
    name: 'original-scorer', // Updated config with same name
    serialized_scorer: JSON.stringify({
      mlflow_version: '3.3.2+ui',
      serialization_version: 1,
      name: 'original-scorer',
      builtin_scorer_class: 'Guidelines',
      builtin_scorer_pydantic_data: {
        name: 'original-scorer',
        required_columns: ['outputs', 'inputs'],
        guidelines: ['Updated guideline 1', 'Updated guideline 2'],
      },
    }),
    builtin: { name: 'original-scorer' },
    sample_rate: 0.75,
    filter_string: 'updated_column = "value"',
  };

  const mockExistingScorersResponse = {
    experiment_id: mockExperimentId,
    scheduled_scorers: {
      scorers: [mockOriginalScorerConfig],
    },
  };

  const mockUpdateResponse = {
    experiment_id: mockExperimentId,
    scheduled_scorers: {
      scorers: [mockUpdatedScorerConfig],
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

  describe('Golden Path - Successful Updates', () => {
    it('should successfully update existing scorer in place', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      mockListScheduledScorers.mockResolvedValue(mockExistingScorersResponse);
      mockUpdateScheduledScorers.mockResolvedValue(mockUpdateResponse);

      const { result } = renderHook(() => useUpdateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
        scheduledScorers: [mockUpdatedLLMScorer],
      });

      // Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      const response = await mutationPromise;
      expect(response).toEqual(mockUpdateResponse);

      // Verify API calls
      expect(mockListScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
      expect(mockUpdateScheduledScorers).toHaveBeenCalledWith(mockExperimentId, {
        scorers: [mockUpdatedScorerConfig],
      });

      // Verify cache was updated with the transformed data
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'original-scorer',
            sampleRate: 75,
            filterString: 'updated_column = "value"',
            type: 'llm',
            llmTemplate: 'Guidelines',
            guidelines: ['Updated guideline 1', 'Updated guideline 2'],
          },
        ],
      });
    });

    it('should successfully update multiple scorers', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const existingScorer1 = {
        name: 'scorer-1',
        serialized_scorer: '{"name": "scorer-1"}',
        sample_rate: 0.3,
      };
      const existingScorer2 = {
        name: 'scorer-2',
        serialized_scorer: '{"name": "scorer-2"}',
        sample_rate: 0.4,
      };

      const multipleScorersResponse = {
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: [existingScorer1, existingScorer2],
        },
      };

      const updatedScorer1 = { ...mockUpdatedLLMScorer, name: 'scorer-1' };
      const updatedScorer2 = { ...mockUpdatedCustomCodeScorer, name: 'scorer-2' };

      const transformedScorer1 = {
        name: 'scorer-1',
        serialized_scorer: JSON.stringify({
          mlflow_version: '3.3.2+ui',
          serialization_version: 1,
          name: 'scorer-1',
          builtin_scorer_class: 'Guidelines',
          builtin_scorer_pydantic_data: {
            name: 'scorer-1',
            required_columns: ['outputs', 'inputs'],
            guidelines: ['Updated guideline 1', 'Updated guideline 2'],
          },
        }),
        builtin: { name: 'scorer-1' },
        sample_rate: 0.75,
        filter_string: 'updated_column = "value"',
      };
      const transformedScorer2 = {
        name: 'scorer-2',
        serialized_scorer: JSON.stringify({
          mlflow_version: '3.3.2+ui',
          serialization_version: 1,
          name: 'scorer-2',
          call_source: 'def updated_scorer(inputs, outputs):\n    return {"score": 0.8}',
          call_signature: '',
          original_func_name: '',
        }),
        custom: {},
        sample_rate: 0.6,
        filter_string: 'status = "success"',
      };

      mockListScheduledScorers.mockResolvedValue(multipleScorersResponse);
      mockUpdateScheduledScorers.mockResolvedValue({
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: [transformedScorer1, transformedScorer2],
        },
      });

      const { result } = renderHook(() => useUpdateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
        scheduledScorers: [updatedScorer1, updatedScorer2],
      });

      // Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      await mutationPromise;
      expect(mockUpdateScheduledScorers).toHaveBeenCalledWith(mockExperimentId, {
        scorers: [transformedScorer1, transformedScorer2],
      });

      // Verify cache was updated with the transformed data
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'scorer-1',
            sampleRate: 75,
            filterString: 'updated_column = "value"',
            type: 'llm',
            llmTemplate: 'Guidelines',
            guidelines: ['Updated guideline 1', 'Updated guideline 2'],
          },
          {
            name: 'scorer-2',
            sampleRate: 60,
            filterString: 'status = "success"',
            type: 'custom-code',
            code: 'def updated_scorer(inputs, outputs):\n    return {"score": 0.8}',
            callSignature: '',
            originalFuncName: '',
          },
        ],
      });
    });

    it('should handle scorer updates and additions correctly', async () => {
      // Arrange - Set up multiple existing scorers
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const scorer1 = { name: 'scorer-1', serialized_scorer: '{}', sample_rate: 0.3 };
      const scorer2 = { name: 'scorer-2', serialized_scorer: '{}', sample_rate: 0.4 };

      const multipleExistingScorersResponse = {
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: [scorer1, scorer2],
        },
      };

      // Create updated scorer-1 and new scorer-3
      const updatedScorer1 = { ...mockUpdatedLLMScorer, name: 'scorer-1' };
      const newScorer3 = { ...mockUpdatedCustomCodeScorer, name: 'scorer-3' };

      const expectedScorerConfigs = [
        {
          name: 'scorer-1',
          serialized_scorer: JSON.stringify({
            mlflow_version: '3.3.2+ui',
            serialization_version: 1,
            name: 'scorer-1',
            builtin_scorer_class: 'Guidelines',
            builtin_scorer_pydantic_data: {
              name: 'scorer-1',
              required_columns: ['outputs', 'inputs'],
              guidelines: ['Updated guideline 1', 'Updated guideline 2'],
            },
          }),
          builtin: { name: 'scorer-1' },
          sample_rate: 0.75,
          filter_string: 'updated_column = "value"',
        },
        scorer2, // Preserved unchanged
        {
          name: 'scorer-3',
          serialized_scorer: JSON.stringify({
            mlflow_version: '3.3.2+ui',
            serialization_version: 1,
            name: 'scorer-3',
            call_source: 'def updated_scorer(inputs, outputs):\n    return {"score": 0.8}',
            call_signature: '',
            original_func_name: '',
          }),
          custom: {},
          sample_rate: 0.6,
          filter_string: 'status = "success"',
        },
      ];

      mockListScheduledScorers.mockResolvedValue(multipleExistingScorersResponse);
      mockUpdateScheduledScorers.mockResolvedValue({
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: expectedScorerConfigs,
        },
      });

      const { result } = renderHook(() => useUpdateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act - Update existing scorer-1 and add new scorer-3
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
        scheduledScorers: [updatedScorer1, newScorer3],
      });

      // Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      await mutationPromise;

      // Verify that scorer-1 was updated, scorer-2 preserved, and scorer-3 added
      expect(mockUpdateScheduledScorers).toHaveBeenCalledWith(mockExperimentId, {
        scorers: expectedScorerConfigs,
      });

      // Verify cache reflects the changes with transformed data
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: expect.arrayContaining([
          expect.objectContaining({
            name: 'scorer-1',
            type: 'llm',
            llmTemplate: 'Guidelines',
            sampleRate: 75,
          }),
          expect.objectContaining({
            name: 'scorer-2',
            sampleRate: 40,
          }),
          expect.objectContaining({
            name: 'scorer-3',
            type: 'custom-code',
            sampleRate: 60,
          }),
        ]),
      });
    });
  });

  describe('Edge Cases and Error Conditions', () => {
    it('should handle listScheduledScorers API failure', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const notFoundError = new NotFoundError({});
      mockListScheduledScorers.mockRejectedValue(notFoundError);

      const { result } = renderHook(() => useUpdateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act & Assert
      await expect(
        result.current.mutateAsync({
          experimentId: mockExperimentId,
          scheduledScorers: [mockUpdatedLLMScorer],
        }),
      ).rejects.toThrow(NotFoundError);

      expect(mockUpdateScheduledScorers).not.toHaveBeenCalled();
    });

    it('should handle updateScheduledScorers API failure', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const serverError = new InternalServerError({});
      mockListScheduledScorers.mockResolvedValue(mockExistingScorersResponse);
      mockUpdateScheduledScorers.mockRejectedValue(serverError);

      const { result } = renderHook(() => useUpdateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act & Assert
      await expect(
        result.current.mutateAsync({
          experimentId: mockExperimentId,
          scheduledScorers: [mockUpdatedLLMScorer],
        }),
      ).rejects.toThrow(InternalServerError);

      expect(mockListScheduledScorers).toHaveBeenCalledWith(mockExperimentId);
    });

    it('should handle missing existing scorers gracefully', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const emptyScorersResponse = {
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: [],
        },
      };

      mockListScheduledScorers.mockResolvedValue(emptyScorersResponse);
      mockUpdateScheduledScorers.mockResolvedValue({
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: [mockUpdatedScorerConfig],
        },
      });

      const { result } = renderHook(() => useUpdateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
        scheduledScorers: [mockUpdatedLLMScorer],
      });

      // Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      await mutationPromise;
      // Should add the new scorer to the empty list
      expect(mockUpdateScheduledScorers).toHaveBeenCalledWith(mockExperimentId, {
        scorers: [mockUpdatedScorerConfig],
      });

      // Verify cache was updated with the new scorer
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'original-scorer',
            sampleRate: 75,
            filterString: 'updated_column = "value"',
            type: 'llm',
            llmTemplate: 'Guidelines',
            guidelines: ['Updated guideline 1', 'Updated guideline 2'],
          },
        ],
      });
    });

    it('should handle response with undefined scheduled_scorers', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const responseWithUndefinedScorers = {
        experiment_id: mockExperimentId,
      };

      mockListScheduledScorers.mockResolvedValue(responseWithUndefinedScorers);
      mockUpdateScheduledScorers.mockResolvedValue(mockUpdateResponse);

      const { result } = renderHook(() => useUpdateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
        scheduledScorers: [mockUpdatedLLMScorer],
      });

      // Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      await mutationPromise;
      // Should handle undefined scorers as empty array
      expect(mockUpdateScheduledScorers).toHaveBeenCalledWith(mockExperimentId, {
        scorers: [mockUpdatedScorerConfig],
      });

      // Verify cache was updated correctly despite undefined input
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: [
          {
            name: 'original-scorer',
            sampleRate: 75,
            filterString: 'updated_column = "value"',
            type: 'llm',
            llmTemplate: 'Guidelines',
            guidelines: ['Updated guideline 1', 'Updated guideline 2'],
          },
        ],
      });
    });
  });
});
