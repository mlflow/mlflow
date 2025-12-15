import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { renderHook, waitFor } from '@testing-library/react';
import { InternalServerError } from '@databricks/web-shared/errors';
import { useUpdateScheduledScorerMutation } from './useUpdateScheduledScorer';
import { registerScorer, type RegisterScorerResponse } from '../api';
import { transformScheduledScorer } from '../utils/scorerTransformUtils';

// Mock external dependencies
jest.mock('../api');

const mockRegisterScorer = jest.mocked(registerScorer);

describe('useUpdateScheduledScorerMutation', () => {
  let queryClient: QueryClient;

  const mockExperimentId = 'experiment-123';

  const mockUpdatedLLMScorer = {
    name: 'original-scorer',
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

  const mockRegisterResponse: RegisterScorerResponse = {
    version: 2,
    scorer_id: 'scorer-id-123',
    experiment_id: mockExperimentId,
    name: 'original-scorer',
    serialized_scorer: '{"type": "llm", "guidelines": ["Updated guideline 1", "Updated guideline 2"]}',
    creation_time: 1234567890,
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
    it('should successfully update a single scorer', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      mockRegisterScorer.mockResolvedValue(mockRegisterResponse);

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

      // Verify the response structure
      expect(response).toMatchObject({
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: [
            {
              scorer_version: 2,
              serialized_scorer: '{"type": "llm", "guidelines": ["Updated guideline 1", "Updated guideline 2"]}',
            },
          ],
        },
      });

      // Verify registerScorer was called with transformed config
      const expectedTransformedConfig = transformScheduledScorer(mockUpdatedLLMScorer);
      expect(mockRegisterScorer).toHaveBeenCalledWith(mockExperimentId, expectedTransformedConfig);
      expect(mockRegisterScorer).toHaveBeenCalledTimes(1);

      // Verify cache was updated
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: expect.arrayContaining([
          expect.objectContaining({
            name: 'original-scorer',
            version: 2,
          }),
        ]),
      });
    });

    it('should successfully update multiple scorers in parallel', async () => {
      // Arrange
      const scorer1 = { ...mockUpdatedLLMScorer, name: 'scorer-1' };
      const scorer2 = { ...mockUpdatedCustomCodeScorer, name: 'scorer-2' };

      const response1: RegisterScorerResponse = {
        version: 2,
        scorer_id: 'scorer-id-1',
        experiment_id: mockExperimentId,
        name: 'scorer-1',
        serialized_scorer: '{"type": "llm"}',
        creation_time: 1234567890,
      };

      const response2: RegisterScorerResponse = {
        version: 3,
        scorer_id: 'scorer-id-2',
        experiment_id: mockExperimentId,
        name: 'scorer-2',
        serialized_scorer: '{"type": "custom-code"}',
        creation_time: 1234567891,
      };

      // Mock registerScorer to return different responses based on the scorer name
      mockRegisterScorer.mockImplementation(async (experimentId, scorerConfig) => {
        if (scorerConfig.name === 'scorer-1') {
          return response1;
        }
        return response2;
      });

      const { result } = renderHook(() => useUpdateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
        scheduledScorers: [scorer1, scorer2],
      });

      // Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      const response = await mutationPromise;

      // Verify both scorers were updated
      expect(response.scheduled_scorers?.scorers).toHaveLength(2);
      expect(response.scheduled_scorers?.scorers[0]).toMatchObject({
        scorer_version: 2,
      });
      expect(response.scheduled_scorers?.scorers[1]).toMatchObject({
        scorer_version: 3,
      });

      // Verify registerScorer was called twice
      expect(mockRegisterScorer).toHaveBeenCalledTimes(2);

      const expectedTransformedConfig1 = transformScheduledScorer(scorer1);
      const expectedTransformedConfig2 = transformScheduledScorer(scorer2);
      expect(mockRegisterScorer).toHaveBeenCalledWith(mockExperimentId, expectedTransformedConfig1);
      expect(mockRegisterScorer).toHaveBeenCalledWith(mockExperimentId, expectedTransformedConfig2);

      // Verify cache was updated
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: expect.arrayContaining([
          expect.objectContaining({
            name: 'scorer-1',
            version: 2,
          }),
          expect.objectContaining({
            name: 'scorer-2',
            version: 3,
          }),
        ]),
      });
    });

    it('should handle custom code scorer updates', async () => {
      // Arrange
      const customCodeResponse: RegisterScorerResponse = {
        version: 1,
        scorer_id: 'scorer-id-custom',
        experiment_id: mockExperimentId,
        name: 'updated-custom-scorer',
        serialized_scorer:
          '{"type": "custom-code", "code": "def updated_scorer(inputs, outputs):\\n    return {\\"score\\": 0.8}"}',
        creation_time: 1234567891,
      };

      mockRegisterScorer.mockResolvedValue(customCodeResponse);

      const { result } = renderHook(() => useUpdateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const mutationPromise = result.current.mutateAsync({
        experimentId: mockExperimentId,
        scheduledScorers: [mockUpdatedCustomCodeScorer],
      });

      // Assert
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      const response = await mutationPromise;

      // Verify the response structure
      expect(response).toMatchObject({
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: [
            {
              scorer_version: 1,
            },
          ],
        },
      });

      // Verify registerScorer was called with transformed config
      const expectedTransformedConfig = transformScheduledScorer(mockUpdatedCustomCodeScorer);
      expect(mockRegisterScorer).toHaveBeenCalledWith(mockExperimentId, expectedTransformedConfig);
    });
  });

  describe('Edge Cases and Error Conditions', () => {
    it('should handle registerScorer API failure', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const serverError = new InternalServerError({});
      mockRegisterScorer.mockRejectedValue(serverError);

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

      expect(mockRegisterScorer).toHaveBeenCalled();

      // Verify cache was not modified due to error
      const cacheAfterError = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheAfterError).toBeUndefined();
    });

    it('should handle partial failure when updating multiple scorers', async () => {
      // Arrange
      const scorer1 = { ...mockUpdatedLLMScorer, name: 'scorer-1' };
      const scorer2 = { ...mockUpdatedCustomCodeScorer, name: 'scorer-2' };

      const serverError = new InternalServerError({});

      // Mock first call succeeds, second call fails
      mockRegisterScorer
        .mockResolvedValueOnce({
          version: 2,
          scorer_id: 'scorer-id-1',
          experiment_id: mockExperimentId,
          name: 'scorer-1',
          serialized_scorer: '{"type": "llm"}',
          creation_time: 1234567890,
        })
        .mockRejectedValueOnce(serverError);

      const { result } = renderHook(() => useUpdateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act & Assert
      await expect(
        result.current.mutateAsync({
          experimentId: mockExperimentId,
          scheduledScorers: [scorer1, scorer2],
        }),
      ).rejects.toThrow(InternalServerError);

      // Verify both registerScorer calls were made (Promise.all doesn't short-circuit)
      expect(mockRegisterScorer).toHaveBeenCalledTimes(2);

      // Verify cache was not modified due to error
      const cacheAfterError = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheAfterError).toBeUndefined();
    });

    it('should transform scorer config correctly before calling API', async () => {
      // Arrange
      mockRegisterScorer.mockResolvedValue(mockRegisterResponse);

      const { result } = renderHook(() => useUpdateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      await result.current.mutateAsync({
        experimentId: mockExperimentId,
        scheduledScorers: [mockUpdatedLLMScorer],
      });

      // Assert - verify the transformation was applied
      const expectedTransformedConfig = transformScheduledScorer(mockUpdatedLLMScorer);
      expect(mockRegisterScorer).toHaveBeenCalledWith(mockExperimentId, expectedTransformedConfig);
      expect(mockRegisterScorer).toHaveBeenCalledTimes(1);
    });

    it('should convert responses correctly to ScorerConfig format', async () => {
      // Arrange
      const scorer1 = { ...mockUpdatedLLMScorer, name: 'scorer-1' };
      const scorer2 = { ...mockUpdatedCustomCodeScorer, name: 'scorer-2' };

      const response1: RegisterScorerResponse = {
        version: 5,
        scorer_id: 'unique-id-1',
        experiment_id: '999',
        name: 'scorer-1',
        serialized_scorer: '{"custom": "data1"}',
        creation_time: 9999999991,
      };

      const response2: RegisterScorerResponse = {
        version: 6,
        scorer_id: 'unique-id-2',
        experiment_id: '999',
        name: 'scorer-2',
        serialized_scorer: '{"custom": "data2"}',
        creation_time: 9999999992,
      };

      mockRegisterScorer.mockImplementation(async (experimentId, scorerConfig) => {
        if (scorerConfig.name === 'scorer-1') {
          return response1;
        }
        return response2;
      });

      const { result } = renderHook(() => useUpdateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const response = await result.current.mutateAsync({
        experimentId: '999',
        scheduledScorers: [scorer1, scorer2],
      });

      // Assert - verify the response conversion
      expect(response.scheduled_scorers?.scorers[0]).toMatchObject({
        scorer_version: 5,
        serialized_scorer: '{"custom": "data1"}',
      });

      expect(response.scheduled_scorers?.scorers[1]).toMatchObject({
        scorer_version: 6,
        serialized_scorer: '{"custom": "data2"}',
      });
    });

    it('should handle empty scheduledScorers array', async () => {
      // Arrange
      const { result } = renderHook(() => useUpdateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const response = await result.current.mutateAsync({
        experimentId: mockExperimentId,
        scheduledScorers: [],
      });

      // Assert
      expect(response).toMatchObject({
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: [],
        },
      });

      // Verify registerScorer was not called
      expect(mockRegisterScorer).not.toHaveBeenCalled();
    });
  });
});
