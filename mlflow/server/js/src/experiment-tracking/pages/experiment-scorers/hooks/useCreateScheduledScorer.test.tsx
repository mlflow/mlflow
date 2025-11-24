import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { renderHook, waitFor } from '@testing-library/react';
import { InternalServerError } from '@databricks/web-shared/errors';
import { useCreateScheduledScorerMutation } from './useCreateScheduledScorer';
import { registerScorer, type RegisterScorerResponse } from '../api';
import { transformScheduledScorer } from '../utils/scorerTransformUtils';

// Mock external dependencies
jest.mock('../api');

const mockRegisterScorer = jest.mocked(registerScorer);

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

  const mockRegisterResponse: RegisterScorerResponse = {
    version: 1,
    scorer_id: 'scorer-id-123',
    experiment_id: mockExperimentId,
    name: 'test-llm-scorer',
    serialized_scorer: '{"type": "llm", "guidelines": ["Test guideline 1", "Test guideline 2"]}',
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

  describe('Golden Path - Successful Operations', () => {
    it('should successfully register a new LLM scorer', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      mockRegisterScorer.mockResolvedValue(mockRegisterResponse);

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

      // Verify the response structure
      expect(response).toMatchObject({
        experiment_id: mockExperimentId,
        scheduled_scorers: {
          scorers: [
            {
              scorer_version: 1,
              serialized_scorer: '{"type": "llm", "guidelines": ["Test guideline 1", "Test guideline 2"]}',
            },
          ],
        },
      });

      // Verify registerScorer was called with transformed config
      const expectedTransformedConfig = transformScheduledScorer(mockLLMScorer);
      expect(mockRegisterScorer).toHaveBeenCalledWith(mockExperimentId, expectedTransformedConfig);

      // Verify cache was updated
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: expect.arrayContaining([
          expect.objectContaining({
            name: 'test-llm-scorer',
            version: 1,
          }),
        ]),
      });
    });

    it('should successfully register a new custom code scorer', async () => {
      // Arrange
      const customCodeResponse: RegisterScorerResponse = {
        version: 1,
        scorer_id: 'scorer-id-456',
        experiment_id: mockExperimentId,
        name: 'test-custom-scorer',
        serialized_scorer:
          '{"type": "custom-code", "code": "def my_scorer(inputs, outputs):\\n    return {\\"score\\": 1}"}',
        creation_time: 1234567891,
      };

      mockRegisterScorer.mockResolvedValue(customCodeResponse);

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
      const expectedTransformedConfig = transformScheduledScorer(mockCustomCodeScorer);
      expect(mockRegisterScorer).toHaveBeenCalledWith(mockExperimentId, expectedTransformedConfig);

      // Verify cache was updated
      const updatedCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(updatedCache).toEqual({
        experimentId: mockExperimentId,
        scheduledScorers: expect.arrayContaining([
          expect.objectContaining({
            name: 'test-custom-scorer',
            version: 1,
          }),
        ]),
      });
    });
  });

  describe('Edge Cases and Error Conditions', () => {
    it('should handle registerScorer API failure', async () => {
      // Arrange
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(initialCache).toBeUndefined();

      const networkError = new InternalServerError({});
      mockRegisterScorer.mockRejectedValue(networkError);

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

      expect(mockRegisterScorer).toHaveBeenCalled();

      // Verify cache was not modified due to error
      const cacheAfterError = queryClient.getQueryData(['mlflow', 'scheduled-scorers', mockExperimentId]);
      expect(cacheAfterError).toBeUndefined();
    });

    it('should transform scorer config correctly before calling API', async () => {
      // Arrange
      mockRegisterScorer.mockResolvedValue(mockRegisterResponse);

      const { result } = renderHook(() => useCreateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      await result.current.mutateAsync({
        experimentId: mockExperimentId,
        scheduledScorer: mockLLMScorer,
      });

      // Assert - verify the transformation was applied
      const expectedTransformedConfig = transformScheduledScorer(mockLLMScorer);
      expect(mockRegisterScorer).toHaveBeenCalledWith(mockExperimentId, expectedTransformedConfig);
      expect(mockRegisterScorer).toHaveBeenCalledTimes(1);
    });

    it('should convert response correctly to ScorerConfig format', async () => {
      // Arrange
      const customResponse: RegisterScorerResponse = {
        version: 5,
        scorer_id: 'unique-id-789',
        experiment_id: '999',
        name: 'custom-scorer-name',
        serialized_scorer: '{"custom": "data"}',
        creation_time: 9999999999,
      };

      mockRegisterScorer.mockResolvedValue(customResponse);

      const { result } = renderHook(() => useCreateScheduledScorerMutation(), {
        wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
      });

      // Act
      const response = await result.current.mutateAsync({
        experimentId: '999',
        scheduledScorer: mockLLMScorer,
      });

      // Assert - verify the response conversion
      expect(response.scheduled_scorers?.scorers[0]).toMatchObject({
        scorer_version: 5,
        serialized_scorer: '{"custom": "data"}',
      });
    });
  });
});
