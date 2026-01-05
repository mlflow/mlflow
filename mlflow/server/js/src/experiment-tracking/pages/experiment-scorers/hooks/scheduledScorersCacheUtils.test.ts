import { describe, beforeEach, it, expect, jest } from '@jest/globals';
import { QueryClient } from '@databricks/web-shared/query-client';
import { updateScheduledScorersCache } from './scheduledScorersCacheUtils';
import type { CreateScheduledScorersResponse } from './useCreateScheduledScorer';
import type { ScheduledScorersResponse } from './useGetScheduledScorers';

describe('scheduledScorersCacheUtils', () => {
  describe('updateScheduledScorersCache', () => {
    let queryClient: QueryClient;
    const experimentId = '2091164207513440';

    beforeEach(() => {
      queryClient = new QueryClient({
        defaultOptions: {
          queries: { retry: false },
        },
      });
    });

    it('should optimistically update cache when response is valid and experiment ID matches', () => {
      // Arrange - Set up existing cache data
      const existingCacheData: ScheduledScorersResponse = {
        experimentId: '2091164207513440',
        scheduledScorers: [
          {
            name: 'existing_scorer',
            sampleRate: 50,
            type: 'llm',
            llmTemplate: 'Correctness',
          },
        ],
      };

      queryClient.setQueryData(['mlflow', 'scheduled-scorers', experimentId], existingCacheData);

      const validResponse: CreateScheduledScorersResponse = {
        experiment_id: '2091164207513440',
        scheduled_scorers: {
          scorers: [
            {
              name: 'guidelines_scorer',
              serialized_scorer:
                '{"name": "guidelines", "builtin_scorer_class": "Guidelines", "builtin_scorer_pydantic_data": {"guidelines": ["Be helpful", "Be accurate"]}}',
              builtin: { name: 'guidelines' },
              sample_rate: 0.8,
              filter_string: 'status = "completed"',
              scorer_version: 1,
            },
            {
              name: 'custom_scorer',
              serialized_scorer: '{"name": "custom", "call_source": "return True"}',
              custom: {},
              sample_rate: 0.6,
              scorer_version: 1,
            },
          ],
        },
      };

      // Act
      updateScheduledScorersCache(queryClient, validResponse, experimentId);

      // Assert - Verify cache was updated with transformed data
      const updatedCache = queryClient.getQueryData<ScheduledScorersResponse>([
        'mlflow',
        'scheduled-scorers',
        experimentId,
      ]);

      // Scorers are sorted alphabetically by name (custom_scorer < guidelines_scorer)
      expect(updatedCache).toEqual({
        experimentId: '2091164207513440',
        scheduledScorers: [
          {
            name: 'custom_scorer',
            sampleRate: 60, // 0.6 * 100
            type: 'custom-code',
            code: 'return True',
            version: 1,
            disableMonitoring: true,
            originalFuncName: undefined,
            callSignature: undefined,
          },
          {
            name: 'guidelines_scorer',
            sampleRate: 80, // 0.8 * 100
            filterString: 'status = "completed"',
            type: 'llm',
            llmTemplate: 'Guidelines',
            guidelines: ['Be helpful', 'Be accurate'],
            version: 1,
            disableMonitoring: true,
            is_instructions_judge: false,
            model: undefined,
          },
        ],
      });
    });

    it('should update cache when no existing data and response is valid', () => {
      // Arrange - No existing cache data
      const initialCache = queryClient.getQueryData(['mlflow', 'scheduled-scorers', experimentId]);
      expect(initialCache).toBeUndefined();

      const validResponse: CreateScheduledScorersResponse = {
        experiment_id: '2091164207513440',
        scheduled_scorers: {
          scorers: [
            {
              name: 'correctness_scorer',
              serialized_scorer: '{"name": "correctness", "builtin_scorer_class": "Correctness"}',
              builtin: { name: 'correctness' },
              sample_rate: 0.5,
              scorer_version: 1,
            },
          ],
        },
      };

      // Act
      updateScheduledScorersCache(queryClient, validResponse, experimentId);

      // Assert
      const updatedCache = queryClient.getQueryData<ScheduledScorersResponse>([
        'mlflow',
        'scheduled-scorers',
        experimentId,
      ]);

      expect(updatedCache).toEqual({
        experimentId: '2091164207513440',
        scheduledScorers: [
          {
            name: 'correctness_scorer',
            sampleRate: 50, // 0.5 * 100
            type: 'llm',
            llmTemplate: 'Correctness',
            version: 1,
            disableMonitoring: true,
            is_instructions_judge: false,
            model: undefined,
          },
        ],
      });
    });

    it('should handle empty scorers array in valid response', () => {
      // Arrange
      const validResponseWithEmptyScorers: CreateScheduledScorersResponse = {
        experiment_id: '2091164207513440',
        scheduled_scorers: {
          scorers: [],
        },
      };

      // Act
      updateScheduledScorersCache(queryClient, validResponseWithEmptyScorers, experimentId);

      // Assert
      const updatedCache = queryClient.getQueryData<ScheduledScorersResponse>([
        'mlflow',
        'scheduled-scorers',
        experimentId,
      ]);

      expect(updatedCache).toEqual({
        experimentId: '2091164207513440',
        scheduledScorers: [],
      });
    });

    it('should invalidate queries when response structure is invalid', () => {
      // Arrange
      const spy = jest.spyOn(queryClient, 'invalidateQueries');
      const invalidResponse = {
        // Missing required fields
        experiment_id: undefined,
        scheduled_scorers: undefined,
      } as any;

      // Act
      updateScheduledScorersCache(queryClient, invalidResponse, experimentId);

      // Assert - Should have called invalidateQueries instead of setQueryData
      expect(spy).toHaveBeenCalledWith(['mlflow', 'scheduled-scorers', experimentId]);

      // Verify cache was not updated
      const cacheData = queryClient.getQueryData(['mlflow', 'scheduled-scorers', experimentId]);
      expect(cacheData).toBeUndefined();
    });

    it('should invalidate queries when experiment ID mismatch occurs', () => {
      // Arrange
      const existingCacheData: ScheduledScorersResponse = {
        experimentId: 'different-experiment-id',
        scheduledScorers: [],
      };
      queryClient.setQueryData(['mlflow', 'scheduled-scorers', experimentId], existingCacheData);

      const spy = jest.spyOn(queryClient, 'invalidateQueries');
      const responseWithMismatchedId: CreateScheduledScorersResponse = {
        experiment_id: 'another-experiment-id', // Different from cache and function parameter
        scheduled_scorers: {
          scorers: [
            {
              name: 'test_scorer',
              serialized_scorer: '{"name": "test", "builtin_scorer_class": "Correctness"}',
              builtin: { name: 'test' },
              sample_rate: 0.5,
              scorer_version: 1,
            },
          ],
        },
      };

      // Act
      updateScheduledScorersCache(queryClient, responseWithMismatchedId, experimentId);

      // Assert
      expect(spy).toHaveBeenCalledWith(['mlflow', 'scheduled-scorers', experimentId]);

      // Verify existing cache data is still unchanged
      const cacheDataAfter = queryClient.getQueryData(['mlflow', 'scheduled-scorers', experimentId]);
      expect(cacheDataAfter).toEqual(existingCacheData);
    });

    it('should handle undefined/null response gracefully', () => {
      // Arrange
      const spy = jest.spyOn(queryClient, 'invalidateQueries');

      // Act
      updateScheduledScorersCache(queryClient, undefined as any, experimentId);

      // Assert
      expect(spy).toHaveBeenCalledWith(['mlflow', 'scheduled-scorers', experimentId]);
    });
  });
});
