import { jest, describe, test, expect, it, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';

import { IntlProvider } from '@databricks/i18n';
import type { Assessment, FeedbackAssessment, ModelTraceInfoV3 } from '../../model-trace-explorer/ModelTrace.types';
import { TracesServiceV4, fetchTraceInfoV3 } from '../../model-trace-explorer/api';
import { getAssessmentValue } from '../../model-trace-explorer/assessments-pane/utils';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

import { useGenAiTraceEvaluationArtifacts } from './useGenAiTraceEvaluationArtifacts';
import {
  createMlflowSearchFilter,
  extractTraceIdFromSearchQuery,
  getSearchMlflowTracesQueryCacheConfig,
  invalidateMlflowSearchTracesCache,
  useMlflowTracesTableMetadata,
  useSearchMlflowTraces,
} from './useMlflowTraces';
import {
  EXECUTION_DURATION_COLUMN_ID,
  LINKED_PROMPTS_COLUMN_ID,
  SESSION_COLUMN_ID,
  LOGGED_MODEL_COLUMN_ID,
  SPAN_NAME_COLUMN_ID,
  SPAN_TYPE_COLUMN_ID,
  SPAN_STATUS_COLUMN_ID,
  SPAN_CONTENT_COLUMN_ID,
  STATE_COLUMN_ID,
  INPUTS_COLUMN_ID,
  RESPONSE_COLUMN_ID,
} from './useTableColumns';
import { FilterOperator, TracesTableColumnGroup, TracesTableColumnType } from '../types';
import { shouldUseInfinitePaginatedTraces, shouldUseTracesV4API } from '../utils/FeatureUtils';
import { fetchAPI } from '../utils/FetchUtils';

// Mock shouldEnableUnifiedEvalTab
jest.mock('../utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../utils/FeatureUtils')>('../utils/FeatureUtils'),
  shouldEnableUnifiedEvalTab: jest.fn(),
  shouldUseTracesV4API: jest.fn().mockReturnValue(false),
  shouldUseInfinitePaginatedTraces: jest.fn().mockReturnValue(false),
  getMlflowTracesSearchPageSize: jest.fn().mockReturnValue(10000),
}));

// Mock the artifact hook
jest.mock('./useGenAiTraceEvaluationArtifacts', () => ({
  useGenAiTraceEvaluationArtifacts: jest.fn(),
}));

// Mock fetch utilities
jest.mock('../utils/FetchUtils', () => ({
  fetchAPI: jest.fn(),
  getAjaxUrl: jest.fn().mockImplementation((relativeUrl) => '/' + relativeUrl),
  getDefaultHeaders: jest.fn().mockReturnValue({}),
}));

// Mock fetchTraceInfoV3 from model-trace-explorer API
jest.mock('../../model-trace-explorer/api', () => ({
  ...jest.requireActual<typeof import('../../model-trace-explorer/api')>('../../model-trace-explorer/api'),
  fetchTraceInfoV3: jest.fn(),
}));

// Mock global window.fetch
// @ts-expect-error -- TODO(FEINF-4162)
jest.spyOn(global, 'fetch').mockImplementation();

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false, // Turn off retries to simplify testing
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <IntlProvider locale="en">
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </IntlProvider>
  );
}

describe('useMlflowTracesTableMetadata', () => {
  test('returns empty data and isLoading = false when disabled is true', async () => {
    const { result } = renderHook(
      () =>
        useMlflowTracesTableMetadata({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'some-experiment',
              },
            },
          ],
          disabled: true,
        }),
      {
        wrapper: createWrapper(),
      },
    );

    expect(result.current.isLoading).toBe(false);
  });

  test('extracts prompt options from single trace with single linked prompt', async () => {
    jest.mocked(useGenAiTraceEvaluationArtifacts).mockReturnValue({
      data: [],
      isLoading: false,
    } as any);

    jest.mocked(fetchAPI).mockResolvedValue({
      traces: [
        {
          trace_id: 'trace_1',
          tags: {
            'mlflow.linkedPrompts': JSON.stringify([{ name: 'qa-agent-system-prompt', version: '4' }]),
          },
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useMlflowTracesTableMetadata({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-prompt-options',
              },
            },
          ],
          runUuid: 'run-prompt-options',
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.tableFilterOptions.prompt).toHaveLength(1);
    expect(result.current.tableFilterOptions.prompt?.[0].value).toBe('qa-agent-system-prompt/4');
  });

  test('deduplicates prompt options across multiple traces', async () => {
    jest.mocked(useGenAiTraceEvaluationArtifacts).mockReturnValue({
      data: [],
      isLoading: false,
    } as any);

    jest.mocked(fetchAPI).mockResolvedValue({
      traces: [
        {
          trace_id: 'trace_1',
          tags: {
            'mlflow.linkedPrompts': JSON.stringify([
              { name: 'qa-agent-system-prompt', version: '4' },
              { name: 'chat-assistant-prompt', version: '1' },
            ]),
          },
        },
        {
          trace_id: 'trace_2',
          tags: {
            'mlflow.linkedPrompts': JSON.stringify([
              { name: 'qa-agent-system-prompt', version: '4' },
              { name: 'chat-assistant-prompt', version: '2' },
            ]),
          },
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useMlflowTracesTableMetadata({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-dedupe-prompts',
              },
            },
          ],
          runUuid: 'run-dedupe-prompts',
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.tableFilterOptions.prompt).toHaveLength(3);
    expect(result.current.tableFilterOptions.prompt?.map((p: any) => p.value)).toEqual([
      'chat-assistant-prompt/1',
      'chat-assistant-prompt/2',
      'qa-agent-system-prompt/4',
    ]);
  });

  test('handles invalid JSON in linkedPrompts tag gracefully', async () => {
    jest.mocked(useGenAiTraceEvaluationArtifacts).mockReturnValue({
      data: [],
      isLoading: false,
    } as any);

    jest.mocked(fetchAPI).mockResolvedValue({
      traces: [
        {
          trace_id: 'trace_1',
          tags: {
            'mlflow.linkedPrompts': 'invalid-json',
          },
        },
        {
          trace_id: 'trace_2',
          tags: {
            'mlflow.linkedPrompts': JSON.stringify([{ name: 'valid-prompt', version: '1' }]),
          },
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useMlflowTracesTableMetadata({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-invalid-json',
              },
            },
          ],
          runUuid: 'run-invalid-json',
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.tableFilterOptions.prompt).toHaveLength(1);
    expect(result.current.tableFilterOptions.prompt?.[0].value).toBe('valid-prompt/1');
  });
});

describe('getSearchMlflowTracesQueryCacheConfig', () => {
  it('returns staleTime: Infinity and cacheTime: Inifity for OSS (non-V4). keepPreviousData should be true and refetchOnWindowFocus should be false to prevent list bounce when search/filter changes', () => {
    const config = getSearchMlflowTracesQueryCacheConfig(false);
    expect(config).toEqual({
      staleTime: Infinity,
      cacheTime: Infinity,
      keepPreviousData: true,
      refetchOnWindowFocus: false,
    });
  });

  it('returns keepPreviousData and refetchOnWindowFocus: false for V4 APIs', () => {
    const config = getSearchMlflowTracesQueryCacheConfig(true);
    expect(config).toEqual({
      keepPreviousData: true,
      refetchOnWindowFocus: false,
    });
  });
});

describe('useSearchMlflowTraces', () => {
  beforeEach(() => {
    jest.mocked(shouldUseTracesV4API).mockReturnValue(false);
    jest.mocked(shouldUseInfinitePaginatedTraces).mockReturnValue(false);
    jest.mocked(fetchAPI).mockClear();
    jest.mocked(fetchTraceInfoV3).mockClear();
  });
  test('returns empty data and isLoading = false when disabled is true', async () => {
    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'some-experiment',
              },
            },
          ],
          disabled: true,
        }),
      {
        wrapper: createWrapper(),
      },
    );

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toEqual([]);
  });

  test('makes network call to fetch traces when enabled', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value"}',
          response: '{"output": "value"}',
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(fetchAPI).toHaveBeenCalledTimes(1); // only one page
    expect(result.current.data).toHaveLength(1);
    expect(result.current.data?.[0]).toEqual({
      trace_id: 'trace_1',
      request: '{"input": "value"}',
      response: '{"output": "value"}',
    });
  });

  test('uses filters to fetch traces', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value"}',
          response: '{"output": "value"}',
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          runUuid: 'run-xyz',
          timeRange: {
            startTime: '100',
            endTime: '200',
          },
          filters: [
            {
              column: TracesTableColumnGroup.ASSESSMENT,
              key: 'overall',
              operator: FilterOperator.EQUALS,
              value: 'success',
            },
            {
              column: TracesTableColumnGroup.TAG,
              operator: FilterOperator.EQUALS,
              value: 'user_1',
              key: 'user',
            },
            // no tag key so should be ignored
            {
              column: TracesTableColumnGroup.TAG,
              operator: FilterOperator.EQUALS,
              value: 'user_1',
            },
            {
              column: TracesTableColumnGroup.TAG,
              operator: FilterOperator.EQUALS,
              value: 'user_2',
              key: 'user',
            },
            {
              column: 'execution_duration',
              operator: FilterOperator.GREATER_THAN,
              value: 1000,
            },
            {
              column: 'user',
              operator: FilterOperator.EQUALS,
              value: 'user_3',
              key: 'user',
            },
            {
              column: 'run_name',
              operator: FilterOperator.EQUALS,
              value: 'run_1',
              key: 'run_name',
            },
            {
              column: LOGGED_MODEL_COLUMN_ID,
              operator: FilterOperator.EQUALS,
              value: 'version_1',
              key: 'version',
            },
            {
              column: 'state',
              operator: FilterOperator.EQUALS,
              value: 'OK',
            },
            {
              column: 'trace_name',
              operator: FilterOperator.EQUALS,
              value: 'trace_1',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;
    const expectedAssessmentFilter = "AND feedback.`overall` = 'success' ";

    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    expect(body).toEqual({
      locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
      filter: `attributes.run_id = 'run-xyz' AND attributes.timestamp_ms > 100 AND attributes.timestamp_ms < 200 ${expectedAssessmentFilter}AND tags.user = 'user_1' AND tags.user = 'user_2' AND attributes.execution_time_ms > 1000 AND request_metadata."mlflow.trace.user" = 'user_3' AND attributes.run_id = 'run_1' AND request_metadata."mlflow.modelId" = 'version_1' AND attributes.status = 'OK' AND attributes.name = 'trace_1'`,
      max_results: 10000,
    });
  });

  test('handles custom metadata filters correctly', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value"}',
          response: '{"output": "value"}',
          trace_metadata: {
            user_id: 'user123',
            environment: 'production',
            'mlflow.internal.key': 'internal_value', // Should be excluded
          },
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          filters: [
            {
              column: 'custom_metadata:user_id',
              operator: FilterOperator.EQUALS,
              value: 'user123',
            },
            {
              column: 'custom_metadata:environment',
              operator: FilterOperator.EQUALS,
              value: 'production',
            },
            {
              column: 'custom_metadata:mlflow.trace.session',
              operator: FilterOperator.CONTAINS,
              value: '',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;

    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    expect(body).toEqual({
      locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
      filter: `request_metadata.user_id = 'user123' AND request_metadata.environment = 'production' AND request_metadata.mlflow.trace.session ILIKE '%%'`,
      max_results: 10000,
    });
  });

  test('excludes MLflow internal keys from custom metadata filters', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value"}',
          response: '{"output": "value"}',
          trace_metadata: {
            user_id: 'user123',
            'mlflow.run_id': 'run123', // Should be excluded
            'mlflow.internal.key': 'internal_value', // Should be excluded
          },
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          filters: [
            {
              column: 'custom_metadata:user_id',
              operator: FilterOperator.EQUALS,
              value: 'user123',
            },
            {
              column: 'custom_metadata:mlflow.run_id', // This should be ignored
              operator: FilterOperator.EQUALS,
              value: 'run123',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;

    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    expect(body).toEqual({
      locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
      filter: `request_metadata.user_id = 'user123' AND request_metadata.mlflow.run_id = 'run123'`,
      max_results: 10000,
    });
  });

  test('handles span name filters with case-insensitive equals', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value"}',
          response: '{"output": "value"}',
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          filters: [
            {
              column: SPAN_NAME_COLUMN_ID,
              operator: FilterOperator.EQUALS,
              value: 'my_span',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;

    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    expect(body).toEqual({
      locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
      filter: `span.name ILIKE 'my_span'`,
      max_results: 10000,
    });
  });

  test('handles span name filters with CONTAINS operator', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value"}',
          response: '{"output": "value"}',
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          filters: [
            {
              column: SPAN_NAME_COLUMN_ID,
              operator: FilterOperator.CONTAINS,
              value: 'span',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;

    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    expect(body).toEqual({
      locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
      filter: `span.name ILIKE '%span%'`,
      max_results: 10000,
    });
  });

  test('handles span name filters with NOT_EQUALS operator', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value"}',
          response: '{"output": "value"}',
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          filters: [
            {
              column: SPAN_NAME_COLUMN_ID,
              operator: FilterOperator.NOT_EQUALS,
              value: 'excluded_span',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;

    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    expect(body).toEqual({
      locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
      filter: `span.name != 'excluded_span'`,
      max_results: 10000,
    });
  });

  test('handles span type filters with case-insensitive equals', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value"}',
          response: '{"output": "value"}',
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          filters: [
            {
              column: SPAN_TYPE_COLUMN_ID,
              operator: FilterOperator.EQUALS,
              value: 'llm',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;

    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    expect(body).toEqual({
      locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
      filter: `span.type ILIKE 'llm'`,
      max_results: 10000,
    });
  });

  test('handles span type filters with CONTAINS operator', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value"}',
          response: '{"output": "value"}',
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          filters: [
            {
              column: SPAN_TYPE_COLUMN_ID,
              operator: FilterOperator.CONTAINS,
              value: 'chain',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;

    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    expect(body).toEqual({
      locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
      filter: `span.type ILIKE '%chain%'`,
      max_results: 10000,
    });
  });

  test('handles span status filters with EQUALS operator', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value"}',
          response: '{"output": "value"}',
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          filters: [
            {
              column: SPAN_STATUS_COLUMN_ID,
              operator: FilterOperator.EQUALS,
              value: 'ERROR',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;

    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    expect(body).toEqual({
      locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
      filter: `span.status = 'ERROR'`,
      max_results: 10000,
    });
  });

  test('handles multiple span filters combined', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value"}',
          response: '{"output": "value"}',
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          filters: [
            {
              column: SPAN_NAME_COLUMN_ID,
              operator: FilterOperator.CONTAINS,
              value: 'query',
            },
            {
              column: SPAN_TYPE_COLUMN_ID,
              operator: FilterOperator.EQUALS,
              value: 'llm',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;

    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    expect(body).toEqual({
      locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
      filter: `span.name ILIKE '%query%' AND span.type ILIKE 'llm'`,
      max_results: 10000,
    });
  });

  test('handles span content filter with CONTAINS operator', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value"}',
          response: '{"output": "value"}',
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          filters: [
            {
              column: SPAN_CONTENT_COLUMN_ID,
              operator: FilterOperator.CONTAINS,
              value: 'search text',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;

    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    expect(body).toEqual({
      locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
      filter: `span.content ILIKE '%search text%'`,
      max_results: 10000,
    });
  });

  test.each([INPUTS_COLUMN_ID, RESPONSE_COLUMN_ID])('handles %s filter with RLIKE operator', async (testedColumn) => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          filters: [
            {
              column: testedColumn,
              operator: FilterOperator.RLIKE,
              value: 'hello.*world',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;

    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    expect(body).toEqual({
      locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
      filter: `${testedColumn} RLIKE 'hello.*world'`,
      max_results: 10000,
    });
  });

  test.each([INPUTS_COLUMN_ID, RESPONSE_COLUMN_ID])(
    'handles %s filter with "equals" operator',
    async (testedColumn) => {
      jest.mocked(fetchAPI).mockResolvedValueOnce({
        traces: [],
        next_page_token: undefined,
      });

      const { result } = renderHook(
        () =>
          useSearchMlflowTraces({
            locations: [
              {
                type: 'MLFLOW_EXPERIMENT',
                mlflow_experiment: {
                  experiment_id: 'experiment-xyz',
                },
              },
            ],
            filters: [
              {
                column: testedColumn,
                operator: FilterOperator.EQUALS,
                value: 'hello.*world',
              },
            ],
          }),
        {
          wrapper: createWrapper(),
        },
      );

      await waitFor(() => expect(result.current.isLoading).toBe(false));

      const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;

      expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
      expect(body).toEqual({
        locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
        filter: `${testedColumn} = 'hello.*world'`,
        max_results: 10000,
      });
    },
  );

  test('handles combined request and response filters', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          filters: [
            {
              column: INPUTS_COLUMN_ID,
              operator: FilterOperator.RLIKE,
              value: 'hello',
            },
            {
              column: RESPONSE_COLUMN_ID,
              operator: FilterOperator.EQUALS,
              value: 'world',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;

    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    expect(body).toEqual({
      locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
      filter: `request RLIKE 'hello' AND response = 'world'`,
      max_results: 10000,
    });
  });

  test('uses order_by to fetch traces for server sortable column', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value"}',
          response: '{"output": "value"}',
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          runUuid: 'run-xyz',
          timeRange: {
            startTime: '100',
            endTime: '200',
          },
          tableSort: {
            key: EXECUTION_DURATION_COLUMN_ID,
            asc: false,
            type: TracesTableColumnType.INPUT,
          },
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;

    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    expect(body).toEqual({
      locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
      filter: `attributes.run_id = 'run-xyz' AND attributes.timestamp_ms > 100 AND attributes.timestamp_ms < 200`,
      max_results: 10000,
      order_by: ['execution_time DESC'],
    });
  });

  test('Does not use order_by to fetch traces for non-server sortable column', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value"}',
          response: '{"output": "value"}',
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          runUuid: 'run-xyz',
          timeRange: {
            startTime: '100',
            endTime: '200',
          },
          tableSort: {
            key: SESSION_COLUMN_ID,
            asc: false,
            type: TracesTableColumnType.INPUT,
          },
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;

    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    expect(body).toEqual({
      locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
      filter: `attributes.run_id = 'run-xyz' AND attributes.timestamp_ms > 100 AND attributes.timestamp_ms < 200`,
      max_results: 10000,
      order_by: [],
    });
  });

  test("use loggedModelId and sqlWarehouseId to fetch model's online traces", async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value"}',
          response: '{"output": "value"}',
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          runUuid: 'run-xyz',
          loggedModelId: 'model-123',
          sqlWarehouseId: 'warehouse-456',
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;

    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    expect(body).toEqual({
      locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
      filter: `attributes.run_id = 'run-xyz'`,
      max_results: 10000,
      model_id: 'model-123',
      sql_warehouse_id: 'warehouse-456',
    });
  });

  it('handles assessment filters via backend', async () => {
    // Mock returns only matching trace (simulating backend filtering)
    jest.mocked(fetchAPI).mockResolvedValue({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value1"}',
          response: '{"output": "value"}',
          assessments: [
            {
              assessment_id: 'overall_assessment',
              assessment_name: 'overall_assessment',
              trace_id: 'trace_1',
              feedback: {
                value: 'pass',
              },
            },
            {
              assessment_id: 'correctness',
              assessment_name: 'correctness_assessment',
              trace_id: 'trace_1',
              feedback: {
                value: 'pass',
              },
            },
          ],
        },
      ] as ModelTraceInfoV3[],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          currentRunDisplayName: 'run-xyz',
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          runUuid: 'run-xyz',
          timeRange: {
            startTime: '100',
            endTime: '200',
          },
          filters: [
            {
              column: TracesTableColumnGroup.ASSESSMENT,
              key: 'overall_assessment',
              operator: FilterOperator.EQUALS,
              value: 'pass',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // Verify the assessment filter was sent to the backend
    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;
    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
    const expectedFilter = "feedback.`overall_assessment` = 'pass'";
    expect(body.filter).toContain(expectedFilter);

    expect(result.current.data).toHaveLength(1);
    expect(result.current.data?.[0].trace_id).toBe('trace_1');
    expect(result.current.data?.[0].assessments).toHaveLength(2);
    expect(result.current.data?.[0].assessments?.[0].assessment_id).toBe('overall_assessment');
    expect((result.current.data?.[0].assessments?.[0] as FeedbackAssessment)?.feedback?.value).toBe('pass');
    expect(result.current.data?.[0].assessments?.[1].assessment_id).toBe('correctness');
    expect((result.current.data?.[0].assessments?.[1] as FeedbackAssessment)?.feedback?.value).toBe('pass');
  });

  it('handles search query filtering via backend', async () => {
    jest.mocked(fetchAPI).mockResolvedValue({
      traces: [
        {
          trace_id: 'trace_1',
          request: '{"input": "value1"}',
          response: '{"output": "value"}',
        },
      ] as ModelTraceInfoV3[],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          searchQuery: 'test query',
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // Verify the search query was sent to the backend
    const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;
    expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');

    const expectedFilter = "trace.text ILIKE '%test query%'";
    expect(body.filter).toBe(expectedFilter);
  });

  it('fetches trace by ID when search query is a 32-char hex trace ID', async () => {
    const traceId = '11301f0bdf2dfa5a762a4bac74b45db1';

    // Mock the search API to return empty results
    jest.mocked(fetchAPI).mockResolvedValue({
      traces: [],
      next_page_token: undefined,
    });

    // Mock fetchTraceInfoV3 to return the trace when looked up by ID
    jest.mocked(fetchTraceInfoV3).mockImplementation(() =>
      Promise.resolve({
        trace: {
          trace_info: {
            trace_id: traceId,
            request_preview: '{"input": "found by ID"}',
            response_preview: '{"output": "result"}',
            state: 'OK',
          },
        },
      }),
    );

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'MLFLOW_EXPERIMENT',
              mlflow_experiment: {
                experiment_id: 'experiment-xyz',
              },
            },
          ],
          searchQuery: traceId,
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // Verify fetchTraceInfoV3 was called with the trace ID
    expect(jest.mocked(fetchTraceInfoV3)).toHaveBeenCalledWith({ traceId });

    // The trace should be found via the get_trace API and included in results
    expect(result.current.data).toBeDefined();
    expect(result.current.data?.some((t) => t.trace_id === traceId)).toBe(true);
  });

  it('fetches trace via V4 batch get when search query is a full V4 trace ID with known location', async () => {
    const traceId = 'aabbccdd11223344aabbccdd11223344';
    const searchQuery = `trace:/test_catalog.test_schema/${traceId}`;

    jest.mocked(shouldUseTracesV4API).mockReturnValue(true);

    // Mock the V4 search API to return empty results
    const searchSpy = jest.spyOn(TracesServiceV4, 'searchTracesV4').mockResolvedValue([]);

    // Mock getBatchTracesV4 to return the trace when looked up by ID + location
    const batchGetSpy = jest.spyOn(TracesServiceV4, 'getBatchTracesV4').mockResolvedValue({
      traces: [
        {
          trace_info: {
            trace_id: traceId,
            request_preview: '{"input": "found by V4 batch get"}',
            response_preview: '{"output": "result"}',
            state: 'OK',
            trace_location: {
              type: 'UC_SCHEMA',
              uc_schema: { catalog_name: 'test_catalog', schema_name: 'test_schema' },
            },
            request_time: '2026-03-26T00:00:00Z',
            tags: {},
          },
          spans: [],
        },
      ],
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'UC_SCHEMA',
              uc_schema: {
                catalog_name: 'test_catalog',
                schema_name: 'test_schema',
              },
            },
          ],
          searchQuery,
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // Verify getBatchTracesV4 was called with the parsed location from the trace ID
    expect(batchGetSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        traceIds: [traceId],
        traceLocation: {
          type: 'UC_SCHEMA',
          uc_schema: { catalog_name: 'test_catalog', schema_name: 'test_schema' },
        },
      }),
    );

    // fetchTraceInfoV3 should NOT have been called
    expect(jest.mocked(fetchTraceInfoV3)).not.toHaveBeenCalled();

    expect(result.current.data).toBeDefined();
    expect(result.current.data?.some((t) => t.trace_id === traceId)).toBe(true);

    searchSpy.mockRestore();
    batchGetSpy.mockRestore();
  });

  it('does not look up trace when V4 trace ID location does not match linked experiment locations', async () => {
    const traceId = 'aabbccdd11223344aabbccdd11223344';
    const searchQuery = `trace:/other_catalog.other_schema/${traceId}`;

    jest.mocked(shouldUseTracesV4API).mockReturnValue(true);

    // Mock the V4 search API to return empty results
    const searchSpy = jest.spyOn(TracesServiceV4, 'searchTracesV4').mockResolvedValue([]);

    const batchGetSpy = jest.spyOn(TracesServiceV4, 'getBatchTracesV4');

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'UC_SCHEMA',
              uc_schema: {
                catalog_name: 'my_catalog',
                schema_name: 'my_schema',
              },
            },
          ],
          searchQuery,
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // getBatchTracesV4 should NOT have been called — the location doesn't match
    expect(batchGetSpy).not.toHaveBeenCalled();

    // fetchTraceInfoV3 should NOT have been called either
    expect(jest.mocked(fetchTraceInfoV3)).not.toHaveBeenCalled();

    // No trace should be in the results
    expect(result.current.data).toEqual([]);

    searchSpy.mockRestore();
    batchGetSpy.mockRestore();
  });

  it('fetches trace via V4 sequential location fallback when search query is a hex ID', async () => {
    const traceId = 'eeff00112233445566778899aabbccdd';

    jest.mocked(shouldUseTracesV4API).mockReturnValue(true);

    // Mock the V4 search API to return empty results
    const searchSpy = jest.spyOn(TracesServiceV4, 'searchTracesV4').mockResolvedValue([]);

    // First location returns empty, second location returns the trace
    const batchGetSpy = jest
      .spyOn(TracesServiceV4, 'getBatchTracesV4')
      .mockResolvedValueOnce({ traces: [] })
      .mockResolvedValueOnce({
        traces: [
          {
            trace_info: {
              trace_id: traceId,
              request_preview: '{"input": "found on second location"}',
              response_preview: '{"output": "result"}',
              state: 'OK',
              trace_location: {
                type: 'UC_SCHEMA',
                uc_schema: { catalog_name: 'catalog_b', schema_name: 'schema_b' },
              },
              request_time: '2026-03-26T00:00:00Z',
              tags: {},
            },
            spans: [],
          },
        ],
      });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'UC_SCHEMA',
              uc_schema: {
                catalog_name: 'catalog_a',
                schema_name: 'schema_a',
              },
            },
            {
              type: 'UC_SCHEMA',
              uc_schema: {
                catalog_name: 'catalog_b',
                schema_name: 'schema_b',
              },
            },
          ],
          searchQuery: traceId,
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // Verify getBatchTracesV4 was called for both locations
    expect(batchGetSpy).toHaveBeenCalledTimes(2);
    expect(batchGetSpy).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        traceIds: [traceId],
        traceLocation: {
          type: 'UC_SCHEMA',
          uc_schema: { catalog_name: 'catalog_a', schema_name: 'schema_a' },
        },
      }),
    );
    expect(batchGetSpy).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        traceIds: [traceId],
        traceLocation: {
          type: 'UC_SCHEMA',
          uc_schema: { catalog_name: 'catalog_b', schema_name: 'schema_b' },
        },
      }),
    );

    // fetchTraceInfoV3 should NOT have been called
    expect(jest.mocked(fetchTraceInfoV3)).not.toHaveBeenCalled();

    expect(result.current.data).toBeDefined();
    expect(result.current.data?.some((t) => t.trace_id === traceId)).toBe(true);

    searchSpy.mockRestore();
    batchGetSpy.mockRestore();
  });

  it('uses server-side assessment filters when applicable', async () => {
    jest.mocked(shouldUseTracesV4API).mockReturnValue(true);

    const apiCallSpy = jest.spyOn(TracesServiceV4, 'searchTracesV4').mockResolvedValueOnce([]);

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          currentRunDisplayName: 'run-xyz',
          locations: [
            {
              type: 'UC_SCHEMA',
              uc_schema: {
                catalog_name: 'catalog',
                schema_name: 'schema',
              },
            },
          ],
          timeRange: {
            startTime: '100',
            endTime: '200',
          },
          filters: [
            {
              column: TracesTableColumnGroup.ASSESSMENT,
              key: 'overall_assessment',
              operator: FilterOperator.EQUALS,
              value: 'pass',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(apiCallSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        filter:
          "attributes.timestamp_ms > 100 AND attributes.timestamp_ms < 200 AND feedback.`overall_assessment` = 'pass'",
        locations: [{ type: 'UC_SCHEMA', uc_schema: { catalog_name: 'catalog', schema_name: 'schema' } }],
      }),
    );
  });

  it('uses server-side search query filtering for V4 APIs', async () => {
    jest.mocked(shouldUseTracesV4API).mockReturnValue(true);

    const apiCallSpy = jest.spyOn(TracesServiceV4, 'searchTracesV4').mockResolvedValueOnce([]);

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [
            {
              type: 'UC_SCHEMA',
              uc_schema: {
                catalog_name: 'catalog',
                schema_name: 'schema',
              },
            },
          ],
          searchQuery: 'test query',
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    const expectedFilter = "trace.text ILIKE '%test query%'";
    expect(apiCallSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        filter: expectedFilter,
        locations: [{ type: 'UC_SCHEMA', uc_schema: { catalog_name: 'catalog', schema_name: 'schema' } }],
      }),
    );
  });

  test('filters out assessments belong to another run', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          client_request_id: 'req-1',
          trace_location: {
            type: 'MLFLOW_EXPERIMENT',
            mlflow_experiment: { experiment_id: 'experiment-xyz' },
          },
          request_time: '2024-01-01T00:00:00Z',
          state: 'OK',
          assessments: [
            {
              assessment_id: 'a-1',
              assessment_name: 'overall',
              trace_id: 'trace_1',
              create_time: '2024-01-01T00:00:00Z',
              last_update_time: '2024-01-01T00:00:00Z',
              metadata: { 'mlflow.assessment.sourceRunId': 'run-xyz' },
              feedback: { value: 'pass' },
            },
            {
              assessment_id: 'a-2',
              assessment_name: 'overall',
              trace_id: 'trace_1',
              create_time: '2024-01-01T00:00:00Z',
              last_update_time: '2024-01-01T00:00:00Z',
              metadata: { 'mlflow.assessment.sourceRunId': 'other-run-1' },
              feedback: { value: 'pass' },
            },
            {
              assessment_id: 'a-3',
              assessment_name: 'guidelines',
              trace_id: 'trace_1',
              create_time: '2024-01-01T00:00:00Z',
              last_update_time: '2024-01-01T00:00:00Z',
              metadata: { 'mlflow.assessment.sourceRunId': 'other-run-2' },
              feedback: { value: 'pass' },
            },
          ],
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'experiment-xyz' } }],
          runUuid: 'run-xyz',
          filterByAssessmentSourceRun: true,
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.data).toHaveLength(1);
    expect(result.current.data?.[0].trace_id).toBe('trace_1');
    expect(result.current.data?.[0].assessments).toHaveLength(1);
    expect(result.current.data?.[0].assessments?.[0].assessment_id).toBe('a-1');
    expect((result.current.data?.[0].assessments?.[0] as FeedbackAssessment)?.feedback?.value).toBe('pass');
  });

  test('keep traces without assessments', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          client_request_id: 'req-1',
          trace_location: {
            type: 'MLFLOW_EXPERIMENT',
            mlflow_experiment: { experiment_id: 'experiment-xyz' },
          },
          request_time: '2024-01-01T00:00:00Z',
          state: 'OK',
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'experiment-xyz' } }],
          runUuid: 'run-xyz',
          filterByAssessmentSourceRun: true,
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.data).toHaveLength(1);
    expect(result.current.data?.[0].trace_id).toBe('trace_1');
  });

  test('keep traces with manual assessments', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          client_request_id: 'req-1',
          trace_location: {
            type: 'MLFLOW_EXPERIMENT',
            mlflow_experiment: { experiment_id: 'experiment-xyz' },
          },
          request_time: '2024-01-01T00:00:00Z',
          state: 'OK',
          assessments: [
            {
              assessment_id: 'a-1',
              assessment_name: 'overall',
              trace_id: 'trace_1',
              create_time: '2024-01-01T00:00:00Z',
              last_update_time: '2024-01-01T00:00:00Z',
              feedback: { value: 'pass' },
              source: {
                source_type: 'HUMAN',
                source_id: 'user-1',
              },
            },
            {
              assessment_id: 'a-2',
              assessment_name: 'overall',
              trace_id: 'trace_1',
              create_time: '2024-01-01T00:00:00Z',
              last_update_time: '2024-01-01T00:00:00Z',
              expectation: { value: 'expected value' },
              source: {
                source_type: 'HUMAN',
                source_id: 'user-2',
              },
            },
          ],
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'experiment-xyz' } }],
          runUuid: 'run-xyz',
          filterByAssessmentSourceRun: true,
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.data).toHaveLength(1);
    expect(result.current.data?.[0].trace_id).toBe('trace_1');
    expect(result.current.data?.[0].assessments).toHaveLength(2);
    expect(result.current.data?.[0].assessments?.[0].assessment_id).toBe('a-1');
    expect(result.current.data?.[0].assessments?.[0]?.source?.source_id).toBe('user-1');
    expect(getAssessmentValue(result.current.data?.[0].assessments?.[1] as Assessment)).toBe('expected value');
    expect(result.current.data?.[0].assessments?.[1]?.assessment_id).toBe('a-2');
    expect(result.current.data?.[0].assessments?.[1]?.source?.source_id).toBe('user-2');
  });

  test('filters out scorer traces', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          client_request_id: 'req-1',
          trace_location: {
            type: 'MLFLOW_EXPERIMENT',
            mlflow_experiment: { experiment_id: 'experiment-xyz' },
          },
          request_time: '2024-01-01T00:00:00Z',
          state: 'OK',
          tags: {
            'mlflow.trace.sourceScorer': 'scorer-1',
          },
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'experiment-xyz' } }],
          runUuid: 'run-xyz',
          filterByAssessmentSourceRun: true,
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.data).toHaveLength(0);
  });

  test('keeps traces when assessments match the run or lack metadata', async () => {
    jest.mocked(fetchAPI).mockResolvedValueOnce({
      traces: [
        {
          trace_id: 'trace_1',
          client_request_id: 'req-1',
          trace_location: {
            type: 'MLFLOW_EXPERIMENT',
            mlflow_experiment: { experiment_id: 'experiment-xyz' },
          },
          request_time: '2024-01-01T00:00:00Z',
          state: 'OK',
          assessments: [
            {
              assessment_id: 'a-1',
              assessment_name: 'overall',
              trace_id: 'trace_1',
              create_time: '2024-01-01T00:00:00Z',
              last_update_time: '2024-01-01T00:00:00Z',
              metadata: { 'mlflow.assessment.sourceRunId': 'run-xyz' },
              feedback: { value: 'pass' },
            },
            {
              assessment_id: 'a-2',
              assessment_name: 'guidelines',
              trace_id: 'trace_1',
              create_time: '2024-01-01T00:00:00Z',
              last_update_time: '2024-01-01T00:00:00Z',
              feedback: { value: 'pass' },
            },
          ],
        },
      ],
      next_page_token: undefined,
    });

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'experiment-xyz' } }],
          runUuid: 'run-xyz',
          filterByAssessmentSourceRun: true,
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.data).toHaveLength(1);
    expect(result.current.data?.[0].trace_id).toBe('trace_1');
    expect(result.current.data?.[0].assessments?.length).toBe(2);
  });

  describe('when shouldUseInfinitePaginatedTraces is true', () => {
    beforeEach(() => {
      jest.mocked(shouldUseInfinitePaginatedTraces).mockReturnValue(true);
    });

    test('fetches a single page with max_results=100 and exposes pagination state', async () => {
      jest.mocked(fetchAPI).mockResolvedValueOnce({
        traces: [{ trace_id: 'trace_1' }],
        next_page_token: undefined,
      });

      const { result } = renderHook(
        () =>
          useSearchMlflowTraces({
            locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'experiment-xyz' } }],
          }),
        { wrapper: createWrapper() },
      );

      await waitFor(() => expect(result.current.isLoading).toBe(false));

      expect(fetchAPI).toHaveBeenCalledTimes(1);
      const [url, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;
      expect(url).toEqual('/ajax-api/3.0/mlflow/traces/search');
      expect(body).toEqual({
        locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
        filter: undefined,
        max_results: 100,
        order_by: undefined,
      });
      expect(result.current.data).toHaveLength(1);
      expect(result.current.hasNextPage).toBe(false);
      expect(typeof result.current.fetchNextPage).toBe('function');
      expect(result.current.isFetchingNextPage).toBe(false);
    });

    test('surfaces hasNextPage when the server returns a next_page_token', async () => {
      jest.mocked(fetchAPI).mockResolvedValueOnce({
        traces: [{ trace_id: 'trace_1' }],
        next_page_token: 'token-page-2',
      });

      const { result } = renderHook(
        () =>
          useSearchMlflowTraces({
            locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'experiment-xyz' } }],
          }),
        { wrapper: createWrapper() },
      );

      await waitFor(() => expect(result.current.isLoading).toBe(false));

      expect(fetchAPI).toHaveBeenCalledTimes(1);
      expect(result.current.data).toHaveLength(1);
      expect(result.current.hasNextPage).toBe(true);
    });

    test('fetchNextPage passes page_token and appends results to the existing list', async () => {
      jest
        .mocked(fetchAPI)
        .mockResolvedValueOnce({
          traces: [{ trace_id: 'trace_1' }],
          next_page_token: 'token-page-2',
        })
        .mockResolvedValueOnce({
          traces: [{ trace_id: 'trace_2' }],
          next_page_token: undefined,
        });

      const { result } = renderHook(
        () =>
          useSearchMlflowTraces({
            locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'experiment-xyz' } }],
          }),
        { wrapper: createWrapper() },
      );

      await waitFor(() => expect(result.current.isLoading).toBe(false));
      expect(result.current.hasNextPage).toBe(true);

      result.current.fetchNextPage?.();

      await waitFor(() => expect(jest.mocked(fetchAPI)).toHaveBeenCalledTimes(2));
      await waitFor(() => expect(result.current.hasNextPage).toBe(false));

      const [, { body: secondBody }] = jest.mocked(fetchAPI).mock.calls[1] as any;
      expect(secondBody).toEqual({
        locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
        filter: undefined,
        max_results: 100,
        order_by: undefined,
        page_token: 'token-page-2',
      });

      expect(result.current.data?.map((t) => t.trace_id)).toEqual(['trace_1', 'trace_2']);
    });

    test('falls back to the eager-fetch path when enablePagination is false', async () => {
      jest.mocked(fetchAPI).mockResolvedValueOnce({
        traces: [{ trace_id: 'trace_1' }],
        next_page_token: undefined,
      });

      const { result } = renderHook(
        () =>
          useSearchMlflowTraces({
            locations: [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'experiment-xyz' } }],
            enablePagination: false,
          }),
        { wrapper: createWrapper() },
      );

      await waitFor(() => expect(result.current.isLoading).toBe(false));

      const [, { body }] = jest.mocked(fetchAPI).mock.lastCall as any;
      expect(body).toEqual(
        expect.objectContaining({
          max_results: 10000,
        }),
      );
      expect(result.current.hasNextPage).toBeUndefined();
      expect(result.current.fetchNextPage).toBeUndefined();
    });
  });
});

describe('invalidateMlflowSearchTracesCache', () => {
  test('invalidates queries with searchMlflowTraces key', () => {
    const queryClient = new QueryClient();
    const invalidateQueriesSpy = jest.spyOn(queryClient, 'invalidateQueries');

    // Call the function
    invalidateMlflowSearchTracesCache({ queryClient });

    // Verify that invalidateQueries was called with the correct key
    expect(invalidateQueriesSpy).toHaveBeenCalledTimes(1);
    // @ts-expect-error 'queryKey' does not exist in type
    expect(invalidateQueriesSpy).toHaveBeenCalledWith({ queryKey: ['searchMlflowTraces'] });
  });
});

describe('createMlflowSearchFilter', () => {
  test('creates correct filter string for linked prompts', () => {
    const networkFilters = [
      {
        column: LINKED_PROMPTS_COLUMN_ID,
        operator: FilterOperator.EQUALS,
        value: 'qa-agent-system-prompt/4',
      },
    ];

    const filterString = createMlflowSearchFilter(undefined, undefined, networkFilters);
    expect(filterString).toBe("prompt = 'qa-agent-system-prompt/4'");
  });

  test('combines prompt filter with other filters', () => {
    const networkFilters = [
      {
        column: LINKED_PROMPTS_COLUMN_ID,
        operator: FilterOperator.EQUALS,
        value: 'my-prompt/1',
      },
      {
        column: STATE_COLUMN_ID,
        operator: FilterOperator.EQUALS,
        value: 'OK',
      },
    ];

    const filterString = createMlflowSearchFilter(undefined, undefined, networkFilters);
    expect(filterString).toBe("prompt = 'my-prompt/1' AND attributes.status = 'OK'");
  });

  test('creates correct filter string for assessment IS NULL', () => {
    const networkFilters = [
      {
        column: TracesTableColumnGroup.ASSESSMENT,
        operator: FilterOperator.IS_NULL,
        key: 'uses_tools_appropriately',
        value: undefined,
      },
    ];

    const filterString = createMlflowSearchFilter(undefined, undefined, networkFilters);

    expect(filterString).toBe('feedback.`uses_tools_appropriately` IS NULL');
  });

  test('creates correct filter string for assessment IS NOT NULL', () => {
    const networkFilters = [
      {
        column: TracesTableColumnGroup.ASSESSMENT,
        operator: FilterOperator.IS_NOT_NULL,
        key: 'safety_score',
        value: undefined,
      },
    ];

    const filterString = createMlflowSearchFilter(undefined, undefined, networkFilters);

    expect(filterString).toBe('feedback.`safety_score` IS NOT NULL');
  });

  test('combines assessment IS NULL/IS NOT NULL filters with other filters', () => {
    const networkFilters = [
      {
        column: TracesTableColumnGroup.ASSESSMENT,
        operator: FilterOperator.IS_NOT_NULL,
        key: 'overall_assessment',
        value: undefined,
      },
      {
        column: STATE_COLUMN_ID,
        operator: FilterOperator.EQUALS,
        value: 'OK',
      },
    ];

    const filterString = createMlflowSearchFilter(undefined, undefined, networkFilters);

    expect(filterString).toContain('feedback.`overall_assessment` IS NOT NULL');
    expect(filterString).toContain("attributes.status = 'OK'");
    expect(filterString).toContain(' AND ');
  });

  test('creates correct filter string for tag IS NULL', () => {
    const networkFilters = [
      {
        column: TracesTableColumnGroup.TAG,
        operator: FilterOperator.IS_NULL,
        key: 'environment',
        value: undefined,
      },
    ];

    const filterString = createMlflowSearchFilter(undefined, undefined, networkFilters);

    expect(filterString).toBe('tags.environment IS NULL');
  });

  test('creates correct filter string for tag IS NOT NULL', () => {
    const networkFilters = [
      {
        column: TracesTableColumnGroup.TAG,
        operator: FilterOperator.IS_NOT_NULL,
        key: 'model_name',
        value: undefined,
      },
    ];

    const filterString = createMlflowSearchFilter(undefined, undefined, networkFilters);

    expect(filterString).toBe('tags.model_name IS NOT NULL');
  });

  test('creates correct filter string for tag IS NULL with special characters in key', () => {
    const networkFilters = [
      {
        column: TracesTableColumnGroup.TAG,
        operator: FilterOperator.IS_NULL,
        key: 'my.tag',
        value: undefined,
      },
    ];

    const filterString = createMlflowSearchFilter(undefined, undefined, networkFilters);

    expect(filterString).toBe('tags.`my.tag` IS NULL');
  });

  test('creates correct filter string for session ID equals', () => {
    const networkFilters = [
      {
        column: SESSION_COLUMN_ID,
        operator: FilterOperator.EQUALS,
        value: 'my-session-123',
      },
    ];

    const filterString = createMlflowSearchFilter(undefined, undefined, networkFilters);
    expect(filterString).toBe("request_metadata.mlflow.trace.session = 'my-session-123'");
  });

  test('creates correct filter string for session ID contains', () => {
    const networkFilters = [
      {
        column: SESSION_COLUMN_ID,
        operator: FilterOperator.CONTAINS,
        value: 'session',
      },
    ];

    const filterString = createMlflowSearchFilter(undefined, undefined, networkFilters);
    expect(filterString).toBe("request_metadata.mlflow.trace.session ILIKE '%session%'");
  });

  test('combines tag IS NULL/IS NOT NULL filters with other filters', () => {
    const networkFilters = [
      {
        column: TracesTableColumnGroup.TAG,
        operator: FilterOperator.IS_NOT_NULL,
        key: 'environment',
        value: undefined,
      },
      {
        column: STATE_COLUMN_ID,
        operator: FilterOperator.EQUALS,
        value: 'OK',
      },
    ];

    const filterString = createMlflowSearchFilter(undefined, undefined, networkFilters);

    expect(filterString).toContain('tags.environment IS NOT NULL');
    expect(filterString).toContain("attributes.status = 'OK'");
    expect(filterString).toContain(' AND ');
  });
});

describe('extractTraceIdFromSearchQuery', () => {
  test('extracts backend trace ID from full V4 trace ID', () => {
    const result = extractTraceIdFromSearchQuery(
      'trace:/euirim_non_arclight.complete_experiment_schema/11301f0bdf2dfa5a762a4bac74b45db1',
    );
    expect(result).toEqual({
      backendTraceId: '11301f0bdf2dfa5a762a4bac74b45db1',
      traceLocation: 'euirim_non_arclight.complete_experiment_schema',
    });
  });

  test('extracts backend trace ID from 32-char hex string', () => {
    const result = extractTraceIdFromSearchQuery('11301f0bdf2dfa5a762a4bac74b45db1');
    expect(result).toEqual({
      backendTraceId: '11301f0bdf2dfa5a762a4bac74b45db1',
    });
  });

  test('extracts backend trace ID from 32-char uppercase hex string', () => {
    const result = extractTraceIdFromSearchQuery('11301F0BDF2DFA5A762A4BAC74B45DB1');
    expect(result).toEqual({
      backendTraceId: '11301F0BDF2DFA5A762A4BAC74B45DB1',
    });
  });

  test('handles whitespace-padded trace IDs', () => {
    const result = extractTraceIdFromSearchQuery('  11301f0bdf2dfa5a762a4bac74b45db1  ');
    expect(result).toEqual({
      backendTraceId: '11301f0bdf2dfa5a762a4bac74b45db1',
    });
  });

  test('returns undefined for regular search queries', () => {
    expect(extractTraceIdFromSearchQuery('hello world')).toBeUndefined();
    expect(extractTraceIdFromSearchQuery('test query')).toBeUndefined();
    expect(extractTraceIdFromSearchQuery('')).toBeUndefined();
  });

  test('returns undefined for non-hex strings of 32 chars', () => {
    expect(extractTraceIdFromSearchQuery('this is not a valid hex string!')).toBeUndefined();
    expect(extractTraceIdFromSearchQuery('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')).toBeUndefined();
  });

  test('returns undefined for hex strings that are not 32 chars', () => {
    expect(extractTraceIdFromSearchQuery('11301f0bdf2dfa5a')).toBeUndefined();
    expect(extractTraceIdFromSearchQuery('11301f0bdf2dfa5a762a4bac74b45db1aa')).toBeUndefined();
  });

  test('returns undefined for invalid V4 trace ID format', () => {
    // trace:/ with missing parts
    expect(extractTraceIdFromSearchQuery('trace:/')).toBeUndefined();
  });

  test('extracts V4 trace ID with experiment location', () => {
    const result = extractTraceIdFromSearchQuery('trace:/experiment-123/abc123def456abc123def456abc123de');
    expect(result).toEqual({
      backendTraceId: 'abc123def456abc123def456abc123de',
      traceLocation: 'experiment-123',
    });
  });
});
