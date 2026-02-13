import { jest, describe, test, expect, it, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';

import { IntlProvider } from '@databricks/i18n';
import type { Assessment, FeedbackAssessment, ModelTraceInfoV3 } from '../../model-trace-explorer/ModelTrace.types';
import { TracesServiceV4 } from '../../model-trace-explorer/api';
import { getAssessmentValue } from '../../model-trace-explorer/assessments-pane/utils';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

import { useGenAiTraceEvaluationArtifacts } from './useGenAiTraceEvaluationArtifacts';
import {
  createMlflowSearchFilter,
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
import { shouldUseTracesV4API } from '../utils/FeatureUtils';
import { fetchAPI } from '../utils/FetchUtils';

// Mock shouldEnableUnifiedEvalTab
jest.mock('../utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../utils/FeatureUtils')>('../utils/FeatureUtils'),
  shouldEnableUnifiedEvalTab: jest.fn(),
  shouldUseTracesV4API: jest.fn().mockReturnValue(false),
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
    jest.mocked(fetchAPI).mockClear();
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

    const [url, { body }] = jest.mocked(fetchFn).mock.lastCall as any;
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
    expect(JSON.parse(body).filter).toContain(expectedFilter);

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

    const expectedFilter = "span.attributes.mlflow.spanInputs ILIKE '%test query%'";
    expect(JSON.parse(body).filter).toBe(expectedFilter);
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
    const expectedFilter = "span.attributes.mlflow.spanInputs ILIKE '%test query%'";
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
});
