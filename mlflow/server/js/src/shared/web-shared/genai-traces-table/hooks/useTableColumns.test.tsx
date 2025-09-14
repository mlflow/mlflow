import { renderHook } from '@testing-library/react';

import type { IntlShape } from '@databricks/i18n';

import { useTableColumns, REQUEST_TIME_COLUMN_ID, createAssessmentColumnId } from './useTableColumns';
import type { RunEvaluationTracesDataEntry, AssessmentInfo } from '../types';
import { TracesTableColumnType, TracesTableColumnGroup } from '../types';
import { shouldUseTraceInfoV3 } from '../utils/TraceUtils';

jest.mock('../utils/TraceUtils', () => ({
  shouldUseTraceInfoV3: jest.fn(),
  createCustomMetadataColumnId: jest.fn((key: string) => `custom_metadata:${key}`),
  createTagColumnId: jest.fn((key: string) => `tag:${key}`),
  MLFLOW_INTERNAL_PREFIX: 'mlflow.',
}));

jest.mock('../utils/FeatureUtils', () => ({
  shouldEnableTagGrouping: jest.fn(() => false),
}));

describe('useTableColumns', () => {
  const baseEvalDataEntry: RunEvaluationTracesDataEntry = {
    evaluationId: 'eval-123',
    requestId: 'req-456',
    inputsId: 'eval-123',
    inputs: {},
    outputs: {},
    targets: {},
    errorCode: undefined,
    errorMessage: undefined,
    overallAssessments: [],
    responseAssessmentsByName: {},
    metrics: {},
    retrievalChunks: undefined,
    traceInfo: undefined,
  };

  const baseAssessmentInfo: AssessmentInfo = {
    name: 'overall_assessment',
    displayName: 'Overall',
    isKnown: true,
    isOverall: true,
    metricName: 'overall_assessment',
    source: undefined,
    isCustomMetric: false,
    isEditable: false,
    isRetrievalAssessment: false,
    dtype: 'pass-fail',
    uniqueValues: new Set(['yes', 'no']),
    docsLink: 'https://example.com/docs',
    missingTooltip: '',
    description: 'An example built-in assessment.',
    containsErrors: false,
  };

  const mockIntl: IntlShape = {
    formatMessage: ({ defaultMessage }: { defaultMessage: string }) => defaultMessage,
    // ... mock other IntlShape properties if needed
  } as IntlShape;

  const renderUseTableColumnsHook = (
    currentEvaluationResults: RunEvaluationTracesDataEntry[],
    assessmentInfos: AssessmentInfo[],
    runUuid?: string,
    otherEvaluationResults?: RunEvaluationTracesDataEntry[],
  ) => {
    return renderHook(() =>
      useTableColumns(mockIntl, currentEvaluationResults, assessmentInfos, runUuid, otherEvaluationResults),
    );
  };

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('returns only input columns + no additional info columns when Trace Info V3 is false', () => {
    // 2. Make sure the mock returns false for this scenario
    jest.mocked(shouldUseTraceInfoV3).mockReturnValue(false);

    const fakeResults: RunEvaluationTracesDataEntry[] = [
      {
        ...baseEvalDataEntry,
        inputs: { request: 'Hello world', kwarg2: 'value2' },
        traceInfo: undefined,
      },
      {
        ...baseEvalDataEntry,
        inputs: { request: 'Another input', kwarg1: 'value1' },
        traceInfo: undefined,
      },
    ];

    // Some fake assessments (none are retrieval-based here)
    const fakeAssessments: AssessmentInfo[] = [
      { ...baseAssessmentInfo, name: 'overall_assessment', displayName: 'Overall', isRetrievalAssessment: false },
    ];

    const { result } = renderUseTableColumnsHook(fakeResults, fakeAssessments);

    expect(result.current).toHaveLength(4);

    // Check the columns we expect
    const [col1, col2, col3, col4] = result.current;
    expect(col1.id).toBe('request');
    expect(col1.type).toBe(TracesTableColumnType.INPUT);

    expect(col2.id).toBe('kwarg2');
    expect(col2.type).toBe(TracesTableColumnType.INPUT);

    expect(col3.id).toBe('kwarg1');
    expect(col3.type).toBe(TracesTableColumnType.INPUT);

    expect(col4.id).toBe(createAssessmentColumnId('overall_assessment'));
    expect(col4.type).toBe(TracesTableColumnType.ASSESSMENT);
  });

  it('returns standard columns when Trace Info V3 is true (request + trace info columns + assessments)', () => {
    jest.mocked(shouldUseTraceInfoV3).mockReturnValue(true);

    // In V3, the hook ignores inputs except for a single "Request" column
    const fakeResults: RunEvaluationTracesDataEntry[] = [
      {
        ...baseEvalDataEntry,
        inputs: { userQuery: 'Hello world' },
        traceInfo: {
          request: '',
          execution_duration: '123',
          request_time: '456',
          state: 'OK',
          tags: {},
          trace_id: '',
          trace_location: {} as any,
        },
      },
    ];

    const fakeAssessments: AssessmentInfo[] = [
      {
        ...baseAssessmentInfo,
        name: 'quality',
        displayName: 'Quality Score',
        isRetrievalAssessment: false,
      },
    ];

    const { result } = renderUseTableColumnsHook(fakeResults, fakeAssessments);

    // Expected columns:
    // 1) request (INPUT)
    // 2) trace_id (TRACE_INFO)
    // 3) trace_name (TRACE_INFO)
    // 4) response (TRACE_INFO)
    // 5) user (TRACE_INFO)
    // 6) session (TRACE_INFO)
    // 7) execution_duration (TRACE_INFO)
    // 8) request_time (TRACE_INFO)
    // 9) state (TRACE_INFO)
    // 10) source (TRACE_INFO)
    // 11) run_name (TRACE_INFO)
    // 12) tags (TRACE_INFO)
    // 13) quality (ASSESSMENT)
    expect(result.current).toHaveLength(15);

    const colIds = result.current.map((c) => c.id);
    expect(colIds).toContain('request');
    expect(colIds).toContain('trace_id');
    expect(colIds).toContain('trace_name');
    expect(colIds).toContain('response');
    expect(colIds).toContain('user');
    expect(colIds).toContain('session');
    expect(colIds).toContain('execution_duration');
    expect(colIds).toContain(REQUEST_TIME_COLUMN_ID);
    expect(colIds).toContain('state');
    expect(colIds).toContain('source');
    expect(colIds).toContain('run_name');
    expect(colIds).toContain('tags');
    expect(colIds).toContain(createAssessmentColumnId('quality'));
    expect(colIds).toContain('logged_model');
    expect(colIds).toContain('tokens');
  });

  it('returns standard columns when run id is provided', () => {
    jest.mocked(shouldUseTraceInfoV3).mockReturnValue(true);

    // In V3, the hook ignores inputs except for a single "Request" column
    const fakeResults: RunEvaluationTracesDataEntry[] = [
      {
        ...baseEvalDataEntry,
        inputs: { userQuery: 'Hello world' },
        traceInfo: {
          request: '',
          execution_duration: '123',
          request_time: '456',
          state: 'OK',
          tags: {},
          trace_id: '',
          trace_location: {} as any,
        },
      },
    ];

    const fakeAssessments: AssessmentInfo[] = [
      {
        ...baseAssessmentInfo,
        name: 'quality',
        displayName: 'Quality Score',
        isRetrievalAssessment: false,
      },
    ];

    const { result } = renderUseTableColumnsHook(fakeResults, fakeAssessments, 'run-123');

    // Expected columns:
    // 1) request (INPUT)
    // 2) trace_id (TRACE_INFO)
    // 3) trace_name (TRACE_INFO)
    // 4) response (TRACE_INFO)
    // 5) user (TRACE_INFO)
    // 6) session (TRACE_INFO)
    // 7) execution_duration (TRACE_INFO)
    // 8) request_time (TRACE_INFO)
    // 9) state (TRACE_INFO)
    // 10) source (TRACE_INFO)
    // 11) tags (TRACE_INFO)
    // 12) quality (ASSESSMENT)
    expect(result.current).toHaveLength(14);

    const colIds = result.current.map((c) => c.id);
    expect(colIds).toContain('request');
    expect(colIds).toContain('trace_id');
    expect(colIds).toContain('trace_name');
    expect(colIds).toContain('response');
    expect(colIds).toContain('user');
    expect(colIds).toContain('session');
    expect(colIds).toContain('execution_duration');
    expect(colIds).toContain(REQUEST_TIME_COLUMN_ID);
    expect(colIds).toContain('state');
    expect(colIds).toContain('source');
    expect(colIds).toContain('tags');
    expect(colIds).toContain(createAssessmentColumnId('quality'));
    expect(colIds).not.toContain('run_name');
    expect(colIds).toContain('logged_model');
    expect(colIds).toContain('tokens');
  });

  it('excludes retrieval-based assessments from columns', () => {
    jest.mocked(shouldUseTraceInfoV3).mockReturnValue(false);

    const fakeResults: RunEvaluationTracesDataEntry[] = [{ ...baseEvalDataEntry, inputs: { request: 'Hello world' } }];

    const fakeAssessments: AssessmentInfo[] = [
      {
        ...baseAssessmentInfo,
        name: 'overall_assessment',
        displayName: 'Overall Assessment',
        isRetrievalAssessment: false,
      },
      {
        ...baseAssessmentInfo,
        name: 'retrieval_only',
        displayName: 'Retrieval Only',
        isRetrievalAssessment: true,
      },
    ];

    const { result } = renderUseTableColumnsHook(fakeResults, fakeAssessments);

    // We only expect:
    //  1) request (INPUT)
    //  2) overall_assessment (ASSESSMENT)
    // The 'retrieval_only' column should be filtered out

    expect(result.current).toHaveLength(2);

    const colIds = result.current.map((col) => col.id);
    expect(colIds).toEqual(['request', createAssessmentColumnId('overall_assessment')]);
  });

  it('handles an empty results and assessments array gracefully', () => {
    jest.mocked(shouldUseTraceInfoV3).mockReturnValue(false);

    const fakeResults: RunEvaluationTracesDataEntry[] = [];
    const fakeAssessments: AssessmentInfo[] = [];

    const { result } = renderUseTableColumnsHook(fakeResults, fakeAssessments);

    // With no data, there should be no columns at all
    expect(result.current).toHaveLength(0);
  });

  it('includes custom metadata columns when Trace Info V3 is true and custom metadata is present', () => {
    jest.mocked(shouldUseTraceInfoV3).mockReturnValue(true);

    const fakeResults: RunEvaluationTracesDataEntry[] = [
      {
        ...baseEvalDataEntry,
        inputs: { userQuery: 'Hello world' },
        traceInfo: {
          request: '',
          execution_duration: '123',
          request_time: '456',
          state: 'OK',
          tags: {},
          trace_id: '',
          trace_location: {} as any,
          trace_metadata: {
            user_id: 'user123',
            environment: 'production',
            deployment_version: 'v1.2.3',
          },
        },
      },
    ];

    const fakeAssessments: AssessmentInfo[] = [
      {
        ...baseAssessmentInfo,
        name: 'quality',
        displayName: 'Quality Score',
        isRetrievalAssessment: false,
      },
    ];

    const { result } = renderUseTableColumnsHook(fakeResults, fakeAssessments);

    // Expected columns:
    // 1) request (INPUT)
    // 2-15) standard trace info columns (trace_id, trace_name, response, user, session, execution_duration, request_time, state, source, logged_model, tokens, run_name, tags)
    // 16-18) custom metadata columns (user_id, environment, deployment_version)
    // 19) quality (ASSESSMENT)
    expect(result.current).toHaveLength(18);

    const colIds = result.current.map((c) => c.id);
    expect(colIds).toContain('custom_metadata:user_id');
    expect(colIds).toContain('custom_metadata:environment');
    expect(colIds).toContain('custom_metadata:deployment_version');

    // Verify custom metadata columns have correct properties
    const userIdColumn = result.current.find((c) => c.id === 'custom_metadata:user_id');
    expect(userIdColumn).toEqual({
      id: 'custom_metadata:user_id',
      label: 'user_id',
      type: TracesTableColumnType.TRACE_INFO,
      group: TracesTableColumnGroup.INFO,
    });
  });

  it('excludes MLflow internal keys from custom metadata columns', () => {
    jest.mocked(shouldUseTraceInfoV3).mockReturnValue(true);

    const fakeResults: RunEvaluationTracesDataEntry[] = [
      {
        ...baseEvalDataEntry,
        inputs: { userQuery: 'Hello world' },
        traceInfo: {
          request: '',
          execution_duration: '123',
          request_time: '456',
          state: 'OK',
          tags: {},
          trace_id: '',
          trace_location: {} as any,
          trace_metadata: {
            user_id: 'user123',
            'mlflow.internal.key': 'internal_value',
            'mlflow.run_id': 'run123',
            environment: 'production',
          },
        },
      },
    ];

    const fakeAssessments: AssessmentInfo[] = [];

    const { result } = renderUseTableColumnsHook(fakeResults, fakeAssessments);

    const colIds = result.current.map((c) => c.id);
    // Should include custom metadata
    expect(colIds).toContain('custom_metadata:user_id');
    expect(colIds).toContain('custom_metadata:environment');
    // Should exclude MLflow internal keys
    expect(colIds).not.toContain('custom_metadata:mlflow.internal.key');
    expect(colIds).not.toContain('custom_metadata:mlflow.run_id');
  });

  it('handles multiple results with different custom metadata keys', () => {
    jest.mocked(shouldUseTraceInfoV3).mockReturnValue(true);

    const fakeResults: RunEvaluationTracesDataEntry[] = [
      {
        ...baseEvalDataEntry,
        inputs: { userQuery: 'Hello world' },
        traceInfo: {
          request: '',
          execution_duration: '123',
          request_time: '456',
          state: 'OK',
          tags: {},
          trace_id: '',
          trace_location: {} as any,
          trace_metadata: {
            user_id: 'user123',
            environment: 'production',
          },
        },
      },
      {
        ...baseEvalDataEntry,
        inputs: { userQuery: 'Another query' },
        traceInfo: {
          request: '',
          execution_duration: '456',
          request_time: '789',
          state: 'OK',
          tags: {},
          trace_id: '',
          trace_location: {} as any,
          trace_metadata: {
            deployment_version: 'v1.2.3',
            region: 'us-west-2',
          },
        },
      },
    ];

    const fakeAssessments: AssessmentInfo[] = [];

    const { result } = renderUseTableColumnsHook(fakeResults, fakeAssessments);

    const colIds = result.current.map((c) => c.id);
    // Should include all unique custom metadata keys from both results
    expect(colIds).toContain('custom_metadata:user_id');
    expect(colIds).toContain('custom_metadata:environment');
    expect(colIds).toContain('custom_metadata:deployment_version');
    expect(colIds).toContain('custom_metadata:region');

    // Should not duplicate keys
    const customMetadataColumns = colIds.filter((id) => id.startsWith('custom_metadata:'));
    expect(customMetadataColumns).toHaveLength(4);
  });

  it('handles results with no trace_metadata gracefully', () => {
    jest.mocked(shouldUseTraceInfoV3).mockReturnValue(true);

    const fakeResults: RunEvaluationTracesDataEntry[] = [
      {
        ...baseEvalDataEntry,
        inputs: { userQuery: 'Hello world' },
        traceInfo: {
          request: '',
          execution_duration: '123',
          request_time: '456',
          state: 'OK',
          tags: {},
          trace_id: '',
          trace_location: {} as any,
          // No trace_metadata
        },
      },
    ];

    const fakeAssessments: AssessmentInfo[] = [];

    const { result } = renderUseTableColumnsHook(fakeResults, fakeAssessments);

    const colIds = result.current.map((c) => c.id);
    // Should not include any custom metadata columns
    const customMetadataColumns = colIds.filter((id) => id.startsWith('custom_metadata:'));
    expect(customMetadataColumns).toHaveLength(0);
  });

  it('handles empty trace_metadata object gracefully', () => {
    jest.mocked(shouldUseTraceInfoV3).mockReturnValue(true);

    const fakeResults: RunEvaluationTracesDataEntry[] = [
      {
        ...baseEvalDataEntry,
        inputs: { userQuery: 'Hello world' },
        traceInfo: {
          request: '',
          execution_duration: '123',
          request_time: '456',
          state: 'OK',
          tags: {},
          trace_id: '',
          trace_location: {} as any,
          trace_metadata: {}, // Empty object
        },
      },
    ];

    const fakeAssessments: AssessmentInfo[] = [];

    const { result } = renderUseTableColumnsHook(fakeResults, fakeAssessments);

    const colIds = result.current.map((c) => c.id);
    // Should not include any custom metadata columns
    const customMetadataColumns = colIds.filter((id) => id.startsWith('custom_metadata:'));
    expect(customMetadataColumns).toHaveLength(0);
  });

  it('includes custom metadata columns from otherEvaluationResults when provided', () => {
    jest.mocked(shouldUseTraceInfoV3).mockReturnValue(true);

    const fakeResults: RunEvaluationTracesDataEntry[] = [
      {
        ...baseEvalDataEntry,
        inputs: { userQuery: 'Hello world' },
        traceInfo: {
          request: '',
          execution_duration: '123',
          request_time: '456',
          state: 'OK',
          tags: {},
          trace_id: '',
          trace_location: {} as any,
          trace_metadata: {
            user_id: 'user123',
          },
        },
      },
    ];

    const otherResults: RunEvaluationTracesDataEntry[] = [
      {
        ...baseEvalDataEntry,
        inputs: { userQuery: 'Other query' },
        traceInfo: {
          request: '',
          execution_duration: '789',
          request_time: '012',
          state: 'OK',
          tags: {},
          trace_id: '',
          trace_location: {} as any,
          trace_metadata: {
            environment: 'staging',
            deployment_version: 'v2.0.0',
          },
        },
      },
    ];

    const fakeAssessments: AssessmentInfo[] = [];

    const { result } = renderUseTableColumnsHook(fakeResults, fakeAssessments, undefined, otherResults);

    const colIds = result.current.map((c) => c.id);
    // Should include custom metadata from both current and other results
    expect(colIds).toContain('custom_metadata:user_id');
    expect(colIds).toContain('custom_metadata:environment');
    expect(colIds).toContain('custom_metadata:deployment_version');
  });
});
