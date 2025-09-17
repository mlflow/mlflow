import { uniq } from 'lodash';

import { ModelTraceSpanType } from '@databricks/web-shared/model-trace-explorer';

import {
  convertTraceInfoV3ToModelTraceInfo,
  getRetrievedContextFromTrace,
  getTracesTagKeys,
  MLFLOW_INTERNAL_PREFIX,
  applyTraceInfoV3ToEvalEntry,
  getCustomMetadataKeyFromColumnId,
  createCustomMetadataColumnId,
  getTagKeyFromColumnId,
  createTagColumnId,
} from './TraceUtils';
import { KnownEvaluationResultAssessmentName } from '../components/GenAiEvaluationTracesReview.utils';
import type { RunEvaluationResultAssessment, RunEvaluationTracesDataEntry, TraceInfoV3 } from '../types';

const makeTrace = (spans: any[], tags: Record<string, string> | undefined = undefined) =>
  ({
    info: { tags },
    data: { spans },
  } as any); // Cast to any to avoid bringing full ModelTrace typings into the test

const makeAssessment = (name: string, rest: Partial<RunEvaluationResultAssessment>) =>
  ({
    name,
    ...rest,
  } as Partial<RunEvaluationResultAssessment>); // Cast to any to keep the test lightweight

describe('getTracesTagKeys', () => {
  it('returns unique tag keys that do not start with the internal prefix', () => {
    const traces: any[] = [
      {
        tags: {
          tag1: 'value1',
          [`${MLFLOW_INTERNAL_PREFIX}secret`]: 'dont_include',
        },
      },
      {
        tags: {
          tag2: 'value2',
          tag1: 'differentValue', // duplicate key "tag1"
        },
      },
    ];

    // Expected keys: "tag1" and "tag2"
    const expected = uniq(['tag1', 'tag2']);
    const result = getTracesTagKeys(traces);
    expect(result.sort()).toEqual(expected.sort());
  });

  it('returns an empty array if no tags are present', () => {
    const traces: any[] = [{ traceInfo: { tags: {} } }, { traceInfo: undefined }];
    expect(getTracesTagKeys(traces)).toEqual([]);
  });

  it('ignores keys that are falsey', () => {
    const traces: any[] = [
      {
        tags: {
          '': 'empty', // falsey key
          tag3: 'value3',
        },
      },
    ];
    const result = getTracesTagKeys(traces);
    expect(result).toEqual(['tag3']);
  });
});

describe('convertTraceInfoV3ToModelTraceInfo', () => {
  it('uses client_request_id if available', () => {
    const trace: Partial<TraceInfoV3> = {
      trace_id: 'trace-id-123',
      client_request_id: 'client-req-456',
      tags: {
        alpha: 'a',
        beta: 'b',
      },
    };

    const result = convertTraceInfoV3ToModelTraceInfo(trace as TraceInfoV3);
    expect(result.request_id).toEqual('client-req-456');

    // Check that tags are converted to an array of { key, value }
    expect(result.tags).toEqual([
      { key: 'alpha', value: 'a' },
      { key: 'beta', value: 'b' },
    ]);
  });

  it('falls back to trace_id if client_request_id is not provided', () => {
    const trace: Partial<TraceInfoV3> = {
      trace_id: 'trace-id-789',
      tags: {
        gamma: 'c',
      },
    };
    const result = convertTraceInfoV3ToModelTraceInfo(trace as TraceInfoV3);
    expect(result.request_id).toEqual('trace-id-789');
    expect(result.tags).toEqual([{ key: 'gamma', value: 'c' }]);
  });

  it('returns undefined tags if no tags are provided', () => {
    const trace: Partial<TraceInfoV3> = {
      trace_id: 'trace-id-000',
      client_request_id: 'client-req-000',
      // tags is omitted
    };
    const result = convertTraceInfoV3ToModelTraceInfo(trace as TraceInfoV3);
    expect(result.tags).toBeUndefined();
  });
});

describe('applyTraceInfoV3ToEvalEntry', () => {
  const traceInfoWithExpectations: TraceInfoV3 = {
    trace_id: 'trace123',
    trace_location: {
      type: 'MLFLOW_EXPERIMENT',
      mlflow_experiment: { experiment_id: 'exp123' },
    },
    request_time: '2023-10-01T00:00:00Z',
    state: 'OK',
    client_request_id: 'client456',
    request: '{"messages": [{"content": "Hello"}]}',
    response: '{"data": "output"}',
    tags: { key1: 'value1' },
    assessments: [
      {
        assessment_id: 'a-fe16ebce1999476c95d5b73cf78e47f8',
        assessment_name: 'json_array_expect',
        trace_id: 'tr-2f6e2efbac5d2eb3f9b46a67e973a533',
        source: {
          source_type: 'HUMAN',
          source_id: 'test',
        },
        create_time: '2025-09-17T04:13:36.753Z',
        last_update_time: '2025-09-17T04:13:36.753Z',
        expectation: {
          serialized_value: {
            serialization_format: 'JSON_FORMAT',
            value: '["str", 123, {"nested": "obj"}]',
          },
        },
        rationale: '',
      },
      {
        assessment_id: 'a-4eb4ea163a9a4b41935f953a63a36706',
        assessment_name: 'number_expect',
        trace_id: 'tr-2f6e2efbac5d2eb3f9b46a67e973a533',
        source: {
          source_type: 'HUMAN',
          source_id: 'test',
        },
        create_time: '2025-09-17T04:13:58.740Z',
        last_update_time: '2025-09-17T04:13:58.740Z',
        expectation: {
          value: 1234,
        },
        rationale: '',
      },
      {
        assessment_id: 'a-c180c83a996042a99b735540292b30ba',
        assessment_name: 'json_obj_expect',
        trace_id: 'tr-2f6e2efbac5d2eb3f9b46a67e973a533',
        source: {
          source_type: 'HUMAN',
          source_id: 'test',
        },
        create_time: '2025-09-17T04:13:12.356Z',
        last_update_time: '2025-09-17T04:13:12.356Z',
        expectation: {
          serialized_value: {
            serialization_format: 'JSON_FORMAT',
            value: '{"test": "value"}',
          },
        },
        rationale: '',
      },
      {
        assessment_id: 'a-5ebe632dbdd8489ebb1b2c3a3f703632',
        assessment_name: 'str_expect',
        trace_id: 'tr-2f6e2efbac5d2eb3f9b46a67e973a533',
        source: {
          source_type: 'HUMAN',
          source_id: 'test',
        },
        create_time: '2025-09-17T04:14:11.445Z',
        last_update_time: '2025-09-17T04:14:11.445Z',
        expectation: {
          value: 'test',
        },
        rationale: '',
      },
      {
        assessment_id: 'a-2243eda2cc5644109ebbc8c3271266cc',
        assessment_name: 'bool_expect',
        trace_id: 'tr-2f6e2efbac5d2eb3f9b46a67e973a533',
        source: {
          source_type: 'HUMAN',
          source_id: 'test',
        },
        create_time: '2025-09-17T04:10:11.897Z',
        last_update_time: '2025-09-17T04:10:11.897Z',
        expectation: {
          value: true,
        },
        rationale: '',
      },
    ],
  } as TraceInfoV3;

  const dummyTraceInfo: TraceInfoV3 = {
    trace_id: 'trace123',
    trace_location: {
      type: 'MLFLOW_EXPERIMENT',
      mlflow_experiment: { experiment_id: 'exp123' },
    },
    request_time: '2023-10-01T00:00:00Z',
    state: 'OK',
    client_request_id: 'client456',
    request: '{"messages": [{"content": "Hello"}]}',
    response: '{"data": "output"}',
    tags: { key1: 'value1' },
    assessments: [
      // Expectation assessment: will be processed into targets.
      {
        assessment_id: 'exp123',
        trace_id: 'trace123',
        create_time: '2023-10-01T00:00:00Z',
        last_update_time: '2023-10-01T00:00:00Z',
        assessment_name: 'expTest',
        expectation: { value: 'Expected target' },
        feedback: undefined,
        error: undefined,
        metadata: {},
        rationale: '',
      },
      // Feedback assessment: will be processed into responseAssessmentsByName.
      {
        assessment_id: 'feed123',
        trace_id: 'trace123',
        create_time: '2023-10-01T00:00:00Z',
        last_update_time: '2023-10-01T00:00:00Z',
        assessment_name: 'feedTest',
        expectation: undefined,
        feedback: { value: 5 },
        error: undefined,
        metadata: {},
        rationale: '',
        source: {
          source_type: 'HUMAN',
          source_id: 'me',
        },
      },
      // Overall assessment: will be processed into overallAssessments.
      {
        assessment_id: 'overall123',
        trace_id: 'trace123',
        create_time: '2023-10-01T00:00:00Z',
        last_update_time: '2023-10-01T00:00:00Z',
        assessment_name: KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT,
        expectation: undefined,
        feedback: { value: 'pass' },
        error: undefined,
        metadata: {
          root_cause_assessment: 'rootCause123',
          root_cause_rationale: 'rationale123',
          suggested_action: 'action123',
        },
        rationale: '',
        source: {
          source_type: 'LLM_JUDGE',
          source_id: 'databricks',
        },
      },
      // An assessment to ignore.
      {
        assessment_id: 'ignore123',
        trace_id: 'trace123',
        create_time: '2023-10-01T00:00:00Z',
        last_update_time: '2023-10-01T00:00:00Z',
        assessment_name: 'agent/latency_seconds',
        expectation: { value: 'ignore_me' },
        feedback: undefined,
        error: undefined,
        metadata: {},
        rationale: '',
      },
    ],
  };

  // Build a base RunEvaluationTracesDataEntry whose traceInfo is our dummy.
  const baseEvalEntry: RunEvaluationTracesDataEntry = {
    evaluationId: 'baseId',
    requestId: 'baseId',
    inputsId: 'baseId',
    inputsTitle: '',
    inputs: {},
    outputs: {},
    targets: {},
    overallAssessments: [],
    responseAssessmentsByName: {},
    metrics: {},
    retrievalChunks: [],
    traceInfo: dummyTraceInfo,
  };

  it('should correctly convert a complex TraceInfoV3 entry', () => {
    const input: RunEvaluationTracesDataEntry[] = [baseEvalEntry];

    // Call the function under test.
    const result = applyTraceInfoV3ToEvalEntry(input);
    expect(result).toHaveLength(1);

    // Build the expected output.
    const expected: RunEvaluationTracesDataEntry = {
      // Evaluation data is replaced by converted fields
      evaluationId: dummyTraceInfo.trace_id, // "trace123"
      requestId: dummyTraceInfo.client_request_id || '', // "client456"
      inputsId: dummyTraceInfo.trace_id, // same as trace_id
      // inputsTitle should be the content of the last message in the "messages" array.
      inputsTitle: 'Hello',
      // inputs: parsed from the request field.
      inputs: { messages: [{ content: 'Hello' }] },
      // outputs: parsed from the response field.
      outputs: { response: { data: 'output' } },
      // targets: from the expectation assessment "expTest"
      targets: { expTest: 'Expected target' },
      // overallAssessments
      overallAssessments: [
        {
          name: KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT,
          // The metadata is parsed into rootCauseAssessment and suggestedActions.
          rootCauseAssessment: {
            assessmentName: 'rootCause123',
            suggestedActions: 'action123',
          },
          stringValue: 'pass',
          booleanValue: undefined,
          numericValue: undefined,
          errorCode: undefined,
          errorMessage: undefined,
          rationale: 'rationale123',
          metadata: {
            root_cause_assessment: 'rootCause123',
            root_cause_rationale: 'rationale123',
            suggested_action: 'action123',
          },
          source: {
            sourceType: 'AI_JUDGE',
            sourceId: 'databricks',
            metadata: {},
          },
        },
      ],
      // responseAssessmentsByName: feedback assessment "feedTest" is processed.
      responseAssessmentsByName: {
        feedTest: [
          {
            name: 'feedTest',
            // Since the feedback value is a string, it is assigned as stringValue.
            stringValue: undefined,
            booleanValue: undefined,
            numericValue: 5,
            errorCode: undefined,
            errorMessage: undefined,
            rationale: '',
            rootCauseAssessment: undefined,
            metadata: {},
            source: {
              sourceType: 'HUMAN',
              sourceId: 'me',
              metadata: {},
            },
          },
        ],
        overall_assessment: [
          {
            name: KnownEvaluationResultAssessmentName.OVERALL_ASSESSMENT,
            // The metadata is parsed into rootCauseAssessment and suggestedActions.
            rootCauseAssessment: {
              assessmentName: 'rootCause123',
              suggestedActions: 'action123',
            },
            stringValue: 'pass',
            booleanValue: undefined,
            numericValue: undefined,
            errorCode: undefined,
            errorMessage: undefined,
            rationale: 'rationale123',
            metadata: {
              root_cause_assessment: 'rootCause123',
              root_cause_rationale: 'rationale123',
              suggested_action: 'action123',
            },
            source: {
              sourceType: 'AI_JUDGE',
              sourceId: 'databricks',
              metadata: {},
            },
          },
        ],
      },
      metrics: {},
      retrievalChunks: [],
      // The original traceInfo is preserved.
      traceInfo: dummyTraceInfo,
    };

    expect(result[0]).toEqual(expected);
  });

  it('should correctly convert TraceInfoV3 expectations', () => {
    const input: RunEvaluationTracesDataEntry[] = [{ ...baseEvalEntry, traceInfo: traceInfoWithExpectations }];
    const result = applyTraceInfoV3ToEvalEntry(input);

    expect(result[0].targets).toEqual({
      // JSON should be parsed
      json_array_expect: ['str', 123, { nested: 'obj' }],
      json_obj_expect: { test: 'value' },
      // primitives should remain as primitives
      number_expect: 1234,
      str_expect: 'test',
      bool_expect: true,
    });
  });
});

describe('getRetrievedContextFromTrace', () => {
  it('returns undefined when trace is nil', () => {
    expect(getRetrievedContextFromTrace({}, undefined)).toBeUndefined();
  });

  it('returns an empty array when no retriever spans exist', () => {
    const traceWithoutRetriever = makeTrace([
      {
        attributes: { 'mlflow.spanType': JSON.stringify('LLM') },
      },
    ]);

    expect(getRetrievedContextFromTrace({}, traceWithoutRetriever)).toEqual([]);
  });

  it('returns an empty array when retriever span has no outputs', () => {
    const spanMissingOutputs = {
      attributes: {
        'mlflow.spanType': JSON.stringify(ModelTraceSpanType.RETRIEVER),
      },
    };
    const trace = makeTrace([spanMissingOutputs]);

    expect(getRetrievedContextFromTrace({}, trace)).toEqual([]);
  });

  it('parses outputs and filters assessments by chunk index', () => {
    /* ---------- minimal trace with one LLM span + one RETRIEVER span -------- */
    const trace = {
      info: {},
      data: {
        spans: [
          {
            attributes: {
              'mlflow.spanType': 'LLM', // non‑retrieval span, don't JSON.stringify it to verify it doesn't throw an error
            },
          },
          {
            attributes: {
              'mlflow.spanType': JSON.stringify(ModelTraceSpanType.RETRIEVER),
              'mlflow.spanOutputs': JSON.stringify([
                {
                  page_content: 'Document 1',
                  metadata: { doc_uri: 's3://mybucket/doc1' },
                },
                {
                  page_content: 'Document 2',
                  metadata: { doc_uri: 's3://mybucket/doc2' },
                },
              ]),
            },
          },
        ],
      },
    } as any;

    /* ---------- assessments: two chunk‑specific + one generic --------------- */
    const responseAssessmentsByName: any = {
      chunk_relevance: [
        makeAssessment('chunk_relevance', {
          stringValue: 'yes',
          metadata: { span_output_key: '0' },
        }),
        makeAssessment('chunk_relevance', {
          stringValue: 'no',
          metadata: { span_output_key: '1' },
        }),
      ],
      relevance: [makeAssessment('relevance', {})], // no span_output_key → filtered out
    };

    const result = getRetrievedContextFromTrace(responseAssessmentsByName, trace);

    /* ---------- assertions -------------------------------------------------- */
    expect(result).toHaveLength(2);

    // Chunk 0
    expect(result?.[0].docUrl).toBe('s3://mybucket/doc1');
    expect(result?.[0].content).toBe('Document 1');
    expect(result?.[0].retrievalAssessmentsByName?.['chunk_relevance']).toHaveLength(1);
    expect(result?.[0].retrievalAssessmentsByName?.['chunk_relevance']?.[0].stringValue).toBe('yes');
    expect(result?.[0].retrievalAssessmentsByName?.['relevance']).toBeUndefined();

    // Chunk 1
    expect(result?.[1].docUrl).toBe('s3://mybucket/doc2');
    expect(result?.[1].content).toBe('Document 2');
    expect(result?.[1].retrievalAssessmentsByName?.['chunk_relevance']).toHaveLength(1);
    expect(result?.[1].retrievalAssessmentsByName?.['chunk_relevance']?.[0].stringValue).toBe('no');
    expect(result?.[1].retrievalAssessmentsByName?.['relevance']).toBeUndefined();
  });
});

describe('getCustomMetadataKeyFromColumnId', () => {
  it('extracts metadata key from column ID', () => {
    expect(getCustomMetadataKeyFromColumnId('custom_metadata:user_id')).toBe('user_id');
    expect(getCustomMetadataKeyFromColumnId('custom_metadata:environment')).toBe('environment');
    expect(getCustomMetadataKeyFromColumnId('custom_metadata:deployment_version')).toBe('deployment_version');
  });

  it('handles column IDs with multiple colons', () => {
    expect(getCustomMetadataKeyFromColumnId('custom_metadata:user:profile:id')).toBe('id');
  });

  it('returns empty string for invalid column ID', () => {
    expect(getCustomMetadataKeyFromColumnId('custom_metadata:')).toBe('');
    expect(getCustomMetadataKeyFromColumnId('custom_metadata')).toBe('custom_metadata');
  });

  it('handles edge cases', () => {
    expect(getCustomMetadataKeyFromColumnId('')).toBe('');
    expect(getCustomMetadataKeyFromColumnId('invalid:format')).toBe('format');
  });
});

describe('createCustomMetadataColumnId', () => {
  it('creates column ID from metadata key', () => {
    expect(createCustomMetadataColumnId('user_id')).toBe('custom_metadata:user_id');
    expect(createCustomMetadataColumnId('environment')).toBe('custom_metadata:environment');
    expect(createCustomMetadataColumnId('deployment_version')).toBe('custom_metadata:deployment_version');
  });

  it('handles metadata keys with special characters', () => {
    expect(createCustomMetadataColumnId('user-id')).toBe('custom_metadata:user-id');
    expect(createCustomMetadataColumnId('user_id_123')).toBe('custom_metadata:user_id_123');
    expect(createCustomMetadataColumnId('user.id')).toBe('custom_metadata:user.id');
  });

  it('handles empty metadata key', () => {
    expect(createCustomMetadataColumnId('')).toBe('custom_metadata:');
  });
});

describe('Custom Metadata Column ID Round Trip', () => {
  it('should be able to create and extract metadata keys correctly', () => {
    const testKeys = ['user_id', 'environment', 'deployment_version', 'custom_field_123'];

    testKeys.forEach((key) => {
      const columnId = createCustomMetadataColumnId(key);
      const extractedKey = getCustomMetadataKeyFromColumnId(columnId);
      expect(extractedKey).toBe(key);
    });
  });

  it('should be able to create and extract tag keys correctly', () => {
    const testKeys = ['user_id', 'environment', 'deployment_version', 'custom_field_123'];

    testKeys.forEach((key) => {
      const columnId = createTagColumnId(key);
      const extractedKey = getTagKeyFromColumnId(columnId);
      expect(extractedKey).toBe(key);
    });
  });
});

describe('Custom Metadata Integration', () => {
  it('should handle trace metadata with custom fields', () => {
    const traceWithCustomMetadata: TraceInfoV3 = {
      trace_id: 'trace123',
      trace_location: {
        type: 'MLFLOW_EXPERIMENT',
        mlflow_experiment: { experiment_id: 'exp123' },
      },
      request_time: '2023-10-01T00:00:00Z',
      state: 'OK',
      client_request_id: 'client456',
      request: '{"messages": [{"content": "Hello"}]}',
      response: '{"data": "output"}',
      tags: { key1: 'value1' },
      trace_metadata: {
        'mlflow.source.name': 'mlflow_field', // Should be filtered out
        user_id: 'custom_user_123',
        environment: 'production',
        deployment_version: 'v1.2.3',
        custom_field: 'custom_value',
      },
      assessments: [],
    };

    // Test that custom metadata fields are accessible
    expect(traceWithCustomMetadata.trace_metadata?.['user_id']).toBe('custom_user_123');
    expect(traceWithCustomMetadata.trace_metadata?.['environment']).toBe('production');
    expect(traceWithCustomMetadata.trace_metadata?.['deployment_version']).toBe('v1.2.3');
    expect(traceWithCustomMetadata.trace_metadata?.['custom_field']).toBe('custom_value');

    // Test that mlflow fields are present but should be filtered
    expect(traceWithCustomMetadata.trace_metadata?.['mlflow.source.name']).toBe('mlflow_field');
  });

  it('should handle trace metadata with no custom fields', () => {
    const traceWithOnlyMlflowMetadata: TraceInfoV3 = {
      trace_id: 'trace123',
      trace_location: {
        type: 'MLFLOW_EXPERIMENT',
        mlflow_experiment: { experiment_id: 'exp123' },
      },
      request_time: '2023-10-01T00:00:00Z',
      state: 'OK',
      client_request_id: 'client456',
      request: '{"messages": [{"content": "Hello"}]}',
      response: '{"data": "output"}',
      tags: { key1: 'value1' },
      trace_metadata: {
        'mlflow.source.name': 'mlflow_field',
        'mlflow.traceInputs': '{"input": "data"}',
        'mlflow.traceOutputs': '{"output": "data"}',
      },
      assessments: [],
    };

    // Test that only mlflow fields are present
    expect(traceWithOnlyMlflowMetadata.trace_metadata?.['mlflow.source.name']).toBe('mlflow_field');
    expect(traceWithOnlyMlflowMetadata.trace_metadata?.['mlflow.traceInputs']).toBe('{"input": "data"}');
    expect(traceWithOnlyMlflowMetadata.trace_metadata?.['mlflow.traceOutputs']).toBe('{"output": "data"}');
  });

  it('should handle trace with no metadata', () => {
    const traceWithNoMetadata: TraceInfoV3 = {
      trace_id: 'trace123',
      trace_location: {
        type: 'MLFLOW_EXPERIMENT',
        mlflow_experiment: { experiment_id: 'exp123' },
      },
      request_time: '2023-10-01T00:00:00Z',
      state: 'OK',
      client_request_id: 'client456',
      request: '{"messages": [{"content": "Hello"}]}',
      response: '{"data": "output"}',
      tags: { key1: 'value1' },
      // trace_metadata is undefined
      assessments: [],
    };

    expect(traceWithNoMetadata.trace_metadata).toBeUndefined();
  });
});
