import { describe, it, expect } from '@jest/globals';
import { isRootSpan, getRootSpan, extractInputs, extractOutputs, extractRetrievalContext } from './TraceUtils';
import type { ModelTrace, ModelTraceSpanV2, ModelTraceSpanV3 } from '@databricks/web-shared/model-trace-explorer';

describe('isRootSpan', () => {
  describe('Golden Path - Successful Operations', () => {
    it('should identify V3 root span with null parent_span_id', () => {
      const v3Span: ModelTraceSpanV3 = {
        trace_id: 'trace-123',
        span_id: 'span-1',
        trace_state: '',
        parent_span_id: null,
        name: 'root-span',
        start_time_unix_nano: '1000000000',
        end_time_unix_nano: '2000000000',
        status: {
          code: 'STATUS_CODE_OK',
        },
        attributes: {},
      };

      expect(isRootSpan(v3Span)).toBe(true);
    });

    it('should identify V2 root span with null parent_span_id and parent_id', () => {
      const v2Span: ModelTraceSpanV2 = {
        context: {
          span_id: 'span-1',
          trace_id: 'trace-123',
        },
        name: 'root-span',
        parent_span_id: null,
        parent_id: null,
        start_time: 1000,
        end_time: 2000,
      };

      expect(isRootSpan(v2Span)).toBe(true);
    });

    it('should identify V2 root span with undefined parent_span_id and parent_id', () => {
      const v2Span: ModelTraceSpanV2 = {
        context: {
          span_id: 'span-1',
          trace_id: 'trace-123',
        },
        name: 'root-span',
        start_time: 1000,
        end_time: 2000,
      };

      expect(isRootSpan(v2Span)).toBe(true);
    });
  });

  describe('Edge Cases', () => {
    it('should correctly identify non-root V3 span with parent_span_id', () => {
      const v3Span: ModelTraceSpanV3 = {
        trace_id: 'trace-123',
        span_id: 'span-2',
        trace_state: '',
        parent_span_id: 'span-1',
        name: 'child-span',
        start_time_unix_nano: '1000000000',
        end_time_unix_nano: '2000000000',
        status: {
          code: 'STATUS_CODE_OK',
        },
        attributes: {},
      };

      expect(isRootSpan(v3Span)).toBe(false);
    });

    it('should correctly identify non-root V2 span with parent_span_id', () => {
      const v2Span: ModelTraceSpanV2 = {
        context: {
          span_id: 'span-2',
          trace_id: 'trace-123',
        },
        name: 'child-span',
        parent_span_id: 'span-1',
        start_time: 1000,
        end_time: 2000,
      };

      expect(isRootSpan(v2Span)).toBe(false);
    });

    it('should correctly identify non-root V2 span with parent_id', () => {
      const v2Span: ModelTraceSpanV2 = {
        context: {
          span_id: 'span-2',
          trace_id: 'trace-123',
        },
        name: 'child-span',
        parent_id: 'span-1',
        start_time: 1000,
        end_time: 2000,
      };

      expect(isRootSpan(v2Span)).toBe(false);
    });

    it('should handle V3 span with empty string parent_span_id as root (falsy check)', () => {
      const v3Span: ModelTraceSpanV3 = {
        trace_id: 'trace-123',
        span_id: 'span-2',
        trace_state: '',
        parent_span_id: '',
        name: 'child-span',
        start_time_unix_nano: '1000000000',
        end_time_unix_nano: '2000000000',
        status: {
          code: 'STATUS_CODE_OK',
        },
        attributes: {},
      };

      // Empty string is falsy in JavaScript, so this is treated as a root span
      expect(isRootSpan(v3Span)).toBe(true);
    });
  });
});

describe('getRootSpan', () => {
  describe('Golden Path - Successful Operations', () => {
    it('should find root span in trace with V3 spans', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-2',
              trace_state: '',
              parent_span_id: 'span-1',
              name: 'child-span',
              start_time_unix_nano: '2000000000',
              end_time_unix_nano: '3000000000',
              status: {
                code: 'STATUS_CODE_OK',
              },
              attributes: {},
            } as ModelTraceSpanV3,
            {
              trace_id: 'trace-123',
              span_id: 'span-1',
              trace_state: '',
              parent_span_id: null,
              name: 'root-span',
              start_time_unix_nano: '1000000000',
              end_time_unix_nano: '4000000000',
              status: {
                code: 'STATUS_CODE_OK',
              },
              attributes: {},
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      const rootSpan = getRootSpan(trace);
      expect(rootSpan).not.toBeNull();
      expect(rootSpan?.name).toBe('root-span');
    });

    it('should find root span in trace with V2 spans', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              context: {
                span_id: 'span-2',
                trace_id: 'trace-123',
              },
              name: 'child-span',
              parent_span_id: 'span-1',
              start_time: 2000,
              end_time: 3000,
            } as ModelTraceSpanV2,
            {
              context: {
                span_id: 'span-1',
                trace_id: 'trace-123',
              },
              name: 'root-span',
              parent_span_id: null,
              parent_id: null,
              start_time: 1000,
              end_time: 4000,
            } as ModelTraceSpanV2,
          ],
        },
        info: {},
      };

      const rootSpan = getRootSpan(trace);
      expect(rootSpan).not.toBeNull();
      expect(rootSpan?.name).toBe('root-span');
    });
  });

  describe('Edge Cases', () => {
    it('should return null when trace has no spans', () => {
      const trace: ModelTrace = {
        data: {
          spans: [],
        },
        info: {},
      };

      expect(getRootSpan(trace)).toBeNull();
    });

    it('should return null when trace has only non-root spans', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-2',
              trace_state: '',
              parent_span_id: 'span-1',
              name: 'child-span-1',
              start_time_unix_nano: '2000000000',
              end_time_unix_nano: '3000000000',
              status: {
                code: 'STATUS_CODE_OK',
              },
              attributes: {},
            } as ModelTraceSpanV3,
            {
              trace_id: 'trace-123',
              span_id: 'span-3',
              trace_state: '',
              parent_span_id: 'span-1',
              name: 'child-span-2',
              start_time_unix_nano: '2500000000',
              end_time_unix_nano: '3500000000',
              status: {
                code: 'STATUS_CODE_OK',
              },
              attributes: {},
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      expect(getRootSpan(trace)).toBeNull();
    });

    it('should return null when data.spans is undefined', () => {
      const trace: ModelTrace = {
        data: {
          // Using 'as any' to test runtime behavior when spans is unexpectedly undefined
          spans: undefined as any,
        },
        info: {},
      };

      expect(getRootSpan(trace)).toBeNull();
    });
  });
});

describe('extractInputs', () => {
  describe('Golden Path - Successful Operations', () => {
    it('should extract string inputs from V2 span inputs field', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              context: {
                span_id: 'span-1',
                trace_id: 'trace-123',
              },
              name: 'root-span',
              parent_span_id: null,
              parent_id: null,
              start_time: 1000,
              end_time: 2000,
              inputs: 'test input string',
            } as ModelTraceSpanV2,
          ],
        },
        info: {},
      };

      expect(extractInputs(trace)).toBe('test input string');
    });

    it('should extract and stringify object inputs from V2 span inputs field', () => {
      const inputObject = { query: 'hello', params: { limit: 10 } };
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              context: {
                span_id: 'span-1',
                trace_id: 'trace-123',
              },
              name: 'root-span',
              parent_span_id: null,
              parent_id: null,
              start_time: 1000,
              end_time: 2000,
              inputs: inputObject,
            } as ModelTraceSpanV2,
          ],
        },
        info: {},
      };

      expect(extractInputs(trace)).toBe(JSON.stringify(inputObject));
    });

    it('should extract string inputs from V3 span attributes mlflow.spanInputs', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-1',
              trace_state: '',
              parent_span_id: null,
              name: 'root-span',
              start_time_unix_nano: '1000000000',
              end_time_unix_nano: '2000000000',
              status: {
                code: 'STATUS_CODE_OK',
              },
              attributes: {
                'mlflow.spanInputs': 'test input from attributes',
              },
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      expect(extractInputs(trace)).toBe('test input from attributes');
    });

    it('should extract and stringify object inputs from V3 span attributes mlflow.spanInputs', () => {
      const inputObject = { text: 'hello world', model: 'gpt-4' };
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-1',
              trace_state: '',
              parent_span_id: null,
              name: 'root-span',
              start_time_unix_nano: '1000000000',
              end_time_unix_nano: '2000000000',
              status: {
                code: 'STATUS_CODE_OK',
              },
              attributes: {
                'mlflow.spanInputs': inputObject,
              },
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      expect(extractInputs(trace)).toBe(JSON.stringify(inputObject));
    });

    it('should prefer direct inputs field over attributes in V2 span', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              context: {
                span_id: 'span-1',
                trace_id: 'trace-123',
              },
              name: 'root-span',
              parent_span_id: null,
              parent_id: null,
              start_time: 1000,
              end_time: 2000,
              inputs: 'direct inputs',
              attributes: {
                'mlflow.spanInputs': 'attribute inputs',
              },
            } as ModelTraceSpanV2,
          ],
        },
        info: {},
      };

      expect(extractInputs(trace)).toBe('direct inputs');
    });
  });

  describe('Edge Cases', () => {
    it('should return null when trace has no root span', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-2',
              trace_state: '',
              parent_span_id: 'span-1',
              name: 'child-span',
              start_time_unix_nano: '2000000000',
              end_time_unix_nano: '3000000000',
              status: {
                code: 'STATUS_CODE_OK',
              },
              attributes: {
                'mlflow.spanInputs': 'should not be found',
              },
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      expect(extractInputs(trace)).toBeNull();
    });

    it('should return null when root span has no inputs field or attributes', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              context: {
                span_id: 'span-1',
                trace_id: 'trace-123',
              },
              name: 'root-span',
              parent_span_id: null,
              parent_id: null,
              start_time: 1000,
              end_time: 2000,
            } as ModelTraceSpanV2,
          ],
        },
        info: {},
      };

      expect(extractInputs(trace)).toBeNull();
    });

    it('should return null when root span attributes does not contain mlflow.spanInputs', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-1',
              trace_state: '',
              parent_span_id: null,
              name: 'root-span',
              start_time_unix_nano: '1000000000',
              end_time_unix_nano: '2000000000',
              status: {
                code: 'STATUS_CODE_OK',
              },
              attributes: {
                'other.attribute': 'value',
              },
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      expect(extractInputs(trace)).toBeNull();
    });

    it('should handle empty trace data', () => {
      const trace: ModelTrace = {
        data: {
          spans: [],
        },
        info: {},
      };

      expect(extractInputs(trace)).toBeNull();
    });
  });
});

describe('extractOutputs', () => {
  describe('Golden Path - Successful Operations', () => {
    it('should extract string outputs from V2 span outputs field', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              context: {
                span_id: 'span-1',
                trace_id: 'trace-123',
              },
              name: 'root-span',
              parent_span_id: null,
              parent_id: null,
              start_time: 1000,
              end_time: 2000,
              outputs: 'test output string',
            } as ModelTraceSpanV2,
          ],
        },
        info: {},
      };

      expect(extractOutputs(trace)).toBe('test output string');
    });

    it('should extract and stringify object outputs from V2 span outputs field', () => {
      const outputObject = { result: 'success', data: { count: 5 } };
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              context: {
                span_id: 'span-1',
                trace_id: 'trace-123',
              },
              name: 'root-span',
              parent_span_id: null,
              parent_id: null,
              start_time: 1000,
              end_time: 2000,
              outputs: outputObject,
            } as ModelTraceSpanV2,
          ],
        },
        info: {},
      };

      expect(extractOutputs(trace)).toBe(JSON.stringify(outputObject));
    });

    it('should extract string outputs from V3 span attributes mlflow.spanOutputs', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-1',
              trace_state: '',
              parent_span_id: null,
              name: 'root-span',
              start_time_unix_nano: '1000000000',
              end_time_unix_nano: '2000000000',
              status: {
                code: 'STATUS_CODE_OK',
              },
              attributes: {
                'mlflow.spanOutputs': 'test output from attributes',
              },
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      expect(extractOutputs(trace)).toBe('test output from attributes');
    });

    it('should extract and stringify object outputs from V3 span attributes mlflow.spanOutputs', () => {
      const outputObject = { response: 'hello world', tokens: 50 };
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-1',
              trace_state: '',
              parent_span_id: null,
              name: 'root-span',
              start_time_unix_nano: '1000000000',
              end_time_unix_nano: '2000000000',
              status: {
                code: 'STATUS_CODE_OK',
              },
              attributes: {
                'mlflow.spanOutputs': outputObject,
              },
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      expect(extractOutputs(trace)).toBe(JSON.stringify(outputObject));
    });

    it('should prefer direct outputs field over attributes in V2 span', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              context: {
                span_id: 'span-1',
                trace_id: 'trace-123',
              },
              name: 'root-span',
              parent_span_id: null,
              parent_id: null,
              start_time: 1000,
              end_time: 2000,
              outputs: 'direct outputs',
              attributes: {
                'mlflow.spanOutputs': 'attribute outputs',
              },
            } as ModelTraceSpanV2,
          ],
        },
        info: {},
      };

      expect(extractOutputs(trace)).toBe('direct outputs');
    });
  });

  describe('Edge Cases', () => {
    it('should return null when trace has no root span', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-2',
              trace_state: '',
              parent_span_id: 'span-1',
              name: 'child-span',
              start_time_unix_nano: '2000000000',
              end_time_unix_nano: '3000000000',
              status: {
                code: 'STATUS_CODE_OK',
              },
              attributes: {
                'mlflow.spanOutputs': 'should not be found',
              },
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      expect(extractOutputs(trace)).toBeNull();
    });

    it('should return null when root span has no outputs field or attributes', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              context: {
                span_id: 'span-1',
                trace_id: 'trace-123',
              },
              name: 'root-span',
              parent_span_id: null,
              parent_id: null,
              start_time: 1000,
              end_time: 2000,
            } as ModelTraceSpanV2,
          ],
        },
        info: {},
      };

      expect(extractOutputs(trace)).toBeNull();
    });

    it('should return null when root span attributes does not contain mlflow.spanOutputs', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-1',
              trace_state: '',
              parent_span_id: null,
              name: 'root-span',
              start_time_unix_nano: '1000000000',
              end_time_unix_nano: '2000000000',
              status: {
                code: 'STATUS_CODE_OK',
              },
              attributes: {
                'other.attribute': 'value',
              },
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      expect(extractOutputs(trace)).toBeNull();
    });

    it('should handle empty trace data', () => {
      const trace: ModelTrace = {
        data: {
          spans: [],
        },
        info: {},
      };

      expect(extractOutputs(trace)).toBeNull();
    });
  });
});

describe('extractRetrievalContext', () => {
  describe('Golden Path - Successful Operations', () => {
    it('should extract retrieval context from V3 retrieval span', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-1',
              trace_state: '',
              parent_span_id: null,
              name: 'retrieval-span',
              start_time_unix_nano: '1000000000',
              end_time_unix_nano: '2000000000',
              status: {
                code: 'STATUS_CODE_OK',
              },
              attributes: {
                'mlflow.spanType': 'RETRIEVER',
                'mlflow.spanInputs': { query: 'What is machine learning?' },
                'mlflow.spanOutputs': [
                  { doc_uri: 'doc1.pdf', content: 'ML is a subset of AI' },
                  { doc_uri: 'doc2.pdf', content: 'ML involves algorithms' },
                ],
              },
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      const result = extractRetrievalContext(trace);
      expect(result).toEqual({
        retrieved_documents: [
          {
            documents: [
              { doc_uri: 'doc1.pdf', content: 'ML is a subset of AI' },
              { doc_uri: 'doc2.pdf', content: 'ML involves algorithms' },
            ],
            span_id: 'span-1',
            span_name: 'retrieval-span',
          },
        ],
      });
    });

    it('should extract retrieval context from V2 retrieval span', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              context: {
                span_id: 'span-1',
                trace_id: 'trace-123',
              },
              name: 'retrieval-span',
              parent_span_id: null,
              parent_id: null,
              start_time: 1000,
              end_time: 2000,
              span_type: 'RETRIEVER',
              inputs: { query: 'What is deep learning?' },
              outputs: [
                { doc_uri: 'doc3.pdf', content: 'Deep learning uses neural networks' },
                { doc_uri: 'doc4.pdf', content: 'Deep learning is part of ML' },
              ],
            } as ModelTraceSpanV2,
          ],
        },
        info: {},
      };

      const result = extractRetrievalContext(trace);
      expect(result).toEqual({
        retrieved_documents: [
          {
            documents: [
              { doc_uri: 'doc3.pdf', content: 'Deep learning uses neural networks' },
              { doc_uri: 'doc4.pdf', content: 'Deep learning is part of ML' },
            ],
            span_id: 'span-1',
            span_name: 'retrieval-span',
          },
        ],
      });
    });

    it('should aggregate documents from all top-level retrieval spans', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-1',
              trace_state: '',
              parent_span_id: null,
              name: 'first-retrieval',
              start_time_unix_nano: '1000000000',
              end_time_unix_nano: '2000000000',
              status: { code: 'STATUS_CODE_OK' },
              attributes: {
                'mlflow.spanType': 'RETRIEVER',
                'mlflow.spanInputs': { query: 'First query' },
                'mlflow.spanOutputs': [{ doc_uri: 'doc1.pdf', content: 'First result' }],
              },
            } as ModelTraceSpanV3,
            {
              trace_id: 'trace-123',
              span_id: 'span-2',
              trace_state: '',
              parent_span_id: null,
              name: 'second-retrieval',
              start_time_unix_nano: '3000000000',
              end_time_unix_nano: '4000000000',
              status: { code: 'STATUS_CODE_OK' },
              attributes: {
                'mlflow.spanType': 'RETRIEVER',
                'mlflow.spanInputs': { query: 'Second query' },
                'mlflow.spanOutputs': [{ doc_uri: 'doc2.pdf', content: 'Second result' }],
              },
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      const result = extractRetrievalContext(trace);
      expect(result).toEqual({
        retrieved_documents: [
          {
            documents: [{ doc_uri: 'doc1.pdf', content: 'First result' }],
            span_id: 'span-1',
            span_name: 'first-retrieval',
          },
          {
            documents: [{ doc_uri: 'doc2.pdf', content: 'Second result' }],
            span_id: 'span-2',
            span_name: 'second-retrieval',
          },
        ],
      });
    });

    it('should handle alternative field names for doc_uri and content', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-1',
              trace_state: '',
              parent_span_id: null,
              name: 'retrieval-span',
              start_time_unix_nano: '1000000000',
              end_time_unix_nano: '2000000000',
              status: { code: 'STATUS_CODE_OK' },
              attributes: {
                'mlflow.spanType': 'RETRIEVER',
                'mlflow.spanInputs': { query: 'Test query' },
                'mlflow.spanOutputs': [
                  { uri: 'doc1.pdf', page_content: 'Content 1' },
                  { uri: 'doc2.pdf', page_content: 'Content 2' },
                ],
              },
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      const result = extractRetrievalContext(trace);
      expect(result).toEqual({
        retrieved_documents: [
          {
            documents: [
              { doc_uri: 'doc1.pdf', content: 'Content 1' },
              { doc_uri: 'doc2.pdf', content: 'Content 2' },
            ],
            span_id: 'span-1',
            span_name: 'retrieval-span',
          },
        ],
      });
    });

    it('should extract retrieval context even with string inputs', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-1',
              trace_state: '',
              parent_span_id: null,
              name: 'retrieval-span',
              start_time_unix_nano: '1000000000',
              end_time_unix_nano: '2000000000',
              status: { code: 'STATUS_CODE_OK' },
              attributes: {
                'mlflow.spanType': 'RETRIEVER',
                'mlflow.spanInputs': 'String query',
                'mlflow.spanOutputs': [{ doc_uri: 'doc1.pdf', content: 'Content' }],
              },
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      const result = extractRetrievalContext(trace);
      expect(result).toEqual({
        retrieved_documents: [
          {
            documents: [{ doc_uri: 'doc1.pdf', content: 'Content' }],
            span_id: 'span-1',
            span_name: 'retrieval-span',
          },
        ],
      });
    });
  });

  describe('Edge Cases', () => {
    it('should return null when no retrieval spans exist', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-1',
              trace_state: '',
              parent_span_id: null,
              name: 'regular-span',
              start_time_unix_nano: '1000000000',
              end_time_unix_nano: '2000000000',
              status: { code: 'STATUS_CODE_OK' },
              attributes: {},
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      expect(extractRetrievalContext(trace)).toBeNull();
    });

    it('should return null when trace has empty spans', () => {
      const trace: ModelTrace = {
        data: {
          spans: [],
        },
        info: {},
      };

      expect(extractRetrievalContext(trace)).toBeNull();
    });

    it('should return null when retrieval span has no retrieved contexts', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-1',
              trace_state: '',
              parent_span_id: null,
              name: 'retrieval-span',
              start_time_unix_nano: '1000000000',
              end_time_unix_nano: '2000000000',
              status: { code: 'STATUS_CODE_OK' },
              attributes: {
                'mlflow.spanType': 'RETRIEVER',
              },
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      expect(extractRetrievalContext(trace)).toBeNull();
    });

    it('should return null when retrieval span has empty retrieved contexts array', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-1',
              trace_state: '',
              parent_span_id: null,
              name: 'retrieval-span',
              start_time_unix_nano: '1000000000',
              end_time_unix_nano: '2000000000',
              status: { code: 'STATUS_CODE_OK' },
              attributes: {
                'mlflow.spanType': 'RETRIEVER',
                'mlflow.spanOutputs': [],
              },
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      expect(extractRetrievalContext(trace)).toBeNull();
    });

    it('should skip nested retrieval spans and only use top-level', () => {
      const trace: ModelTrace = {
        data: {
          spans: [
            {
              trace_id: 'trace-123',
              span_id: 'span-1',
              trace_state: '',
              parent_span_id: null,
              name: 'top-level-retrieval',
              start_time_unix_nano: '1000000000',
              end_time_unix_nano: '5000000000',
              status: { code: 'STATUS_CODE_OK' },
              attributes: {
                'mlflow.spanType': 'RETRIEVER',
                'mlflow.spanInputs': { query: 'Top level query' },
                'mlflow.spanOutputs': [{ doc_uri: 'top.pdf', content: 'Top content' }],
              },
            } as ModelTraceSpanV3,
            {
              trace_id: 'trace-123',
              span_id: 'span-2',
              trace_state: '',
              parent_span_id: 'span-1',
              name: 'nested-retrieval',
              start_time_unix_nano: '2000000000',
              end_time_unix_nano: '3000000000',
              status: { code: 'STATUS_CODE_OK' },
              attributes: {
                'mlflow.spanType': 'RETRIEVER',
                'mlflow.spanInputs': { query: 'Nested query' },
                'mlflow.spanOutputs': [{ doc_uri: 'nested.pdf', content: 'Nested content' }],
              },
            } as ModelTraceSpanV3,
          ],
        },
        info: {},
      };

      const result = extractRetrievalContext(trace);
      expect(result).toEqual({
        retrieved_documents: [
          {
            documents: [{ doc_uri: 'top.pdf', content: 'Top content' }],
            span_id: 'span-1',
            span_name: 'top-level-retrieval',
          },
        ],
      });
    });
  });
});
