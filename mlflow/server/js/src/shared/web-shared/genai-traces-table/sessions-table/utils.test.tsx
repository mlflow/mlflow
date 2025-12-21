import { describe, it, expect } from '@jest/globals';

import { getSessionTableRows } from './utils';
import type { ModelTraceInfoV3 } from '../../model-trace-explorer';

const MOCK_TRACES: ModelTraceInfoV3[] = [
  {
    trace_id: 'tr-22',
    trace_location: {
      type: 'MLFLOW_EXPERIMENT',
      mlflow_experiment: {
        experiment_id: '0',
      },
    },
    request_time: '2025-10-31T04:34:14.127Z',
    execution_duration: '1.0s',
    state: 'OK',
    trace_metadata: {
      'mlflow.trace_schema.version': '3',
      'mlflow.trace.tokenUsage': '{"input_tokens": 100, "output_tokens": 100, "total_tokens": 200}',
      'mlflow.source.type': 'LOCAL',
      'mlflow.source.name': 'script.py',
      'mlflow.user': 'daniel.lok',
      'mlflow.trace.session': 'session-2',
    },
    tags: {},
    request_preview: 'Session 2 turn 2 request',
    response_preview: 'Session 2 turn 2 response',
  },
  {
    trace_id: 'tr-21',
    trace_location: {
      type: 'MLFLOW_EXPERIMENT',
      mlflow_experiment: {
        experiment_id: '0',
      },
    },
    request_time: '2025-10-31T04:34:04.119Z',
    execution_duration: '1.0s',
    state: 'OK',
    trace_metadata: {
      'mlflow.trace_schema.version': '3',
      'mlflow.trace.tokenUsage': '{"input_tokens": 100, "output_tokens": 100, "total_tokens": 200}',
      'mlflow.source.type': 'LOCAL',
      'mlflow.source.name': 'script.py',
      'mlflow.user': 'daniel.lok',
      'mlflow.trace.session': 'session-2',
    },
    tags: {},
    request_preview: 'Session 2 turn 1 request',
    response_preview: 'Session 2 turn 1 response',
  },
  {
    trace_id: 'tr-11',
    trace_location: {
      type: 'MLFLOW_EXPERIMENT',
      mlflow_experiment: {
        experiment_id: '0',
      },
    },
    request_time: '2025-10-31T04:33:41.004Z',
    execution_duration: '1.0s',
    state: 'OK',
    trace_metadata: {
      'mlflow.trace_schema.version': '3',
      'mlflow.trace.tokenUsage': '{"input_tokens": 100, "output_tokens": 100, "total_tokens": 200}',
      'mlflow.source.type': 'LOCAL',
      'mlflow.source.name': 'script.py',
      'mlflow.user': 'daniel.lok',
      'mlflow.trace.session': 'session-1',
    },
    tags: {},
    request_preview: 'Session 1 turn 1 request',
    response_preview: 'Session 1 turn 1 response',
  },
];

describe('getSessionTableRows', () => {
  it('should parse the sessions correctly', () => {
    const rows = getSessionTableRows('0', MOCK_TRACES);

    // two unique sessions
    expect(rows.length).toBe(2);

    // first session should be session 2 (sorted by request time descending)
    expect(rows[0].sessionId).toBe('session-2');
    expect(rows[0].requestPreview).toBe('Session 2 turn 1 request');
    expect(rows[0].firstTrace.trace_id).toBe('tr-21');
    expect(rows[0].experimentId).toBe('0');
    expect(rows[0].sessionStartTime).toBe('2025-10-31 04:34:04');
    expect(rows[0].sessionDuration).toBe('2.0s');
    expect(rows[0].tokens).toBe(400);
    expect(rows[0].turns).toBe(2);

    expect(rows[1].sessionId).toBe('session-1');
    expect(rows[1].requestPreview).toBe('Session 1 turn 1 request');
    expect(rows[1].firstTrace.trace_id).toBe('tr-11');
    expect(rows[1].experimentId).toBe('0');
    expect(rows[1].sessionStartTime).toBe('2025-10-31 04:33:41');
    expect(rows[1].sessionDuration).toBe('1.0s');
    expect(rows[1].tokens).toBe(200);
    expect(rows[1].turns).toBe(1);
  });
});
