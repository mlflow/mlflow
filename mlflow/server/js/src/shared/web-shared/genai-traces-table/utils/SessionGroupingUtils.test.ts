import { describe, it, expect } from '@jest/globals';

import { SESSION_ID_METADATA_KEY, type ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

import type { EvalTraceComparisonEntry } from '../types';
import { groupTracesBySessionForTable } from './SessionGroupingUtils';

const createMockTraceInfo = (traceId: string, sessionId: string | null, requestTime: string): ModelTraceInfoV3 => ({
  trace_id: traceId,
  trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'exp-1' } },
  request_time: requestTime,
  state: 'OK',
  tags: {},
  trace_metadata: sessionId ? { [SESSION_ID_METADATA_KEY]: sessionId } : undefined,
});

const createMockEntry = (traceId: string, sessionId: string | null, requestTime: string): EvalTraceComparisonEntry => ({
  currentRunValue: {
    evaluationId: `eval-${traceId}`,
    requestId: `req-${traceId}`,
    inputs: {},
    inputsId: `inputs-${traceId}`,
    outputs: {},
    targets: {},
    overallAssessments: [],
    responseAssessmentsByName: {},
    metrics: {},
    traceInfo: createMockTraceInfo(traceId, sessionId, requestTime),
  },
});

describe('groupTracesBySessionForTable', () => {
  it('groups traces by session and adds session headers', () => {
    const entries: EvalTraceComparisonEntry[] = [
      createMockEntry('trace-1', 'session-A', '2025-01-01T10:00:00.000Z'),
      createMockEntry('trace-2', 'session-A', '2025-01-01T11:00:00.000Z'),
      createMockEntry('trace-3', 'session-B', '2025-01-01T09:00:00.000Z'),
    ];

    const result = groupTracesBySessionForTable(entries);

    // Should have 2 session headers + 3 trace rows = 5 items
    expect(result).toHaveLength(5);

    // First session header
    expect(result[0]).toMatchObject({ type: 'sessionHeader', sessionId: 'session-A' });
    // Then trace rows for session A
    expect(result[1]).toMatchObject({ type: 'trace' });
    expect(result[2]).toMatchObject({ type: 'trace' });

    // Second session header
    expect(result[3]).toMatchObject({ type: 'sessionHeader', sessionId: 'session-B' });
    // Then trace row for session B
    expect(result[4]).toMatchObject({ type: 'trace' });
  });

  it('sorts traces within a session by request time ascending', () => {
    const entries: EvalTraceComparisonEntry[] = [
      createMockEntry('trace-late', 'session-A', '2025-01-01T15:00:00.000Z'),
      createMockEntry('trace-early', 'session-A', '2025-01-01T09:00:00.000Z'),
      createMockEntry('trace-middle', 'session-A', '2025-01-01T12:00:00.000Z'),
    ];

    const result = groupTracesBySessionForTable(entries);

    // Should have 1 session header + 3 trace rows = 4 items
    expect(result).toHaveLength(4);
    expect(result[0]).toMatchObject({ type: 'sessionHeader', sessionId: 'session-A' });

    // Verify traces are sorted by request time ascending
    const traceRows = result.filter(
      (row): row is { type: 'trace'; data: EvalTraceComparisonEntry } => row.type === 'trace',
    );
    expect(traceRows[0].data.currentRunValue?.traceInfo?.trace_id).toBe('trace-early');
    expect(traceRows[1].data.currentRunValue?.traceInfo?.trace_id).toBe('trace-middle');
    expect(traceRows[2].data.currentRunValue?.traceInfo?.trace_id).toBe('trace-late');
  });

  it('places standalone traces (without session ID) at the end', () => {
    const entries: EvalTraceComparisonEntry[] = [
      createMockEntry('trace-standalone-1', null, '2025-01-01T08:00:00.000Z'),
      createMockEntry('trace-session', 'session-A', '2025-01-01T10:00:00.000Z'),
      createMockEntry('trace-standalone-2', null, '2025-01-01T12:00:00.000Z'),
    ];

    const result = groupTracesBySessionForTable(entries);

    // Should have 1 session header + 1 session trace + 2 standalone traces = 4 items
    expect(result).toHaveLength(4);

    // Session content first
    expect(result[0]).toMatchObject({ type: 'sessionHeader', sessionId: 'session-A' });
    expect(result[1]).toMatchObject({ type: 'trace' });
    if (result[1].type === 'trace') {
      expect(result[1].data.currentRunValue?.traceInfo?.trace_id).toBe('trace-session');
    }

    // Standalone traces at the end
    expect(result[2]).toMatchObject({ type: 'trace' });
    if (result[2].type === 'trace') {
      expect(result[2].data.currentRunValue?.traceInfo?.trace_id).toBe('trace-standalone-1');
    }
    expect(result[3]).toMatchObject({ type: 'trace' });
    if (result[3].type === 'trace') {
      expect(result[3].data.currentRunValue?.traceInfo?.trace_id).toBe('trace-standalone-2');
    }
  });

  it('returns empty array for empty input', () => {
    const result = groupTracesBySessionForTable([]);
    expect(result).toEqual([]);
  });
});
