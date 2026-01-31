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
  it('groups traces by session and adds session headers when expanded', () => {
    const entries: EvalTraceComparisonEntry[] = [
      createMockEntry('trace-1', 'session-A', '2025-01-01T10:00:00.000Z'),
      createMockEntry('trace-2', 'session-A', '2025-01-01T11:00:00.000Z'),
      createMockEntry('trace-3', 'session-B', '2025-01-01T09:00:00.000Z'),
    ];

    const expandedSessions = new Set(['session-A', 'session-B']);
    const { groupedRows } = groupTracesBySessionForTable(entries, expandedSessions);

    // Should have 2 session headers + 3 trace rows = 5 items
    expect(groupedRows).toHaveLength(5);

    // First session header
    expect(groupedRows[0]).toMatchObject({ type: 'sessionHeader', sessionId: 'session-A' });
    // Then trace rows for session A
    expect(groupedRows[1]).toMatchObject({ type: 'trace' });
    expect(groupedRows[2]).toMatchObject({ type: 'trace' });

    // Second session header
    expect(groupedRows[3]).toMatchObject({ type: 'sessionHeader', sessionId: 'session-B' });
    // Then trace row for session B
    expect(groupedRows[4]).toMatchObject({ type: 'trace' });
  });

  it('hides trace rows for collapsed sessions', () => {
    const entries: EvalTraceComparisonEntry[] = [
      createMockEntry('trace-1', 'session-A', '2025-01-01T10:00:00.000Z'),
      createMockEntry('trace-2', 'session-A', '2025-01-01T11:00:00.000Z'),
      createMockEntry('trace-3', 'session-B', '2025-01-01T09:00:00.000Z'),
    ];

    // Only session-A is expanded
    const expandedSessions = new Set(['session-A']);
    const { groupedRows } = groupTracesBySessionForTable(entries, expandedSessions);

    // Should have 2 session headers + 2 trace rows (only session-A) = 4 items
    expect(groupedRows).toHaveLength(4);

    expect(groupedRows[0]).toMatchObject({ type: 'sessionHeader', sessionId: 'session-A' });
    expect(groupedRows[1]).toMatchObject({ type: 'trace' });
    expect(groupedRows[2]).toMatchObject({ type: 'trace' });
    expect(groupedRows[3]).toMatchObject({ type: 'sessionHeader', sessionId: 'session-B' });
    // No trace rows for session-B since it's collapsed
  });

  it('shows only session headers when all sessions are collapsed', () => {
    const entries: EvalTraceComparisonEntry[] = [
      createMockEntry('trace-1', 'session-A', '2025-01-01T10:00:00.000Z'),
      createMockEntry('trace-2', 'session-B', '2025-01-01T09:00:00.000Z'),
    ];

    const expandedSessions = new Set<string>(); // No expanded sessions
    const { groupedRows } = groupTracesBySessionForTable(entries, expandedSessions);

    // Should have 2 session headers only
    expect(groupedRows).toHaveLength(2);
    expect(groupedRows[0]).toMatchObject({ type: 'sessionHeader', sessionId: 'session-A' });
    expect(groupedRows[1]).toMatchObject({ type: 'sessionHeader', sessionId: 'session-B' });
  });

  it('sorts traces within a session by request time ascending', () => {
    const entries: EvalTraceComparisonEntry[] = [
      createMockEntry('trace-late', 'session-A', '2025-01-01T15:00:00.000Z'),
      createMockEntry('trace-early', 'session-A', '2025-01-01T09:00:00.000Z'),
      createMockEntry('trace-middle', 'session-A', '2025-01-01T12:00:00.000Z'),
    ];

    const expandedSessions = new Set(['session-A']);
    const { groupedRows } = groupTracesBySessionForTable(entries, expandedSessions);

    // Should have 1 session header + 3 trace rows = 4 items
    expect(groupedRows).toHaveLength(4);
    expect(groupedRows[0]).toMatchObject({ type: 'sessionHeader', sessionId: 'session-A' });

    // Verify traces are sorted by request time ascending
    const traceRows = groupedRows.filter(
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

    const expandedSessions = new Set(['session-A']);
    const { groupedRows } = groupTracesBySessionForTable(entries, expandedSessions);

    // Should have 1 session header + 1 session trace + 2 standalone traces = 4 items
    expect(groupedRows).toHaveLength(4);

    // Session content first
    expect(groupedRows[0]).toMatchObject({ type: 'sessionHeader', sessionId: 'session-A' });
    expect(groupedRows[1]).toMatchObject({ type: 'trace' });
    if (groupedRows[1].type === 'trace') {
      expect(groupedRows[1].data.currentRunValue?.traceInfo?.trace_id).toBe('trace-session');
    }

    // Standalone traces at the end
    expect(groupedRows[2]).toMatchObject({ type: 'trace' });
    if (groupedRows[2].type === 'trace') {
      expect(groupedRows[2].data.currentRunValue?.traceInfo?.trace_id).toBe('trace-standalone-1');
    }
    expect(groupedRows[3]).toMatchObject({ type: 'trace' });
    if (groupedRows[3].type === 'trace') {
      expect(groupedRows[3].data.currentRunValue?.traceInfo?.trace_id).toBe('trace-standalone-2');
    }
  });

  it('returns empty result for empty input', () => {
    const { groupedRows, traceIdToTurnMap } = groupTracesBySessionForTable([], new Set());
    expect(groupedRows).toEqual([]);
    expect(traceIdToTurnMap).toEqual({});
  });

  it('returns traceIdToTurnMap with correct turn numbers for expanded sessions', () => {
    const entries: EvalTraceComparisonEntry[] = [
      createMockEntry('trace-late', 'session-A', '2025-01-01T15:00:00.000Z'),
      createMockEntry('trace-early', 'session-A', '2025-01-01T09:00:00.000Z'),
      createMockEntry('trace-middle', 'session-A', '2025-01-01T12:00:00.000Z'),
    ];

    const expandedSessions = new Set(['session-A']);
    const { traceIdToTurnMap } = groupTracesBySessionForTable(entries, expandedSessions);

    // Turn numbers should be 1-indexed and based on sorted order (by request time)
    expect(traceIdToTurnMap['trace-early']).toBe(1);
    expect(traceIdToTurnMap['trace-middle']).toBe(2);
    expect(traceIdToTurnMap['trace-late']).toBe(3);
  });
});
