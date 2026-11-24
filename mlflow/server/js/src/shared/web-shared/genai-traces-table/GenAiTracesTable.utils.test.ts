import { describe, it, expect } from '@jest/globals';

import { getSimulationColumnsToAdd, sortGroupedColumns } from './GenAiTracesTable.utils';
import {
  RESPONSE_COLUMN_ID,
  SIMULATION_GOAL_COLUMN_ID,
  SIMULATION_PERSONA_COLUMN_ID,
  createAssessmentColumnId,
  createExpectationColumnId,
} from './hooks/useTableColumns';
import { SIMULATION_GOAL_KEY, SIMULATION_PERSONA_KEY } from './utils/SessionGroupingUtils';
import type { TracesTableColumn } from './types';
import { TracesTableColumnType, TracesTableColumnGroup } from './types';
import type { ModelTraceInfoV3 } from '../model-trace-explorer/ModelTrace.types';

const createTrace = (metadata: Record<string, string> = {}): ModelTraceInfoV3 =>
  ({
    trace_id: 'test-id',
    trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'exp' } },
    trace_metadata: metadata,
  }) as ModelTraceInfoV3;

const goalColumn: TracesTableColumn = {
  id: SIMULATION_GOAL_COLUMN_ID,
  label: 'Goal',
  type: TracesTableColumnType.TRACE_INFO,
  group: TracesTableColumnGroup.INFO,
};
const personaColumn: TracesTableColumn = {
  id: SIMULATION_PERSONA_COLUMN_ID,
  label: 'Persona',
  type: TracesTableColumnType.TRACE_INFO,
  group: TracesTableColumnGroup.INFO,
};

const goalMetadata = { [SIMULATION_GOAL_KEY]: 'test-goal' };
const personaMetadata = { [SIMULATION_PERSONA_KEY]: 'test-persona' };
const bothMetadata = { ...goalMetadata, ...personaMetadata };

describe('getSimulationColumnsToAdd', () => {
  it('returns empty array when traces is empty or has no metadata', () => {
    expect(getSimulationColumnsToAdd([], [goalColumn, personaColumn], [])).toEqual([]);
    expect(getSimulationColumnsToAdd([createTrace()], [goalColumn, personaColumn], [])).toEqual([]);
  });

  it.each([
    ['goal only', [createTrace(goalMetadata)], [goalColumn]],
    ['persona only', [createTrace(personaMetadata)], [personaColumn]],
    ['both', [createTrace(bothMetadata)], [goalColumn, personaColumn]],
    ['partial traces', [createTrace(), createTrace(goalMetadata)], [goalColumn]],
  ])('returns columns based on metadata: %s', (_, traces, expected) => {
    expect(getSimulationColumnsToAdd(traces, [goalColumn, personaColumn], [])).toEqual(expected);
  });

  it.each([
    ['goal', [createTrace(goalMetadata)], [goalColumn], []],
    ['persona', [createTrace(personaMetadata)], [personaColumn], []],
    ['goal when both present', [createTrace(bothMetadata)], [goalColumn], [personaColumn]],
  ])('skips already selected: %s', (_, traces, selected, expected) => {
    expect(getSimulationColumnsToAdd(traces, [goalColumn, personaColumn], selected)).toEqual(expected);
  });

  it('returns only columns available in allColumns', () => {
    expect(getSimulationColumnsToAdd([createTrace(bothMetadata)], [goalColumn], [])).toEqual([goalColumn]);
    expect(getSimulationColumnsToAdd([createTrace(bothMetadata)], [], [])).toEqual([]);
  });
});

describe('sortGroupedColumns — base / expectations / assessments / other ordering', () => {
  const responseCol: TracesTableColumn = {
    id: RESPONSE_COLUMN_ID,
    label: 'Response',
    type: TracesTableColumnType.TRACE_INFO,
    group: TracesTableColumnGroup.BASE,
  };
  const expectedResponseCol: TracesTableColumn = {
    id: createExpectationColumnId('expected_response'),
    label: 'expected_response',
    type: TracesTableColumnType.EXPECTATION,
    group: TracesTableColumnGroup.EXPECTATION,
    expectationName: 'expected_response',
  };
  const expectedFactsCol: TracesTableColumn = {
    id: createExpectationColumnId('expected_facts'),
    label: 'expected_facts',
    type: TracesTableColumnType.EXPECTATION,
    group: TracesTableColumnGroup.EXPECTATION,
    expectationName: 'expected_facts',
  };
  const executionTimeCol: TracesTableColumn = {
    id: 'execution_duration',
    label: 'Execution time',
    type: TracesTableColumnType.TRACE_INFO,
    group: TracesTableColumnGroup.INFO,
  };
  const qualityCol: TracesTableColumn = {
    id: createAssessmentColumnId('quality'),
    label: 'quality',
    type: TracesTableColumnType.ASSESSMENT,
    group: TracesTableColumnGroup.ASSESSMENT,
  };

  it('orders columns as: BASE → EXPECTATION → ASSESSMENT → INFO', () => {
    // Scramble the input so order in the array can't accidentally pass the test.
    const sorted = sortGroupedColumns([executionTimeCol, qualityCol, expectedResponseCol, responseCol]);
    expect(sorted.map((c) => c.id)).toEqual([
      responseCol.id,
      expectedResponseCol.id,
      qualityCol.id,
      executionTimeCol.id,
    ]);
  });

  it('puts expected_response first within the EXPECTATION group', () => {
    // expected_facts < expected_response alphabetically, so without an explicit
    // priority the sort would put expected_facts first.
    const sorted = sortGroupedColumns([expectedFactsCol, expectedResponseCol]);
    expect(sorted.map((c) => c.id)).toEqual([expectedResponseCol.id, expectedFactsCol.id]);
  });
});
