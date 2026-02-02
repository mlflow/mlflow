import { describe, it, expect } from '@jest/globals';

import { getSimulationColumnsToAdd } from './GenAiTracesTable.utils';
import { SIMULATION_GOAL_COLUMN_ID, SIMULATION_PERSONA_COLUMN_ID } from './hooks/useTableColumns';
import { SIMULATION_GOAL_KEY, SIMULATION_PERSONA_KEY } from './utils/SessionGroupingUtils';
import type { TracesTableColumn } from './types';
import { TracesTableColumnType, TracesTableColumnGroup } from './types';
import type { ModelTraceInfoV3 } from '../model-trace-explorer';

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
