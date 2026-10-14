import { describe, it, expect } from '@jest/globals';

import type { ExperimentRunsSelectorResult } from '../../components/experiment-page/utils/experimentRuns.selector';
import {
  MLFLOW_RUN_TYPE_VALUE_EVALUATION,
  MLFLOW_RUN_TYPE_VALUE_GENAI_EVALUATE,
  MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION,
} from '../../constants';
import type { EvalRunsFilterClause } from './evalRunsFilterFields';
import {
  EvalRunsFilterOperator,
  buildClausesFilterString,
  getEvalRunsFilterFieldForColumn,
  getEvalRunsFilterFieldValues,
  getEvalRunsFilterFields,
  getOperatorsForFieldType,
  quoteFilterValue,
} from './evalRunsFilterFields';
import { EvalRunsTableColumnId } from './ExperimentEvaluationRunsTable.constants';
import { createEvalRunsTableKeyedColumnKey } from './ExperimentEvaluationRunsTable.utils';
import { EvalRunsTableKeyedColumnPrefix } from './ExperimentEvaluationRunsTable.constants';

// A complete, empty ExperimentRunsSelectorResult; tests override only the three lists the field
// builder reads (metricKeyList / paramKeyList / tagsList).
const makeRunsData = (overrides: Partial<ExperimentRunsSelectorResult> = {}): ExperimentRunsSelectorResult => ({
  runInfos: [],
  runUuidsMatchingFilter: [],
  metricKeyList: [],
  paramKeyList: [],
  metricsList: [],
  paramsList: [],
  tagsList: [],
  modelVersionsByRunUuid: {},
  experimentTags: {},
  datasetsList: [],
  ...overrides,
});

describe('buildClausesFilterString', () => {
  // @ui-test-skill-generated
  it('compiles a string field with = to a single-quoted expression', () => {
    const clauses: EvalRunsFilterClause[] = [
      { field: 'attributes.run_name', operator: EvalRunsFilterOperator.EQUALS, value: 'foo' },
    ];
    expect(buildClausesFilterString(clauses)).toBe("attributes.run_name = 'foo'");
  });

  it('compiles the contains operator to a LIKE expression wrapping the value in wildcards', () => {
    const clauses: EvalRunsFilterClause[] = [
      { field: 'attributes.run_name', operator: EvalRunsFilterOperator.CONTAINS, value: 'foo' },
    ];
    expect(buildClausesFilterString(clauses)).toBe("attributes.run_name LIKE '%foo%'");
  });

  it.each([
    { field: 'metrics.`accuracy`', expected: 'metrics.`accuracy` > 0.5', value: '0.5' },
    { field: 'attributes.start_time', expected: 'attributes.start_time > 123', value: '123' },
  ])('emits the value unquoted for the numeric field $field', ({ field, expected, value }) => {
    const clauses: EvalRunsFilterClause[] = [{ field, operator: EvalRunsFilterOperator.GREATER_THAN, value }];
    expect(buildClausesFilterString(clauses)).toBe(expected);
  });

  it('drops a numeric-field clause whose value is not a finite number', () => {
    const clauses: EvalRunsFilterClause[] = [
      { field: 'metrics.`accuracy`', operator: EvalRunsFilterOperator.GREATER_THAN, value: 'abc' },
    ];
    expect(buildClausesFilterString(clauses)).toBe('');
  });

  it.each([
    { name: 'empty field', clause: { field: '', operator: EvalRunsFilterOperator.EQUALS, value: 'foo' } },
    {
      name: 'whitespace-only value',
      clause: { field: 'attributes.run_name', operator: EvalRunsFilterOperator.EQUALS, value: '   ' },
    },
  ])('drops an incomplete clause with $name', ({ clause }) => {
    expect(buildClausesFilterString([clause])).toBe('');
  });

  it('joins multiple complete clauses with AND', () => {
    const clauses: EvalRunsFilterClause[] = [
      { field: 'attributes.run_name', operator: EvalRunsFilterOperator.EQUALS, value: 'foo' },
      { field: 'attributes.status', operator: EvalRunsFilterOperator.EQUALS, value: 'FINISHED' },
    ];
    expect(buildClausesFilterString(clauses)).toBe("attributes.run_name = 'foo' AND attributes.status = 'FINISHED'");
  });
});

describe('quoteFilterValue', () => {
  // @ui-test-skill-generated
  it.each([
    { input: 'foo', expected: "'foo'" },
    { input: 'say "hi"', expected: '\'say "hi"\'' },
    // The search parser has no escape inside a literal, so an apostrophe needs double quotes.
    { input: "it's", expected: '"it\'s"' },
    { input: `it's "both"`, expected: `'it''s "both"'` },
  ])('quotes $input as $expected', ({ input, expected }) => {
    expect(quoteFilterValue(input)).toBe(expected);
  });
});

describe('getOperatorsForFieldType', () => {
  // @ui-test-skill-generated
  it.each([
    {
      type: 'number' as const,
      expected: [
        EvalRunsFilterOperator.EQUALS,
        EvalRunsFilterOperator.NOT_EQUALS,
        EvalRunsFilterOperator.GREATER_THAN,
        EvalRunsFilterOperator.LESS_THAN,
        EvalRunsFilterOperator.GREATER_THAN_OR_EQUALS,
        EvalRunsFilterOperator.LESS_THAN_OR_EQUALS,
      ],
    },
    { type: 'enum' as const, expected: [EvalRunsFilterOperator.EQUALS, EvalRunsFilterOperator.NOT_EQUALS] },
    {
      type: 'string' as const,
      expected: [EvalRunsFilterOperator.EQUALS, EvalRunsFilterOperator.NOT_EQUALS, EvalRunsFilterOperator.CONTAINS],
    },
  ])('returns the $type operator set', ({ type, expected }) => {
    expect(getOperatorsForFieldType(type)).toEqual(expected);
  });
});

describe('getEvalRunsFilterFields', () => {
  // @ui-test-skill-generated
  it('returns the static run attributes plus metric/param/tag fields discovered from the runs', () => {
    const runsData = makeRunsData({
      metricKeyList: ['accuracy'],
      paramKeyList: ['learning_rate'],
      tagsList: [{ team: { key: 'team', value: 'ml' } }],
    });

    const fields = getEvalRunsFilterFields(runsData);
    const byId = (id: string) => fields.find((field) => field.id === id);

    // Static run attributes, including run type surfaced as an enum.
    expect(byId('attributes.run_name')).toMatchObject({ label: 'Run name', type: 'string', group: 'Run' });
    expect(byId('attributes.status')?.type).toBe('enum');
    expect(byId('attributes.status')?.options?.map((option) => option.value)).toEqual([
      'RUNNING',
      'SCHEDULED',
      'FINISHED',
      'FAILED',
      'KILLED',
    ]);
    expect(byId('tags.`mlflow.runType`')).toMatchObject({ label: 'Run type', type: 'enum', group: 'Run' });
    expect(byId('tags.`mlflow.runType`')?.options?.map((option) => option.value)).toEqual([
      MLFLOW_RUN_TYPE_VALUE_EVALUATION,
      MLFLOW_RUN_TYPE_VALUE_GENAI_EVALUATE,
      MLFLOW_RUN_TYPE_VALUE_ISSUE_DETECTION,
    ]);
    expect(byId('attributes.user_id')?.type).toBe('string');
    expect(byId('attributes.start_time')?.type).toBe('number');
    expect(byId('dataset.name')?.type).toBe('string');

    // Discovered keys, backtick-quoted in the field id and grouped by kind.
    expect(byId('metrics.`accuracy`')).toMatchObject({ label: 'accuracy', type: 'number', group: 'Metrics' });
    expect(byId('params.`learning_rate`')).toMatchObject({
      label: 'learning_rate',
      type: 'string',
      group: 'Parameters',
    });
    expect(byId('tags.`team`')).toMatchObject({ label: 'team', type: 'string', group: 'Tags' });
  });

  it('returns only the static attribute fields when the runs expose no metric/param/tag keys', () => {
    const fields = getEvalRunsFilterFields(makeRunsData());

    expect(fields.map((field) => field.id)).toEqual([
      'attributes.run_name',
      'attributes.status',
      'tags.`mlflow.runType`',
      'attributes.user_id',
      'attributes.start_time',
      'dataset.name',
    ]);
  });

  it('excludes internal mlflow.* tag keys from the Tags group while keeping user-facing tags', () => {
    const runsData = makeRunsData({
      tagsList: [
        {
          team: { key: 'team', value: 'ml' },
          'mlflow.source.name': { key: 'mlflow.source.name', value: 'notebook' },
        },
      ],
    });

    const fields = getEvalRunsFilterFields(runsData);

    expect(fields.find((field) => field.id === 'tags.`team`' && field.group === 'Tags')).toBeDefined();
    expect(fields.find((field) => field.id === 'tags.`mlflow.source.name`')).toBeUndefined();
  });
});

describe('getEvalRunsFilterFieldValues', () => {
  it('returns distinct values from the loaded runs for string fields', () => {
    const runsData = makeRunsData({
      runInfos: [
        {
          artifactUri: '',
          endTime: 0,
          experimentId: 'exp-1',
          lifecycleStage: 'active',
          runUuid: 'run-1',
          runName: 'validation',
          startTime: 0,
          status: 'FINISHED',
        },
        {
          artifactUri: '',
          endTime: 0,
          experimentId: 'exp-1',
          lifecycleStage: 'active',
          runUuid: 'run-2',
          runName: 'training',
          startTime: 0,
          status: 'RUNNING',
        },
        {
          artifactUri: '',
          endTime: 0,
          experimentId: 'exp-1',
          lifecycleStage: 'active',
          runUuid: 'run-3',
          runName: 'validation',
          startTime: 0,
          status: 'FINISHED',
        },
      ],
      metricKeyList: ['accuracy'],
      metricsList: [
        [{ key: 'accuracy', step: 0, timestamp: 0, value: 0.9 }],
        [{ key: 'accuracy', step: 0, timestamp: 0, value: 0.8 }],
        [{ key: 'accuracy', step: 0, timestamp: 0, value: 0.9 }],
      ],
      tagsList: [
        { team: { key: 'team', value: 'ml' } },
        { team: { key: 'team', value: 'platform' } },
        { team: { key: 'team', value: 'ml' } },
      ],
      paramsList: [
        [{ key: 'model', value: 'gpt-4o' }],
        [{ key: 'model', value: 'gpt-4o-mini' }],
        [{ key: 'model', value: 'gpt-4o' }],
      ],
    });

    expect(getEvalRunsFilterFieldValues('attributes.run_name', runsData)).toEqual(['training', 'validation']);
    expect(getEvalRunsFilterFieldValues('metrics.`accuracy`', runsData)).toEqual(['0.8', '0.9']);
    expect(getEvalRunsFilterFieldValues('params.`model`', runsData)).toEqual(['gpt-4o', 'gpt-4o-mini']);
    expect(getEvalRunsFilterFieldValues('tags.`team`', runsData)).toEqual(['ml', 'platform']);
  });

  it('returns no values for fields with no loaded observations', () => {
    expect(getEvalRunsFilterFieldValues('params.`missing`', makeRunsData())).toEqual([]);
  });
});

describe('getEvalRunsFilterFieldForColumn', () => {
  it.each([
    { columnId: EvalRunsTableColumnId.runName, field: 'attributes.run_name' },
    { columnId: EvalRunsTableColumnId.status, field: 'attributes.status' },
    { columnId: EvalRunsTableColumnId.createdAt, field: 'attributes.start_time' },
    { columnId: EvalRunsTableColumnId.dataset, field: 'dataset.name' },
    {
      columnId: createEvalRunsTableKeyedColumnKey(EvalRunsTableKeyedColumnPrefix.METRIC, 'accuracy'),
      field: 'metrics.`accuracy`',
    },
    {
      columnId: createEvalRunsTableKeyedColumnKey(EvalRunsTableKeyedColumnPrefix.PARAM, 'model.name'),
      field: 'params.`model.name`',
    },
    {
      columnId: createEvalRunsTableKeyedColumnKey(EvalRunsTableKeyedColumnPrefix.TAG, 'team'),
      field: 'tags.`team`',
    },
  ])('maps $columnId to $field', ({ columnId, field }) => {
    expect(getEvalRunsFilterFieldForColumn(columnId)).toBe(field);
  });

  it.each([
    EvalRunsTableColumnId.checkbox,
    EvalRunsTableColumnId.visibility,
    EvalRunsTableColumnId.datasetVersion,
    EvalRunsTableColumnId.modelVersion,
    'attribute.custom',
    'tag.mlflow.source.name',
    'unknown',
  ])('does not expose unsupported column %s as a structured filter', (columnId) => {
    expect(getEvalRunsFilterFieldForColumn(columnId)).toBeUndefined();
  });
});
