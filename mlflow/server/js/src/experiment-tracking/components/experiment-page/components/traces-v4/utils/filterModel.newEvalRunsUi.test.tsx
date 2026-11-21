import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { FilterOp, type FilterClause } from '@databricks/web-shared/traces-table';
import { shouldEnableNewEvalRunsUi } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { makeFeedbackAssessment, makeTrace } from '../test-utils/mockTraces';
import { compileFilterModel, isSupportedFilterClause, useMlflowTraceFilterFields } from './filterModel';
import { assessmentValueKindKey } from './assessmentColumns';

jest.mock('@mlflow/mlflow/src/common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FeatureUtils')>(
    '@mlflow/mlflow/src/common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: jest.fn(() => true),
}));

const NUMERIC_KEYS: ReadonlySet<string> = new Set([
  assessmentValueKindKey('feedback', 'relevance'),
  assessmentValueKindKey('feedback', 'score'),
  assessmentValueKindKey('feedback', 'delta'),
  assessmentValueKindKey('expectation', 'expected_score'),
]);

const wrapper = ({ children }: { children: React.ReactNode }) => <IntlProvider locale="en">{children}</IntlProvider>;

const TRACES = [
  makeTrace('t1', {
    tags: { 'mlflow.traceName': 'chat', env: 'prod' },
    trace_metadata: { 'mlflow.trace.session': 'session-1', 'mlflow.sourceRun': 'run-1', owner: 'alice' },
    assessments: [makeFeedbackAssessment('relevance', 4), makeFeedbackAssessment('correctness', 'yes')],
  }),
  makeTrace('t2', {
    tags: { 'mlflow.traceName': 'agent', env: 'staging' },
    trace_metadata: { 'mlflow.trace.session': 'session-2' },
    assessments: [makeFeedbackAssessment('relevance', 2.5)],
  }),
];

describe('filters with the new evaluation runs UI', () => {
  beforeEach(() => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
  });

  describe('compileFilterModel', () => {
    test.each<{ name: string; clause: FilterClause; expected: string }>([
      {
        name: 'state != uses the chosen operator',
        clause: { field: 'state', operator: FilterOp.NOT_EQUALS, value: 'OK' },
        expected: "attributes.status != 'OK'",
      },
      {
        name: 'user !=',
        clause: { field: 'user', operator: FilterOp.NOT_EQUALS, value: 'alice' },
        expected: `request_metadata."mlflow.trace.user" != 'alice'`,
      },
      {
        name: 'session !=',
        clause: { field: 'session', operator: FilterOp.NOT_EQUALS, value: 's-1' },
        expected: "request_metadata.mlflow.trace.session != 's-1'",
      },
      {
        name: 'session contains',
        clause: { field: 'session', operator: FilterOp.CONTAINS, value: 's-1' },
        expected: "request_metadata.mlflow.trace.session ILIKE '%s-1%'",
      },
      {
        name: 'numeric feedback = compares as a number',
        clause: { field: 'assessment', key: 'relevance', operator: FilterOp.EQUALS, value: '4' },
        expected: 'feedback.`relevance` >= 4 AND feedback.`relevance` <= 4',
      },
      {
        name: 'numeric feedback = with a decimal spelling',
        clause: { field: 'assessment', key: 'relevance', operator: FilterOp.EQUALS, value: '4.0' },
        expected: 'feedback.`relevance` >= 4 AND feedback.`relevance` <= 4',
      },
      {
        name: 'whole-number feedback != excludes both spellings',
        clause: { field: 'assessment', key: 'relevance', operator: FilterOp.NOT_EQUALS, value: '4' },
        expected: "feedback.`relevance` != '4' AND feedback.`relevance` != '4.0'",
      },
      {
        name: 'fractional feedback !=',
        clause: { field: 'assessment', key: 'score', operator: FilterOp.NOT_EQUALS, value: '0.75' },
        expected: "feedback.`score` != '0.75'",
      },
      {
        name: 'negative numeric feedback =',
        clause: { field: 'assessment', key: 'delta', operator: FilterOp.EQUALS, value: '-1' },
        expected: 'feedback.`delta` >= -1 AND feedback.`delta` <= -1',
      },
      {
        name: 'non-numeric feedback stays a quoted comparison',
        clause: { field: 'assessment', key: 'correctness', operator: FilterOp.EQUALS, value: "it's yes" },
        expected: "feedback.`correctness` = 'it''s yes'",
      },
      {
        name: 'numeric expectation = compares as a number',
        clause: { field: 'expectation', key: 'expected_score', operator: FilterOp.EQUALS, value: '3' },
        expected: 'expectation.`expected_score` >= 3 AND expectation.`expected_score` <= 3',
      },
    ])('$name', ({ clause, expected }) => {
      expect(compileFilterModel([clause], NUMERIC_KEYS)).toEqual([expected]);
    });

    test('a numeric-looking value on a text assessment keeps the literal comparison', () => {
      expect(
        compileFilterModel(
          [
            { field: 'assessment', key: 'rating', operator: FilterOp.EQUALS, value: '5' },
            { field: 'assessment', key: 'rating', operator: FilterOp.NOT_EQUALS, value: '5' },
          ],
          NUMERIC_KEYS,
        ),
      ).toEqual(["feedback.`rating` = '5'", "feedback.`rating` != '5'"]);
    });

    test('an assessment of unknown type keeps the literal comparison', () => {
      expect(
        compileFilterModel([
          { field: 'assessment', key: 'relevance', operator: FilterOp.EQUALS, value: '4' },
          { field: 'expectation', key: 'expected_score', operator: FilterOp.EQUALS, value: '3' },
        ]),
      ).toEqual(["feedback.`relevance` = '4'", "expectation.`expected_score` = '3'"]);
    });

    test('numeric comparisons compile to unquoted literals whether or not the assessment type is known', () => {
      expect(
        compileFilterModel([
          { field: 'assessment', key: 'relevance', operator: FilterOp.LESS_THAN, value: '0.4' },
          { field: 'expectation', key: 'expected_score', operator: FilterOp.GREATER_THAN_OR_EQUALS, value: '3' },
        ]),
      ).toEqual(['feedback.`relevance` < 0.4', 'expectation.`expected_score` >= 3']);
    });

    test('feedback and expectations with the same name are typed separately', () => {
      expect(
        compileFilterModel(
          [{ field: 'expectation', key: 'relevance', operator: FilterOp.EQUALS, value: '4' }],
          new Set([assessmentValueKindKey('feedback', 'relevance')]),
        ),
      ).toEqual(["expectation.`relevance` = '4'"]);
    });
  });

  describe('useMlflowTraceFilterFields', () => {
    test('offers != for state, user and session', () => {
      const { result } = renderHook(() => useMlflowTraceFilterFields(), { wrapper });
      const operatorsOf = (id: string) => result.current.find((field) => field.id === id)?.operators;
      expect(operatorsOf('state')).toEqual([FilterOp.EQUALS, FilterOp.NOT_EQUALS]);
      expect(operatorsOf('user')).toEqual([FilterOp.EQUALS, FilterOp.NOT_EQUALS]);
      expect(operatorsOf('session')).toEqual([FilterOp.EQUALS, FilterOp.NOT_EQUALS, FilterOp.CONTAINS]);
    });

    test('suggests values from the loaded rows', () => {
      const { result } = renderHook(() => useMlflowTraceFilterFields(['relevance'], TRACES), { wrapper });
      const field = (id: string) => result.current.find((candidate) => candidate.id === id);

      expect(field('trace_name')).toMatchObject({
        valueInput: 'combobox',
        options: [
          { value: 'agent', label: 'agent' },
          { value: 'chat', label: 'chat' },
        ],
      });
      expect(field('session')?.options?.map(({ value }) => value)).toEqual(['session-1', 'session-2']);
      expect(field('run_name')).toMatchObject({
        valueInput: 'combobox',
        options: [{ value: 'run-1', label: 'run-1' }],
        valuePlaceholder: 'Select or type a run ID',
      });
      expect(field('tag')?.keyInput).toBe('combobox');
      expect(field('tag')?.keyOptions?.map(({ value }) => value)).toEqual(['env', 'mlflow.traceName']);
      expect(
        field('tag')
          ?.getValueOptions?.('env')
          .map(({ value }) => value),
      ).toEqual(['prod', 'staging']);
      expect(field('metadata')?.getValueOptions?.('owner')).toEqual([{ value: 'alice', label: 'alice' }]);
      expect(field('assessment')?.keyOptions?.map(({ value }) => value)).toEqual(['correctness', 'relevance']);
      expect(
        field('assessment')
          ?.getValueOptions?.('relevance')
          .map(({ value }) => value),
      ).toEqual(['2.5', '4']);
    });

    test('keeps value comboboxes without suggestions when no rows are given', () => {
      const { result } = renderHook(() => useMlflowTraceFilterFields(['relevance']), { wrapper });
      const session = result.current.find((field) => field.id === 'session');
      expect(session).toMatchObject({ valueInput: 'combobox', options: [] });
      expect(result.current.find((field) => field.id === 'assessment')?.getValueOptions?.('relevance')).toEqual([]);
    });

    test('accepts saved != clauses for state', () => {
      const { result } = renderHook(() => useMlflowTraceFilterFields(), { wrapper });
      expect(
        isSupportedFilterClause(result.current, { field: 'state', operator: FilterOp.NOT_EQUALS, value: 'OK' }),
      ).toBe(true);
    });
  });
});

describe('filters with the new evaluation runs UI off', () => {
  beforeEach(() => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
  });

  test('compile as before', () => {
    expect(
      compileFilterModel(
        [
          { field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' },
          { field: 'session', operator: FilterOp.EQUALS, value: 's-1' },
          { field: 'assessment', key: 'relevance', operator: FilterOp.EQUALS, value: '4' },
          { field: 'expectation', key: 'expected_score', operator: FilterOp.NOT_EQUALS, value: '3' },
        ],
        NUMERIC_KEYS,
      ),
    ).toEqual([
      "attributes.status = 'ERROR'",
      "request_metadata.mlflow.trace.session = 's-1'",
      "feedback.`relevance` = '4'",
      "expectation.`expected_score` != '3'",
    ]);
  });

  test('fields keep plain inputs and ignore the loaded rows', () => {
    const names = ['relevance'];
    const { result, rerender } = renderHook(({ traces }) => useMlflowTraceFilterFields(names, traces), {
      wrapper,
      initialProps: { traces: TRACES },
    });
    const first = result.current;
    for (const field of first) {
      expect(field.valueInput).not.toBe('combobox');
      expect(field.getValueOptions).toBeUndefined();
    }
    expect(first.find((field) => field.id === 'state')?.operators).toEqual([FilterOp.EQUALS]);
    expect(first.find((field) => field.id === 'tag')?.keyInput).toBeUndefined();

    rerender({ traces: [...TRACES] });
    expect(result.current).toBe(first);
  });

  test('saved views with the new operators are still dropped', () => {
    const { result } = renderHook(() => useMlflowTraceFilterFields(), { wrapper });
    expect(
      isSupportedFilterClause(result.current, { field: 'state', operator: FilterOp.NOT_EQUALS, value: 'OK' }),
    ).toBe(false);
  });
});
