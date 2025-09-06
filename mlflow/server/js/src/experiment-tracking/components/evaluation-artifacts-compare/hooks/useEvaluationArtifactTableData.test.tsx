import { renderHook } from '@testing-library/react';
import { useEvaluationArtifactTableData } from './useEvaluationArtifactTableData';
import { fromPairs } from 'lodash';
import type { EvaluationArtifactTable, PendingEvaluationArtifactTableEntry } from '../../../types';

describe('useEvaluationArtifactTableData', () => {
  const mountTestComponent = ({
    artifactsByRun = {},
    pendingDataByRun = {},
    draftInputValues = [],
    comparedRunUuids = [],
    tableNames = [],
    groupByColumns = [],
    outputColumn = '',
  }: {
    artifactsByRun: {
      [runUuid: string]: {
        [artifactPath: string]: EvaluationArtifactTable;
      };
    };
    pendingDataByRun?: {
      [runUuid: string]: PendingEvaluationArtifactTableEntry[];
    };
    draftInputValues?: Record<string, string>[];
    comparedRunUuids: string[];
    tableNames: string[];
    groupByColumns: string[];
    outputColumn: string;
  }) => {
    const { result } = renderHook(() =>
      useEvaluationArtifactTableData(
        artifactsByRun,
        pendingDataByRun,
        draftInputValues,
        comparedRunUuids,
        tableNames,
        groupByColumns,
        outputColumn,
      ),
    );

    return { getHookResult: () => result.current };
  };

  describe('properly generates data for a single table', () => {
    const TABLE_NAME = '/table1';
    const COLUMNS = ['input', 'additionalInput', 'answer', 'prompt'];
    const MOCK_STORE = {
      run_a: {
        [TABLE_NAME]: {
          columns: COLUMNS,
          entries: [{ input: 'question', additionalInput: 'alpha', answer: 'answer_a', prompt: 'prompt_a' }],
          path: TABLE_NAME,
          rawArtifact: {},
        },
      },
      run_b: {
        [TABLE_NAME]: {
          columns: COLUMNS,
          entries: [{ input: 'question', additionalInput: 'alpha', answer: 'answer_b', prompt: 'prompt_b' }],
          path: TABLE_NAME,
          rawArtifact: {},
        },
      },
      run_c: {
        [TABLE_NAME]: {
          columns: COLUMNS,
          entries: [{ input: 'question', additionalInput: 'beta', answer: 'answer_c', prompt: 'prompt_c' }],
          path: TABLE_NAME,
          rawArtifact: {},
        },
      },
      run_d: {
        [TABLE_NAME]: {
          columns: COLUMNS,
          entries: [{ input: 'question', additionalInput: 'beta', answer: 'answer_d', prompt: 'prompt_d' }],
          path: TABLE_NAME,
          rawArtifact: {},
        },
      },
      run_e: {
        [TABLE_NAME]: {
          columns: COLUMNS,
          entries: [{ input: 'question', additionalInput: 'beta', answer: 0 }],
          path: TABLE_NAME,
          rawArtifact: {},
        },
      },
      run_f: {
        [TABLE_NAME]: {
          columns: COLUMNS,
          entries: [{ input: 'question', additionalInput: 'beta', answer: -0 }],
          path: TABLE_NAME,
          rawArtifact: {},
        },
      },
      run_g: {
        [TABLE_NAME]: {
          columns: COLUMNS,
          entries: [{ input: 'question', additionalInput: 'beta', answer: false }],
          path: TABLE_NAME,
          rawArtifact: {},
        },
      },
      run_h: {
        [TABLE_NAME]: {
          columns: COLUMNS,
          entries: [{ input: 'question', additionalInput: 'beta', answer: NaN }],
          path: TABLE_NAME,
          rawArtifact: {},
        },
      },
      run_i: {
        [TABLE_NAME]: {
          columns: COLUMNS,
          entries: [{ input: 'question', additionalInput: 'beta', answer: null }],
          path: TABLE_NAME,
          rawArtifact: {},
        },
      },
      run_j: {
        [TABLE_NAME]: {
          columns: COLUMNS,
          entries: [{ input: 'question', additionalInput: 'beta', answer: '' }],
          path: TABLE_NAME,
          rawArtifact: {},
        },
      },
      run_k: {
        [TABLE_NAME]: {
          columns: COLUMNS,
          entries: [{ input: 'question', additionalInput: 'beta', answer: undefined }],
          path: TABLE_NAME,
          rawArtifact: {},
        },
      },
    };

    const getResultsForRuns = (comparedRunUuids: string[], groupByColumns: string[], outputColumn: string) =>
      mountTestComponent({
        artifactsByRun: MOCK_STORE,
        comparedRunUuids,
        groupByColumns,
        outputColumn,
        tableNames: ['/table1'],
      }).getHookResult();

    test('properly groups by a single column having one value', () => {
      const results = getResultsForRuns(
        ['run_a', 'run_b', 'run_c', 'run_e', 'run_f', 'run_g', 'run_h', 'run_i', 'run_j', 'run_k'],
        ['input'],
        'answer',
      );
      expect(results).toHaveLength(1);
      expect(results[0].cellValues?.['run_a']).toEqual('answer_a');
      expect(results[0].cellValues?.['run_b']).toEqual('answer_b');
      expect(results[0].cellValues?.['run_c']).toEqual('answer_c');

      // non-nil and non-strings should get stringified
      expect(results[0].cellValues?.['run_e']).toEqual('0');
      expect(results[0].cellValues?.['run_f']).toEqual('0'); // JSON.stringify(-0) === 0
      expect(results[0].cellValues?.['run_g']).toEqual('false');
      expect(results[0].cellValues?.['run_h']).toEqual('null'); // JSON.stringify(NaN) === 'null'
      expect(results[0].cellValues?.['run_i']).toEqual(null);
      expect(results[0].cellValues?.['run_j']).toEqual('');
      expect(results[0].cellValues?.['run_k']).toEqual(undefined);
    });

    test('properly groups by a single column having distinct values', () => {
      const results = getResultsForRuns(['run_a', 'run_b', 'run_c'], ['additionalInput'], 'answer');
      expect(results).toHaveLength(2);
      const valuesForAlpha = results[0].cellValues;
      const valuesForBeta = results[1].cellValues;
      expect(valuesForAlpha?.['run_a']).toEqual('answer_a');
      expect(valuesForAlpha?.['run_b']).toEqual('answer_b');
      expect(valuesForAlpha?.['run_c']).toBeUndefined();

      expect(valuesForBeta?.['run_a']).toBeUndefined();
      expect(valuesForBeta?.['run_b']).toBeUndefined();
      expect(valuesForBeta?.['run_c']).toEqual('answer_c');
    });

    test('properly groups by a single column having unique value for each run', () => {
      const results = getResultsForRuns(['run_a', 'run_b', 'run_c', 'run_d'], ['answer'], 'answer');
      expect(results).toHaveLength(4);
    });

    test('properly groups by multiple columns having distinct values and proper cell values', () => {
      const results = getResultsForRuns(['run_a', 'run_b', 'run_c', 'run_d'], ['input', 'additionalInput'], 'answer');
      expect(results).toHaveLength(2);

      expect(results[0].groupByCellValues['input']).toEqual('question');
      expect(results[0].groupByCellValues['additionalInput']).toEqual('alpha');

      expect(results[1].groupByCellValues['input']).toEqual('question');
      expect(results[1].groupByCellValues['additionalInput']).toEqual('beta');

      const valuesForAlpha = results[0].cellValues;
      const valuesForBeta = results[1].cellValues;

      expect(valuesForAlpha?.['run_a']).toEqual('answer_a');
      expect(valuesForAlpha?.['run_b']).toEqual('answer_b');
      expect(valuesForAlpha?.['run_c']).toBeUndefined();

      expect(valuesForBeta?.['run_a']).toBeUndefined();
      expect(valuesForBeta?.['run_b']).toBeUndefined();
      expect(valuesForBeta?.['run_c']).toEqual('answer_c');
    });

    test('properly ignores entries yielding empty values in "group by" columns', () => {
      const results = getResultsForRuns(
        ['run_a', 'run_b', 'run_c', 'run_d'],
        ['nonExistingColumn', 'otherNonExistingColumn'],
        'answer',
      );
      expect(results).toHaveLength(0);
    });
  });

  describe('properly pulls and generates data for multiple data tables and columns', () => {
    enum TestColumns {
      StaticData = 'StaticData',
      ValueVaryingPerTable = 'ValueVaryingPerTable',
      ValueVaryingPerRun = 'ValueVaryingPerRun',
      ValueVaryingPerRunAndTable = 'ValueVaryingPerRunAndTable',
    }

    // Prepare ten run UUIDS: run_0, run_1, run_2 etc.
    const RUN_IDS = new Array(10).fill(0).map((_, i) => `run_${i + 1}`);
    const TABLES = ['/table_a', '/table_b', '/table_c'];
    // There are four distinct columns, their purpose is described below
    const MOCK_COLUMNS = [
      TestColumns.StaticData,
      TestColumns.ValueVaryingPerTable,
      TestColumns.ValueVaryingPerRun,
      TestColumns.ValueVaryingPerRunAndTable,
    ];
    const mockedArtifactsByRun = fromPairs(
      RUN_IDS.map((runUuid) => [
        runUuid,
        fromPairs(
          TABLES.map((tableName) => [
            tableName,
            {
              path: tableName,
              columns: MOCK_COLUMNS,
              entries: [
                {
                  // Data assigned to this column will always have the same value
                  [TestColumns.StaticData]: 'static_value',
                  // Data assigned to this column will have data varying by table
                  [TestColumns.ValueVaryingPerTable]: `per_table_value_${tableName}`,
                  // Data assigned to this column will have data varying by run
                  [TestColumns.ValueVaryingPerRun]: `per_run_value_${runUuid}`,
                  // Data assigned to this column will have data varying by both run and table
                  [TestColumns.ValueVaryingPerRunAndTable]: `always_unique_value_${tableName}_${runUuid}`,
                },
              ],
              rawArtifact: {},
            },
          ]),
        ),
      ]),
    );

    test('yields just a single row if the "group by" column always evaluates to a single value', () => {
      // Preparation: we aggregate/group data by the column that always
      // have the same value regardless of the table and run.
      const results = mountTestComponent({
        artifactsByRun: mockedArtifactsByRun,
        comparedRunUuids: RUN_IDS,
        groupByColumns: [TestColumns.StaticData],
        tableNames: TABLES,
        // The output column is not important here
        outputColumn: TestColumns.ValueVaryingPerRunAndTable,
      }).getHookResult();

      // We expect only one row
      expect(results).toHaveLength(1);
    });

    test('yields as many rows as there are tables for the "group by" varying by table', () => {
      const results = mountTestComponent({
        artifactsByRun: mockedArtifactsByRun,
        comparedRunUuids: RUN_IDS,
        groupByColumns: [TestColumns.ValueVaryingPerTable],
        tableNames: TABLES,
        // The output column is not important here
        outputColumn: TestColumns.ValueVaryingPerRunAndTable,
      }).getHookResult();

      // We expect three rows since there are three tables
      expect(results).toHaveLength(3);
    });

    test('yields as many rows as there are runs for the "group by" varying by run', () => {
      const results = mountTestComponent({
        artifactsByRun: mockedArtifactsByRun,
        comparedRunUuids: RUN_IDS,
        groupByColumns: [TestColumns.ValueVaryingPerRun],
        tableNames: TABLES,
        // The output column is not important here
        outputColumn: TestColumns.ValueVaryingPerRunAndTable,
      }).getHookResult();

      // We expect ten rows since there are ten runs
      expect(results).toHaveLength(10);
    });

    test('yields as many rows as there are runs times tables for the "group by" varying by run and by table', () => {
      const results = mountTestComponent({
        artifactsByRun: mockedArtifactsByRun,
        comparedRunUuids: RUN_IDS,
        groupByColumns: [TestColumns.ValueVaryingPerRunAndTable],
        tableNames: TABLES,
        // The output column is not important here
        outputColumn: TestColumns.ValueVaryingPerRunAndTable,
      }).getHookResult();

      // Three tables per ten runs with distinct group values
      expect(results).toHaveLength(30);
    });
  });

  describe('properly displays overlapping and draft data', () => {
    const RUN_IDS = ['run_1'];
    const TABLES = ['/t1', '/t2'];
    const MOCK_COLUMNS = ['colA', 'colB'];
    const mockedArtifactsByRun = {
      run_1: {
        '/t1': {
          path: '/t1',
          columns: MOCK_COLUMNS,
          entries: [
            {
              colA: 'question',
              colB: 'answer_1',
            },
          ],
        },
      },
    } as any;

    it('selects the first valid cell value', () => {
      const mockedDuplicatedArtifactsByRun = {
        run_1: {
          '/t1': {
            path: '/t1',
            columns: MOCK_COLUMNS,
            entries: [
              {
                colA: 'question',
                colB: 'answer_1',
              },
            ],
          },
          '/t2': {
            path: '/t2',
            columns: MOCK_COLUMNS,
            entries: [
              {
                colA: 'question',
                colB: 'answer_2',
              },
            ],
          },
        },
      } as any;

      const results = mountTestComponent({
        artifactsByRun: mockedDuplicatedArtifactsByRun,
        comparedRunUuids: RUN_IDS,
        groupByColumns: ['colA'],
        tableNames: TABLES,
        // The output column is not important here
        outputColumn: 'colB',
      }).getHookResult();

      expect(results).toEqual([
        {
          isPendingInputRow: false,
          cellValues: { run_1: 'answer_1' },
          groupByCellValues: { colA: 'question' },
          key: 'question',
        },
      ]);
    });

    it('correctly overwrites fetched data with the pending data', () => {
      // Given data:
      // - pending evaluation entry with colA set to "question" and colB set to "answer_pending"
      const mockedPendingDataByRun = {
        run_1: [
          {
            isPending: true as const,
            entryData: {
              colA: 'question',
              colB: 'answer_pending',
            },
            evaluationTime: 100,
          },
        ],
      };

      const results = mountTestComponent({
        artifactsByRun: mockedArtifactsByRun,
        pendingDataByRun: mockedPendingDataByRun,
        comparedRunUuids: RUN_IDS,
        groupByColumns: ['colA'],
        tableNames: TABLES,
        outputColumn: 'colB',
      }).getHookResult();

      expect(results).toEqual([
        {
          cellValues: { run_1: 'answer_pending' },
          groupByCellValues: { colA: 'question' },
          key: 'question',
          isPendingInputRow: false,
          outputMetadataByRunUuid: {
            run_1: {
              evaluationTime: 100,
              isPending: true,
            },
          },
        },
      ]);
    });

    it('correctly returns fetched artifact data mixed with the pending data in the draft rows, grouped by single column', () => {
      // Given data:
      // - draft input "draft_question" for "colA"
      // - pending evaluation entry with colA set to "question" and colB set to "answer_pending"
      // - pending evaluation entry with colA set to "draft_question" and colB set to "another_answer_pending"
      const mockedPendingDataByRun = {
        run_1: [
          {
            isPending: true,
            entryData: {
              colA: 'question',
              colB: 'answer_pending',
            },
            evaluationTime: 100,
          },
          {
            isPending: true,
            entryData: {
              colA: 'draft_question',
              colB: 'another_answer_pending',
            },
            evaluationTime: 100,
          },
        ],
      };

      const results = mountTestComponent({
        artifactsByRun: mockedArtifactsByRun,
        pendingDataByRun: mockedPendingDataByRun,
        draftInputValues: [{ colA: 'draft_question' }],
        comparedRunUuids: RUN_IDS,
        groupByColumns: ['colA'],
        tableNames: TABLES,
        outputColumn: 'colB',
      }).getHookResult();

      expect(results).toEqual([
        {
          cellValues: { run_1: 'another_answer_pending' },
          groupByCellValues: {
            colA: 'draft_question',
          },
          isPendingInputRow: true,
          key: 'draft_question',
          outputMetadataByRunUuid: {
            run_1: {
              evaluationTime: 100,
              isPending: true,
            },
          },
        },
        {
          cellValues: { run_1: 'answer_pending' },
          groupByCellValues: { colA: 'question' },
          key: 'question',
          isPendingInputRow: false,
          outputMetadataByRunUuid: {
            run_1: {
              evaluationTime: 100,
              isPending: true,
            },
          },
        },
      ]);
    });

    it('correctly returns fetched artifact data mixed with the pending data in the draft rows', () => {
      // Given data:
      // - draft input "draft_question" for "colA"
      // - pending evaluation entry with colA set to "question" and colB set to "answer_pending"
      const mockedPendingDataByRun = {
        run_1: [
          {
            isPending: true,
            entryData: {
              colA: 'evaluated_with_input_a',
              colB: 'evaluated_with_input_b',
              colC: 'evaluated_output_c',
            },
            evaluationTime: 100,
          },
          {
            isPending: true,
            entryData: {
              colA: 'existing_input_a',
              colB: 'existing_input_b',
              colC: 'evaluated_output_c_for_existing_input',
            },
            evaluationTime: 100,
          },
        ],
      } as any;

      const results = mountTestComponent({
        artifactsByRun: {
          run_1: {
            '/t1': {
              path: '/t1',
              columns: ['colA', 'colB', 'colC'],
              entries: [
                {
                  colA: 'existing_input_a',
                  colB: 'existing_input_b',
                  colC: 'existing_output_c',
                },
                {
                  colA: 'another_existing_input_a',
                  colB: 'another_existing_input_b',
                  colC: 'another_existing_output_c',
                },
              ],
            },
          },
        },
        pendingDataByRun: mockedPendingDataByRun,
        draftInputValues: [
          // An input value set that was not evaluated yet
          { colA: 'not_yet_evaluated_input_a', colB: 'not_yet_evaluated_input_b' },
        ],
        comparedRunUuids: RUN_IDS,
        groupByColumns: ['colA', 'colB'],
        tableNames: ['/t1'],
        outputColumn: 'colC',
      }).getHookResult();

      expect(results).toEqual([
        // Row #1: draft input values, not evaluated yet
        {
          cellValues: {},
          groupByCellValues: {
            colA: 'not_yet_evaluated_input_a',
            colB: 'not_yet_evaluated_input_b',
          },
          isPendingInputRow: true,
          key: 'not_yet_evaluated_input_a.not_yet_evaluated_input_b',
          outputMetadataByRunUuid: undefined,
        },
        // Row #2: a new row with freshly evaluated values but not corresponding to the existing draft row
        {
          cellValues: { run_1: 'evaluated_output_c' },
          groupByCellValues: {
            colA: 'evaluated_with_input_a',
            colB: 'evaluated_with_input_b',
          },
          isPendingInputRow: true,
          key: 'evaluated_with_input_a.evaluated_with_input_b',
          outputMetadataByRunUuid: {
            run_1: {
              evaluationTime: 100,
              isPending: true,
            },
          },
        },
        // Row #3: a pre-existing row (key "question" existing in the original data), containing newly evaluated cells
        {
          cellValues: { run_1: 'evaluated_output_c_for_existing_input' },
          groupByCellValues: {
            colA: 'existing_input_a',
            colB: 'existing_input_b',
          },
          key: 'existing_input_a.existing_input_b',
          isPendingInputRow: false,
          outputMetadataByRunUuid: {
            run_1: {
              evaluationTime: 100,
              isPending: true,
            },
          },
        },
        // Row #4: a pre-existing row (key "question" existing in the original data), untouched by evaluation
        {
          cellValues: {
            run_1: 'another_existing_output_c',
          },
          groupByCellValues: {
            colA: 'another_existing_input_a',
            colB: 'another_existing_input_b',
          },
          isPendingInputRow: false,
          key: 'another_existing_input_a.another_existing_input_b',
          outputMetadataByRunUuid: undefined,
        },
      ]);
    });

    it('correctly returns newly evaluated data with draft rows if user provided data in the other order', () => {
      // Given data:
      // - draft input "bar" for "colB" and "foo" for "colA" (note the order)
      // - table grouped by "colA" and "colB" columns
      // - newly evaluated value "test output" for "colC" matching "colA=foo" and "colB=bar"
      const mockedPendingDataByRun = {
        run_1: [
          {
            isPending: true,
            entryData: {
              colA: 'foo',
              colB: 'bar',
              colC: 'test output',
            },
            evaluationTime: 100,
          },
        ],
      };

      const results = mountTestComponent({
        // Provide draft input values in another order than "group by" columns provided
        draftInputValues: [{ colB: 'bar', colA: 'foo' }],
        groupByColumns: ['colA', 'colB'],

        artifactsByRun: {},
        pendingDataByRun: mockedPendingDataByRun,
        comparedRunUuids: ['run_1'],
        tableNames: [],
        outputColumn: 'colC',
      }).getHookResult();

      // We should get only one row
      expect(results.length).toEqual(1);
      expect(results).toEqual([
        {
          key: 'foo.bar',
          groupByCellValues: { colA: 'foo', colB: 'bar' },
          cellValues: { run_1: 'test output' },
          isPendingInputRow: true,
          outputMetadataByRunUuid: {
            run_1: {
              evaluationTime: 100,
              isPending: true,
            },
          },
        },
      ]);
    });
  });
});
