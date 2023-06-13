import { mount } from 'enzyme';
import { useEvaluationArtifactTableData } from './useEvaluationArtifactTableData';
import { fromPairs } from 'lodash';
import { EvaluationArtifactTable } from '../../../types';

describe('useEvaluationArtifactTableData', () => {
  const mountTestComponent = (
    storeData: {
      [runUuid: string]: {
        [artifactPath: string]: EvaluationArtifactTable;
      };
    },
    comparedRunUuids: string[],
    tableNames: string[],
    groupByColumns: string[],
    outputColumn: string,
    intersectingOnly = false,
  ) => {
    let hookResult: ReturnType<typeof useEvaluationArtifactTableData>;
    const TestComponent = () => {
      hookResult = useEvaluationArtifactTableData(
        storeData,
        comparedRunUuids,
        tableNames,
        groupByColumns,
        outputColumn,
        intersectingOnly,
      );

      return null;
    };

    const wrapper = mount(<TestComponent />);

    return { wrapper, getHookResult: () => hookResult };
  };

  describe('properly generates data for a single table', () => {
    const TABLE_NAME = '/table1';
    const COLUMNS = ['input', 'additionalInput', 'answer', 'prompt'];
    const MOCK_STORE = {
      run_a: {
        [TABLE_NAME]: {
          columns: COLUMNS,
          entries: [
            { input: 'question', additionalInput: 'alpha', answer: 'answer_a', prompt: 'prompt_a' },
          ],
          path: TABLE_NAME,
        },
      },
      run_b: {
        [TABLE_NAME]: {
          columns: COLUMNS,
          entries: [
            { input: 'question', additionalInput: 'alpha', answer: 'answer_b', prompt: 'prompt_b' },
          ],
          path: TABLE_NAME,
        },
      },
      run_c: {
        [TABLE_NAME]: {
          columns: COLUMNS,
          entries: [
            { input: 'question', additionalInput: 'beta', answer: 'answer_c', prompt: 'prompt_c' },
          ],
          path: TABLE_NAME,
        },
      },
      run_d: {
        [TABLE_NAME]: {
          columns: COLUMNS,
          entries: [
            { input: 'question', additionalInput: 'beta', answer: 'answer_d', prompt: 'prompt_d' },
          ],
          path: TABLE_NAME,
        },
      },
    };

    const getResultsForRuns = (runIds: string[], groupBy: string[], outputColumn: string) =>
      mountTestComponent(MOCK_STORE, runIds, ['/table1'], groupBy, outputColumn).getHookResult();

    test('properly groups by a single column having one value', () => {
      const results = getResultsForRuns(['run_a', 'run_b', 'run_c'], ['input'], 'answer');
      expect(results).toHaveLength(1);
      expect(results[0].cellValues?.['run_a']).toEqual('answer_a');
      expect(results[0].cellValues?.['run_b']).toEqual('answer_b');
      expect(results[0].cellValues?.['run_c']).toEqual('answer_c');
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
      const results = getResultsForRuns(
        ['run_a', 'run_b', 'run_c', 'run_d'],
        ['input', 'additionalInput'],
        'answer',
      );
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
    const MOCK_STORE = fromPairs(
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
            },
          ]),
        ),
      ]),
    );

    test('yields just a single row if the "group by" column always evaluates to a single value', () => {
      // Preparation: we aggregate/group data by the column that always
      // have the same value regardless of the table and run.
      const results = mountTestComponent(
        MOCK_STORE,
        RUN_IDS,
        TABLES,

        [TestColumns.StaticData],

        // The output column is not important here
        TestColumns.ValueVaryingPerRunAndTable,
      ).getHookResult();

      // We expect only one row
      expect(results).toHaveLength(1);
    });

    test('yields as many rows as there are tables for the "group by" varying by table', () => {
      const results = mountTestComponent(
        MOCK_STORE,
        RUN_IDS,
        TABLES,
        [TestColumns.ValueVaryingPerTable],

        // The output column is not important here
        TestColumns.ValueVaryingPerRunAndTable,
      ).getHookResult();

      // We expect three rows since there are three tables
      expect(results).toHaveLength(3);
    });

    test('yields as many rows as there are runs for the "group by" varying by run', () => {
      const results = mountTestComponent(
        MOCK_STORE,
        RUN_IDS,
        TABLES,
        [TestColumns.ValueVaryingPerRun],

        // The output column is not important here
        TestColumns.ValueVaryingPerRunAndTable,
      ).getHookResult();

      // We expect ten rows since there are ten runs
      expect(results).toHaveLength(10);
    });

    test('yields as many rows as there are runs times tables for the "group by" varying by run and by table', () => {
      const results = mountTestComponent(
        MOCK_STORE,
        RUN_IDS,
        TABLES,
        [TestColumns.ValueVaryingPerRunAndTable],

        // The output column is not important here
        TestColumns.ValueVaryingPerRunAndTable,
      ).getHookResult();

      // Three tables per ten runs with distinct group values
      expect(results).toHaveLength(30);
    });
  });

  describe('properly behaves when results are not covered by all runs', () => {
    const RUN_IDS = ['run_1', 'run_2'];
    const TABLES = ['/table1'];
    const MOCK_COLUMNS = ['question', 'answer'];
    const MOCK_STORE = {
      run_1: {
        '/table1': {
          path: '/table1',
          columns: MOCK_COLUMNS,
          entries: [
            {
              question: 'first_question',
              answer: 'first_answer_run_1',
            },
            {
              question: 'second_question',
              answer: 'second_answer_run_1',
            },
          ],
        },
      },
      run_2: {
        '/table1': {
          path: '/table1',
          columns: MOCK_COLUMNS,
          entries: [
            {
              question: 'first_question',
              answer: 'first_answer_run_2',
            },
            {
              question: 'second_question',
              // Second run doesn't have "answer" value for the second question
              answer: undefined,
            },
          ],
        },
      },
    } as any;

    it('returns results with empty cells when intersectingOnly is set to false', () => {
      const results = mountTestComponent(
        MOCK_STORE,
        RUN_IDS,
        TABLES,
        ['question'],
        'answer',
        false,
      ).getHookResult();

      expect(results).toEqual([
        {
          key: 'first_question',
          groupByCellValues: { question: 'first_question' },
          cellValues: { run_1: 'first_answer_run_1', run_2: 'first_answer_run_2' },
        },
        {
          key: 'second_question',
          groupByCellValues: { question: 'second_question' },
          cellValues: { run_1: 'second_answer_run_1', run_2: undefined },
        },
      ]);
    });

    it('skips results not covered by every run when intersectingOnly is set to true', () => {
      const results = mountTestComponent(
        MOCK_STORE,
        RUN_IDS,
        TABLES,
        ['question'],
        'answer',
        true,
      ).getHookResult();

      expect(results).toEqual([
        {
          key: 'first_question',
          groupByCellValues: { question: 'first_question' },
          cellValues: { run_1: 'first_answer_run_1', run_2: 'first_answer_run_2' },
        },
      ]);
    });
  });

  describe('properly displays overlapping data', () => {
    const RUN_IDS = ['run_1'];
    const TABLES = ['/t1', '/t2'];
    const MOCK_COLUMNS = ['colA', 'colB'];
    const MOCK_STORE = {
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

    it('selects the last valid cell value', () => {
      const results = mountTestComponent(
        MOCK_STORE,
        RUN_IDS,
        TABLES,
        ['colA'],
        'colB',
      ).getHookResult();

      expect(results).toEqual([
        {
          cellValues: { run_1: 'answer_2' },
          groupByCellValues: { colA: 'question' },
          key: 'question',
        },
      ]);
    });
  });
});
