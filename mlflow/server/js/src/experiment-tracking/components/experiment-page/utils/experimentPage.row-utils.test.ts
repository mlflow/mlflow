import {
  shouldEnableToggleIndividualRunsInGroups,
  shouldUseRunRowsVisibilityMap,
} from '../../../../common/utils/FeatureUtils';
import { fromPairs } from 'lodash';
import Utils from '../../../../common/utils/Utils';
import { RUNS_VISIBILITY_MODE } from '../models/ExperimentPageUIState';
import { RunGroupingAggregateFunction, RunGroupingMode, RunRowVisibilityControl } from './experimentPage.row-types';
import {
  type SingleRunData,
  prepareRunsGridData,
  useExperimentRunRows,
  extractRunRowParam,
  extractRunRowParamFloat,
  extractRunRowParamInteger,
} from './experimentPage.row-utils';
import { renderHook } from '@testing-library/react';

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/FeatureUtils')>(
    '../../../../common/utils/FeatureUtils',
  ),
  shouldUseRunRowsVisibilityMap: jest.fn(() => false),
  shouldEnableToggleIndividualRunsInGroups: jest.fn(),
}));

const LOGGED_MODEL = { LOGGED_MODEL: true };

// Do not test tag->logged model transformation
Utils.getLoggedModelsFromTags = jest.fn().mockReturnValue([LOGGED_MODEL]);

const MOCK_EXPERIMENTS = [
  {
    experimentId: '1',
    name: '/Users/john.doe@databricks.com/test-experiment-1',
  },
  {
    experimentId: '2',
    name: '/Users/john.doe@databricks.com/test-experiment-2',
  },
];
const MOCK_MODEL_MAP = {
  run1_2: [
    {
      this_is_a_registered_model_mock: '1',
    },
  ],
};

const MOCK_RUN_DATA = [
  {
    runInfo: { experimentId: '1', runUuid: 'run1_1' },
    metrics: [{ key: 'met1', value: 111.123456789 }],
    params: [{ key: 'p1', value: '123' }],
    tags: { testtag1: { key: 'testtag1', value: 'testval1' } },
  },
  {
    runInfo: { experimentId: '1', runUuid: 'run1_2' },
    metrics: [{ key: 'met1', value: 222 }],
    tags: {
      testtag1: { key: 'testtag1', value: 'testval2' },
      'mlflow.parentRunId': { key: 'mlflow.parentRunId', value: 'run1_1' },
    },
  },
  {
    runInfo: { experimentId: '1', runUuid: 'run1_3' },
    tags: {
      testtag1: { key: 'testtag1', value: 'testval3' },
      'mlflow.parentRunId': { key: 'mlflow.parentRunId', value: 'run1_1' },
    },
  },
  {
    runInfo: { experimentId: '1', runUuid: 'run1_4' },
    metrics: [
      { key: 'met1', value: 1122 },
      { key: 'met2', value: 2211 },
    ],
    params: [
      { key: 'p1', value: '1234' },
      { key: 'p2', value: '12345' },
    ],
    tags: { 'mlflow.parentRunId': { key: 'mlflow.parentRunId', value: 'run1_2' } },
  },
  {
    runInfo: { experimentId: '2', runUuid: 'run2_1' },
  },
  {
    runInfo: { experimentId: '2', runUuid: 'run2_2' },
  },
];

const METRIC_KEYS = ['met1', 'met2'];
const PARAM_KEYS = ['p1', 'p2'];
const TAG_KEYS = ['testtag1'];

const commonPrepareRunsGridDataParams = {
  experiments: MOCK_EXPERIMENTS as any,
  runData: MOCK_RUN_DATA as any,
  paramKeyList: PARAM_KEYS,
  metricKeyList: METRIC_KEYS,
  nestChildren: false,
  referenceTime: new Date(1000),
  modelVersionsByRunUuid: MOCK_MODEL_MAP as any,
  tagKeyList: TAG_KEYS,
  runsExpanded: {},
  runsPinned: [],
  runsHidden: [],
  runsVisibilityMap: {},
  runUuidsMatchingFilter: MOCK_RUN_DATA.map((r) => r.runInfo.runUuid),
  groupBy: null,
  groupsExpanded: {},
  colorByRunUuid: {},
};

describe('ExperimentViewRuns row utils, nested and flat run hierarchies', () => {
  beforeEach(() => {
    jest.mocked(shouldEnableToggleIndividualRunsInGroups).mockImplementation(() => false);
  });
  test('it creates proper row dataset for a flat run list', () => {
    const runsGridData = prepareRunsGridData(commonPrepareRunsGridDataParams);

    // Assert proper amount of run rows
    expect(runsGridData.length).toBe(MOCK_RUN_DATA.length);

    // Assert runDateInfo construction
    expect(runsGridData.map((runData) => runData.runDateAndNestInfo)).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          isParent: false,
          hasExpander: false,
          expanderOpen: false,
          childrenIds: [],
          level: 0,
          referenceTime: new Date(1000),
        }),
      ]),
    );

    // Assert passing through run infos
    expect(runsGridData.map((runData) => runData.runInfo)).toEqual(MOCK_RUN_DATA.map((d) => d.runInfo));

    // Assert model for the second run (index #1)
    expect(runsGridData[1].models?.registeredModels).toEqual(
      expect.arrayContaining([expect.objectContaining(MOCK_MODEL_MAP.run1_2[0])]),
    );
    expect(runsGridData[1].models?.loggedModels).toEqual(expect.arrayContaining([LOGGED_MODEL]));

    expect(runsGridData.map((runData) => runData.version)).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          version: expect.anything(),
          name: expect.anything(),
          type: expect.anything(),
        }),
      ]),
    );

    // Assert run #1 KV data
    expect(runsGridData[0]['$$$param$$$-p1']).toEqual('123');
    expect(runsGridData[0]['$$$param$$$-p2']).toEqual('-');
    expect(runsGridData[0]['$$$metric$$$-met1']).toEqual(111.123456789);
    expect(runsGridData[0]['$$$metric$$$-met2']).toEqual('-');
    expect(runsGridData[0]['$$$tag$$$-testtag1']).toEqual('testval1');

    // Assert run #2 KV data
    expect(runsGridData[1]['$$$param$$$-p1']).toEqual('-');
    expect(runsGridData[1]['$$$param$$$-p2']).toEqual('-');
    expect(runsGridData[1]['$$$metric$$$-met1']).toEqual(222);
    expect(runsGridData[1]['$$$metric$$$-met2']).toEqual('-');
    expect(runsGridData[1]['$$$tag$$$-testtag1']).toEqual('testval2');

    // Assert run #3 KV data
    expect(runsGridData[2]['$$$param$$$-p1']).toEqual('-');
    expect(runsGridData[2]['$$$param$$$-p2']).toEqual('-');
    expect(runsGridData[2]['$$$metric$$$-met1']).toEqual('-');
    expect(runsGridData[2]['$$$metric$$$-met2']).toEqual('-');
    expect(runsGridData[2]['$$$tag$$$-testtag1']).toEqual('testval3');

    // Assert run #4 KV data
    expect(runsGridData[3]['$$$param$$$-p1']).toEqual('1234');
    expect(runsGridData[3]['$$$param$$$-p2']).toEqual('12345');
    expect(runsGridData[3]['$$$metric$$$-met1']).toEqual(1122);
    expect(runsGridData[3]['$$$metric$$$-met2']).toEqual(2211);
    expect(runsGridData[3]['$$$tag$$$-testtag1']).toEqual('-');

    // Assert run #5 KV data
    expect(runsGridData[4]['$$$param$$$-p1']).toEqual('-');
    expect(runsGridData[4]['$$$param$$$-p2']).toEqual('-');
    expect(runsGridData[4]['$$$metric$$$-met1']).toEqual('-');
    expect(runsGridData[4]['$$$metric$$$-met2']).toEqual('-');
    expect(runsGridData[4]['$$$tag$$$-testtag1']).toEqual('-');

    // Assert run #6 KV data
    expect(runsGridData[5]['$$$param$$$-p1']).toEqual('-');
    expect(runsGridData[5]['$$$param$$$-p2']).toEqual('-');
    expect(runsGridData[5]['$$$metric$$$-met1']).toEqual('-');
    expect(runsGridData[5]['$$$metric$$$-met2']).toEqual('-');
    expect(runsGridData[5]['$$$tag$$$-testtag1']).toEqual('-');
  });

  test('it creates proper row dataset for a nested and unexpanded run list', () => {
    const runsGridData = prepareRunsGridData({
      ...commonPrepareRunsGridDataParams,
      nestChildren: true,
    });

    // Assert proper amount of run rows - only one for experiment #1 and two for experiment #2, equals three
    // Resembling the following structure:
    // [+] run1_1
    //     run2_1
    //     run2_2
    expect(runsGridData.length).toBe(3);
    expect(runsGridData[0].runDateAndNestInfo?.isParent).toEqual(true);
    expect(runsGridData[0].runDateAndNestInfo?.hasExpander).toEqual(true);
    expect(runsGridData[0].runDateAndNestInfo?.childrenIds).toEqual(['run1_2', 'run1_3']);
  });

  test('it creates proper row dataset for a nested run list with one run expanded', () => {
    const runsGridData = prepareRunsGridData({
      ...commonPrepareRunsGridDataParams,
      nestChildren: true,
      runsExpanded: { run1_1: true },
    });

    // Assert proper amount of run rows - three for experiment #1 and two for experiment #2, equals five
    // Resembling the following structure:
    // [-] run1_1
    //     [+]  run1_2
    //          run1_3
    //     run2_1
    //     run2_2
    expect(runsGridData.length).toBe(5);
    expect(runsGridData[0].runDateAndNestInfo?.isParent).toEqual(true);
    expect(runsGridData[0].runDateAndNestInfo?.hasExpander).toEqual(true);
    expect(runsGridData[0].runDateAndNestInfo?.expanderOpen).toEqual(true);
    expect(runsGridData[0].runDateAndNestInfo?.childrenIds).toEqual(['run1_2', 'run1_3']);

    expect(runsGridData[1].runDateAndNestInfo?.isParent).toEqual(true);
    expect(runsGridData[1].runDateAndNestInfo?.hasExpander).toEqual(true);
    expect(runsGridData[1].runDateAndNestInfo?.expanderOpen).toEqual(false);
    expect(runsGridData[1].runDateAndNestInfo?.childrenIds).toEqual(['run1_4']);

    expect(runsGridData[2].runDateAndNestInfo?.isParent).toEqual(false);
    expect(runsGridData[2].runDateAndNestInfo?.hasExpander).toEqual(false);
    expect(runsGridData[2].runDateAndNestInfo?.expanderOpen).toEqual(false);
  });

  test('it creates proper row dataset for a nested run list with two runs expanded', () => {
    const runsGridData = prepareRunsGridData({
      ...commonPrepareRunsGridDataParams,
      nestChildren: true,
      runsExpanded: { run1_1: true, run1_2: true },
    });

    // Assert proper amount of run rows - three for experiment #1 and two for experiment #2, equals five.
    // Resembling the following structure:
    // [-] run1_1
    //     [-]  run1_2
    //               run1_4
    //          run1_3
    //     run2_1
    //     run2_2
    expect(runsGridData.length).toBe(6);
    expect(runsGridData[0].runDateAndNestInfo?.isParent).toEqual(true);
    expect(runsGridData[0].runDateAndNestInfo?.hasExpander).toEqual(true);
    expect(runsGridData[0].runDateAndNestInfo?.expanderOpen).toEqual(true);
    expect(runsGridData[0].runDateAndNestInfo?.childrenIds).toEqual(['run1_2', 'run1_3']);

    expect(runsGridData[1].runDateAndNestInfo?.isParent).toEqual(true);
    expect(runsGridData[1].runDateAndNestInfo?.hasExpander).toEqual(true);
    expect(runsGridData[1].runDateAndNestInfo?.expanderOpen).toEqual(true);
    expect(runsGridData[1].runDateAndNestInfo?.childrenIds).toEqual(['run1_4']);
  });

  test('it disallow expanding nested runs that parents are not expanded', () => {
    const runsGridData = prepareRunsGridData({
      ...commonPrepareRunsGridDataParams,
      nestChildren: true,
      runsExpanded: { run1_2: true },
    });

    // Assert proper amount of run rows - only one for experiment #1 and two for experiment #2, equals three
    // Resembling the following structure:
    // [+] run1_1 - has expanded child inside but it shouldn't matter
    //     run2_1
    //     run2_2
    expect(runsGridData.length).toBe(3);
  });

  test('it does not expand unknown rows', () => {
    const runsGridData = prepareRunsGridData({
      ...commonPrepareRunsGridDataParams,
      nestChildren: true,
      runsExpanded: { someWeirdRowThatDoesNotExist: true },
    });

    // Assert proper amount of run rows - only one for experiment #1 and two for experiment #2, equals three
    // Resembling the following structure:
    // [+] run1_1 - has expanded child inside but it shouldn't matter
    //     run2_1
    //     run2_2
    expect(runsGridData.length).toBe(3);
  });

  test('it does not break on cycle exiting in row-parent mapping', () => {
    const runsGridData = prepareRunsGridData({
      ...commonPrepareRunsGridDataParams,
      runData: [
        {
          runInfo: { experimentId: '1', runUuid: 'run1_1' },
        },
        {
          runInfo: { experimentId: '1', runUuid: 'run1_2' },
          tags: {
            'mlflow.parentRunId': { key: 'mlflow.parentRunId', value: 'run1_4' },
          },
        },
        {
          runInfo: { experimentId: '1', runUuid: 'run1_3' },
          tags: {
            'mlflow.parentRunId': { key: 'mlflow.parentRunId', value: 'run1_2' },
          },
        },
        {
          runInfo: { experimentId: '1', runUuid: 'run1_4' },
          tags: {
            'mlflow.parentRunId': { key: 'mlflow.parentRunId', value: 'run1_3' },
          },
        },
      ] as any,
      nestChildren: true,
      runsExpanded: { run1_1: true, run1_2: true, run1_3: true, run1_4: true },
    });

    // Assert proper amount of run rows - only one run should be used since the rest is erroneous
    expect(runsGridData.length).toBe(1);
  });

  test('it correctly separates pinned and unpinned rows', () => {
    const runsGridData = prepareRunsGridData({
      ...commonPrepareRunsGridDataParams,
      runData: [
        {
          runInfo: { experimentId: '1', runUuid: 'run1_1' },
        },
        {
          runInfo: { experimentId: '1', runUuid: 'run1_2' },
          tags: {
            'mlflow.parentRunId': { key: 'mlflow.parentRunId', value: 'run1_1' },
          },
        },
        {
          runInfo: { experimentId: '1', runUuid: 'run1_3' },
        },
        {
          runInfo: { experimentId: '1', runUuid: 'run1_4' },
          tags: {
            'mlflow.parentRunId': { key: 'mlflow.parentRunId', value: 'run1_3' },
          },
        },
      ] as any,
      runsPinned: ['run1_3', 'run1_4'],
      nestChildren: true,
      runsExpanded: { run1_1: true, run1_2: true, run1_3: true, run1_4: true },
    });

    // Assert properly ordered run rows - first pinned ones (with proper flag) and hierarchy,
    // then the rest.
    // It should resembling the following structure:
    // [-] [X] run1_3
    //              run2_4
    // [-] [ ] run1_1
    //              run2_2
    expect(
      runsGridData.map(({ runUuid, pinned, runDateAndNestInfo: { isParent } = {} }) => ({
        runUuid,
        pinned,
        isParent,
      })),
    ).toEqual([
      {
        pinned: true,
        runUuid: 'run1_3',
        isParent: true,
      },
      {
        pinned: true,
        runUuid: 'run1_4',
        isParent: false,
      },
      {
        pinned: false,
        runUuid: 'run1_1',
        isParent: true,
      },
      {
        pinned: false,
        runUuid: 'run1_2',
        isParent: false,
      },
    ]);
  });

  test('it correctly hides fetched but unpinned rows', () => {
    const runsGridData = prepareRunsGridData({
      ...commonPrepareRunsGridDataParams,
      runData: [
        {
          runInfo: { experimentId: '1', runUuid: 'run1_1' },
        },
        {
          runInfo: { experimentId: '1', runUuid: 'run1_2' },
        },
        {
          runInfo: { experimentId: '1', runUuid: 'run1_3' },
        },
        {
          runInfo: { experimentId: '1', runUuid: 'run1_4' },
        },
      ] as any,
      runsPinned: ['run1_2', 'run1_3'],
      nestChildren: true,
      runUuidsMatchingFilter: ['run1_1', 'run1_2'],
    });

    expect(
      runsGridData.map(({ runUuid, pinned }) => ({
        runUuid,
        pinned,
      })),
    ).toEqual(
      expect.arrayContaining([
        { pinned: false, runUuid: 'run1_1' },
        { pinned: true, runUuid: 'run1_2' },
        { pinned: true, runUuid: 'run1_3' },
      ]),
    );
  });
});

describe.each([
  {
    individualRunsInGroupsFlagValue: false,
  },
  {
    individualRunsInGroupsFlagValue: true,
  },
])(
  'ExperimentViewRuns row utils, grouped run hierarchy - individual run toggling set to $individualRunsInGroupsFlagValue',
  ({ individualRunsInGroupsFlagValue }) => {
    beforeEach(() => {
      jest.mocked(shouldEnableToggleIndividualRunsInGroups).mockImplementation(() => individualRunsInGroupsFlagValue);
    });

    describe.each([
      ['when using runsVisibilityMap UI state', true],
      ['when using legacy runsHidden UI state', false],
    ])('Configurable runs visibility mode  %s', (_, useExplicitRunRowsVisibility) => {
      beforeEach(() => {
        jest.mocked(shouldUseRunRowsVisibilityMap).mockImplementation(() => useExplicitRunRowsVisibility);
      });

      const fiftyRuns: SingleRunData[] = new Array(50).fill(0).map((_, i) => ({
        runInfo: { experimentId: '1', runUuid: `run1_${i}` } as any,
        datasets: [],
        metrics: [],
        params: [],
        tags: {},
      }));

      const userSelectedRunsHidden = ['run1_4', 'run1_22'];

      test.each([10, 20] as const)('it correctly marks first %d runs as visible regardless of the order', (amount) => {
        let runsGridData = prepareRunsGridData({
          ...commonPrepareRunsGridDataParams,
          runsHiddenMode: RUNS_VISIBILITY_MODE[`FIRST_${amount}_RUNS`],
          runUuidsMatchingFilter: fiftyRuns.map((r) => r.runInfo.runUuid),
          runData: fiftyRuns,
          runsHidden: userSelectedRunsHidden,
          runsVisibilityMap: fromPairs(userSelectedRunsHidden.map((runUuid) => [runUuid, false])),
        });
        expect(runsGridData.length).toBe(50);
        expect(runsGridData.slice(0, amount).every((r) => r.hidden)).toBe(false);
        expect(runsGridData.slice(amount).every((r) => r.hidden)).toBe(true);

        const fiftyRunsReversed = [...fiftyRuns].reverse();

        runsGridData = prepareRunsGridData({
          ...commonPrepareRunsGridDataParams,
          runsHiddenMode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS,
          runUuidsMatchingFilter: fiftyRunsReversed.map((r) => r.runInfo.runUuid),
          runData: fiftyRunsReversed,
          runsHidden: userSelectedRunsHidden,
          runsVisibilityMap: fromPairs(userSelectedRunsHidden.map((runUuid) => [runUuid, false])),
        });
        expect(runsGridData.length).toBe(50);
        expect(runsGridData.slice(0, amount).every((r) => r.hidden)).toBe(false);
        expect(runsGridData.slice(amount).every((r) => r.hidden)).toBe(true);
      });

      test('it correctly marks specific runs as hidden', () => {
        const runsGridData = prepareRunsGridData({
          ...commonPrepareRunsGridDataParams,
          runsHiddenMode: RUNS_VISIBILITY_MODE.CUSTOM,
          runUuidsMatchingFilter: fiftyRuns.map((r) => r.runInfo.runUuid),
          runData: fiftyRuns,
          runsHidden: userSelectedRunsHidden,
          runsVisibilityMap: fromPairs(userSelectedRunsHidden.map((runUuid) => [runUuid, false])),
        });
        expect(runsGridData.length).toBe(50);

        for (const resultingRow of runsGridData) {
          if (userSelectedRunsHidden.includes(resultingRow.runUuid)) {
            expect(resultingRow.hidden).toBe(true);
          } else {
            expect(resultingRow.hidden).toBe(false);
          }
        }
      });

      test('it correctly marks all runs as hidden', () => {
        const runsGridData = prepareRunsGridData({
          ...commonPrepareRunsGridDataParams,
          runsHiddenMode: RUNS_VISIBILITY_MODE.HIDEALL,
          runUuidsMatchingFilter: fiftyRuns.map((r) => r.runInfo.runUuid),
          runData: fiftyRuns,
          runsHidden: userSelectedRunsHidden,
          runsVisibilityMap: fromPairs(userSelectedRunsHidden.map((runUuid) => [runUuid, false])),
        });
        expect(runsGridData.length).toBe(50);

        expect(runsGridData.every((r) => r.hidden)).toBe(true);
      });

      test('it correctly marks all runs as visible', () => {
        const runsGridData = prepareRunsGridData({
          ...commonPrepareRunsGridDataParams,
          runsHiddenMode: RUNS_VISIBILITY_MODE.SHOWALL,
          runUuidsMatchingFilter: fiftyRuns.map((r) => r.runInfo.runUuid),
          runData: fiftyRuns,
          runsHidden: userSelectedRunsHidden,
          runsVisibilityMap: fromPairs(userSelectedRunsHidden.map((runUuid) => [runUuid, false])),
        });
        expect(runsGridData.length).toBe(50);

        expect(runsGridData.every((r) => r.hidden)).toBe(false);
      });

      test('it hides finished runs when runsHiddenMode is HIDE_FINISHED_RUNS', () => {
        const runsWithStatuses: SingleRunData[] = [
          {
            runInfo: { experimentId: '1', runUuid: 'run_active', status: 'RUNNING' } as any,
            datasets: [],
            metrics: [],
            params: [],
            tags: {},
          },
          {
            runInfo: { experimentId: '1', runUuid: 'run_finished', status: 'FINISHED' } as any,
            datasets: [],
            metrics: [],
            params: [],
            tags: {},
          },
          {
            runInfo: { experimentId: '1', runUuid: 'run_failed', status: 'FAILED' } as any,
            datasets: [],
            metrics: [],
            params: [],
            tags: {},
          },
        ];

        const runsGridData = prepareRunsGridData({
          ...commonPrepareRunsGridDataParams,
          runsHiddenMode: RUNS_VISIBILITY_MODE.HIDE_FINISHED_RUNS,
          runData: runsWithStatuses,
          runUuidsMatchingFilter: runsWithStatuses.map((r) => r.runInfo.runUuid),
        });

        const visibleRunUuids = runsGridData.filter((r) => !r.hidden).map((r) => r.runUuid);
        expect(visibleRunUuids).toEqual(['run_active']);
      });
    });
  },
);

describe('ExperimentViewRuns row utils, grouped run hierarchy - selecting individual runs', () => {
  beforeEach(() => {
    jest.mocked(shouldEnableToggleIndividualRunsInGroups).mockImplementation(() => true);
  });

  const createNRuns = (n = 30): SingleRunData[] =>
    new Array(n).fill(0).map((_, i) => ({
      runInfo: { experimentId: '1', runUuid: `run1_${i}` } as any,
      datasets: [],
      metrics: [],
      params: [
        {
          key: 'test-param',
          value: `value-${Math.floor(i / 10) + 1}`,
        },
      ],
      tags: {},
    }));

  test.each([10, 20] as const)('it correctly determines visibility for first %s runs', (amount) => {
    const runData = createNRuns(30);
    const runsGridData = prepareRunsGridData({
      ...commonPrepareRunsGridDataParams,
      groupBy: {
        aggregateFunction: RunGroupingAggregateFunction.Min,
        groupByKeys: [{ mode: RunGroupingMode.Param, groupByData: 'test-param' }],
      },
      runsHiddenMode: RUNS_VISIBILITY_MODE[`FIRST_${amount}_RUNS`],
      runUuidsMatchingFilter: runData.map((r) => r.runInfo.runUuid),
      runData,
      useGroupedValuesInCharts: false,
    });

    const runRows = runsGridData.filter(({ runInfo }) => runInfo);
    const firstRunRows = runRows.slice(0, amount);
    const remainingRunRows = runRows.slice(amount);

    expect(firstRunRows.some((row) => row.hidden)).toBe(false);
    expect(remainingRunRows.every((row) => row.hidden)).toBe(true);
  });

  test('it correctly determines visibility control for first 10 runs', () => {
    const runData = createNRuns(30);
    const runsGridData = prepareRunsGridData({
      ...commonPrepareRunsGridDataParams,
      groupBy: {
        aggregateFunction: RunGroupingAggregateFunction.Min,
        groupByKeys: [{ mode: RunGroupingMode.Param, groupByData: 'test-param' }],
      },
      runsHiddenMode: RUNS_VISIBILITY_MODE.FIRST_10_RUNS,
      runUuidsMatchingFilter: runData.map((r) => r.runInfo.runUuid),
      runData,
      useGroupedValuesInCharts: false,
    });

    const runRows = runsGridData.filter(({ runInfo }) => runInfo);
    const groupRows = runsGridData.filter(({ groupParentInfo }) => groupParentInfo);

    expect(runRows.every((row) => row.visibilityControl === RunRowVisibilityControl.Enabled)).toBe(true);

    // When control over individual runs is enabled, we expect all groups to have visibility control enabled as well
    expect(groupRows.every((row) => row.visibilityControl === RunRowVisibilityControl.Enabled)).toBe(true);
  });

  test.each([
    ['when using runsVisibilityMap', true],
    ['when using legacy runsHidden', false],
  ])(
    'it correctly marks specific grouped runs as hidden and exclude them from aggregation %s',
    (_, usingRunsVisibilityMap) => {
      jest.mocked(shouldUseRunRowsVisibilityMap).mockImplementation(() => false);
      jest.mocked(shouldUseRunRowsVisibilityMap).mockImplementation(() => usingRunsVisibilityMap);

      const userSelectedRunsHidden = ['run1_4', 'run1_22'];

      const runData = createNRuns(30);
      const runsGridData = prepareRunsGridData({
        ...commonPrepareRunsGridDataParams,
        runsHiddenMode: RUNS_VISIBILITY_MODE.CUSTOM,
        groupBy: {
          aggregateFunction: RunGroupingAggregateFunction.Min,
          groupByKeys: [{ mode: RunGroupingMode.Param, groupByData: 'test-param' }],
        },
        runUuidsMatchingFilter: runData.map((r) => r.runInfo.runUuid),
        runData,
        runsHidden: userSelectedRunsHidden,
        runsVisibilityMap: fromPairs(userSelectedRunsHidden.map((runUuid) => [runUuid, false])),
        useGroupedValuesInCharts: false,
      });

      expect(runsGridData.length).toBe(33);
      const groupRows = runsGridData.filter(({ groupParentInfo }) => groupParentInfo);

      for (const resultingRow of runsGridData) {
        expect(resultingRow.hidden).toBe(userSelectedRunsHidden.includes(resultingRow.runUuid));
      }

      for (const groupRow of groupRows) {
        for (const excludedRunUuid of userSelectedRunsHidden) {
          if (groupRow.groupParentInfo?.runUuids.includes(excludedRunUuid)) {
            expect(groupRow.groupParentInfo.runUuidsForAggregation).not.toContain(excludedRunUuid);
          }
        }
      }
    },
  );

  test('it correctly disables visibility control for runs in hidden groups', () => {
    const hiddenRows = ['param.test-param.value-2'];

    const runData = createNRuns(30);
    const runsGridData = prepareRunsGridData({
      ...commonPrepareRunsGridDataParams,
      runsHiddenMode: RUNS_VISIBILITY_MODE.CUSTOM,
      groupBy: {
        aggregateFunction: RunGroupingAggregateFunction.Min,
        groupByKeys: [{ mode: RunGroupingMode.Param, groupByData: 'test-param' }],
      },
      runUuidsMatchingFilter: runData.map((r) => r.runInfo.runUuid),
      runData,
      runsHidden: hiddenRows,
      runsVisibilityMap: fromPairs(hiddenRows.map((runUuid) => [runUuid, false])),
    });

    const runRows = runsGridData.filter(({ runInfo }) => runInfo);

    expect(runRows.slice(0, 10).every((row) => row.visibilityControl === RunRowVisibilityControl.Enabled)).toBe(true);
    expect(runRows.slice(10, 20).every((row) => row.visibilityControl === RunRowVisibilityControl.Disabled)).toBe(true);
    expect(runRows.slice(20).every((row) => row.visibilityControl === RunRowVisibilityControl.Enabled)).toBe(true);
  });

  test.each([10, 20] as const)(
    'it correctly determines visibility for grouped runs mixed with ungrouped runs for with %s runs',
    (amount) => {
      // Generate sample set, but only 20 first runs will have params set
      const runData = createNRuns(50).map((run, i) => {
        if (i >= 20) {
          run.params = [];
        }
        return run;
      });

      const runsGridData = prepareRunsGridData({
        ...commonPrepareRunsGridDataParams,
        runsHiddenMode: RUNS_VISIBILITY_MODE[`FIRST_${amount}_RUNS`],
        groupBy: {
          aggregateFunction: RunGroupingAggregateFunction.Min,
          groupByKeys: [{ mode: RunGroupingMode.Param, groupByData: 'test-param' }],
        },
        runUuidsMatchingFilter: runData.map((r) => r.runInfo.runUuid),
        runData,
        groupsExpanded: { 'param.test-param': true },
      });

      const runRows = runsGridData.filter(({ runInfo }) => runInfo);
      const groupRows = runsGridData.filter(({ groupParentInfo }) => groupParentInfo);

      expect(groupRows[0].groupParentInfo?.runUuids).toHaveLength(10);
      expect(groupRows[1].groupParentInfo?.runUuids).toHaveLength(10);
      expect(groupRows[2].groupParentInfo?.runUuids).toHaveLength(30);

      expect(groupRows[0].groupParentInfo?.isRemainingRunsGroup).toBe(false);
      expect(groupRows[1].groupParentInfo?.isRemainingRunsGroup).toBe(false);
      expect(groupRows[2].groupParentInfo?.isRemainingRunsGroup).toBe(true);

      const ungroupedRows = runRows.filter(({ runDateAndNestInfo }) => !runDateAndNestInfo?.belongsToGroup);
      const visibleUngroupedRows = ungroupedRows.filter(({ hidden }) => !hidden);
      const groupRowsWithRuns = groupRows.filter(({ groupParentInfo }) => !groupParentInfo?.isRemainingRunsGroup);

      expect(visibleUngroupedRows).toHaveLength(amount - groupRowsWithRuns.length);
    },
  );
});

describe('ExperimentViewRuns row utils, utility functions', () => {
  const mockRunRow = {
    runUuid: 'test-run',
    params: [
      { key: 'learning_rate', value: '0.01' },
      { key: 'batch_size', value: '32' },
      { key: 'model_type', value: 'cnn' },
      { key: 'epochs', value: '100' },
    ],
  } as any;

  describe('extractRunRowParam', () => {
    test('it extracts existing parameter value', () => {
      expect(extractRunRowParam(mockRunRow, 'learning_rate')).toBe('0.01');
      expect(extractRunRowParam(mockRunRow, 'model_type')).toBe('cnn');
    });

    test('it returns fallback for non-existing parameter', () => {
      expect(extractRunRowParam(mockRunRow, 'non_existing')).toBeUndefined();
      expect(extractRunRowParam(mockRunRow, 'non_existing', undefined)).toBeUndefined();
    });

    test('it handles empty params array', () => {
      const emptyParamsRow = { ...mockRunRow, params: [] };
      expect(extractRunRowParam(emptyParamsRow, 'learning_rate')).toBeUndefined();
      expect(extractRunRowParam(emptyParamsRow, 'learning_rate', undefined)).toBeUndefined();
    });

    test('it handles undefined params', () => {
      const noParamsRow = { ...mockRunRow, params: undefined };
      expect(extractRunRowParam(noParamsRow, 'learning_rate')).toBeUndefined();
      expect(extractRunRowParam(noParamsRow, 'learning_rate', undefined)).toBeUndefined();
    });
  });

  describe('extractRunRowParamFloat', () => {
    test('it extracts and converts valid float parameter', () => {
      expect(extractRunRowParamFloat(mockRunRow, 'learning_rate')).toBe(0.01);
    });

    test('it returns fallback for non-numeric parameter', () => {
      expect(extractRunRowParamFloat(mockRunRow, 'model_type')).toBeUndefined();
      expect(extractRunRowParamFloat(mockRunRow, 'model_type', undefined)).toBeUndefined();
    });

    test('it returns fallback for non-existing parameter', () => {
      expect(extractRunRowParamFloat(mockRunRow, 'non_existing')).toBeUndefined();
      expect(extractRunRowParamFloat(mockRunRow, 'non_existing', undefined)).toBeUndefined();
    });

    test('it handles integer values', () => {
      expect(extractRunRowParamFloat(mockRunRow, 'batch_size')).toBe(32);
      expect(extractRunRowParamFloat(mockRunRow, 'epochs')).toBe(100);
    });
  });

  describe('extractRunRowParamInteger', () => {
    test('it extracts and converts valid integer parameter', () => {
      expect(extractRunRowParamInteger(mockRunRow, 'batch_size')).toBe(32);
      expect(extractRunRowParamInteger(mockRunRow, 'epochs')).toBe(100);
    });

    test('it converts float to integer', () => {
      expect(extractRunRowParamInteger(mockRunRow, 'learning_rate')).toBe(0);
    });

    test('it returns fallback for non-numeric parameter', () => {
      expect(extractRunRowParamInteger(mockRunRow, 'model_type')).toBeUndefined();
      expect(extractRunRowParamInteger(mockRunRow, 'model_type', undefined)).toBeUndefined();
    });

    test('it returns fallback for non-existing parameter', () => {
      expect(extractRunRowParamInteger(mockRunRow, 'non_existing')).toBeUndefined();
      expect(extractRunRowParamInteger(mockRunRow, 'non_existing', undefined)).toBeUndefined();
    });
  });
});

describe('ExperimentViewRuns row utils, useExperimentRunRows hook', () => {
  beforeEach(() => {
    jest.mocked(shouldEnableToggleIndividualRunsInGroups).mockImplementation(() => false);
  });

  test('it returns memoized run rows data', () => {
    const { result, rerender } = renderHook(() => useExperimentRunRows(commonPrepareRunsGridDataParams));

    const firstResult = result.current;
    expect(firstResult).toHaveLength(MOCK_RUN_DATA.length);

    // Rerender with same props should return same reference (memoized)
    rerender();
    expect(result.current).toBe(firstResult);
  });

  test('it recalculates when dependencies change', () => {
    const { result, rerender } = renderHook(
      ({ runsPinned }: { runsPinned: string[] }) =>
        useExperimentRunRows({ ...commonPrepareRunsGridDataParams, runsPinned }),
      { initialProps: { runsPinned: [] as string[] } },
    );

    const firstResult = result.current;
    expect(firstResult.every((row) => !row.pinned)).toBe(true);

    // Rerender with different runsPinned should recalculate
    rerender({ runsPinned: ['run1_1'] as string[] });
    const secondResult = result.current;

    expect(secondResult).not.toBe(firstResult);
    expect(secondResult.some((row) => row.pinned)).toBe(true);
  });

  test('it handles nested children correctly', () => {
    const { result } = renderHook(() =>
      useExperimentRunRows({
        ...commonPrepareRunsGridDataParams,
        nestChildren: true,
        runsExpanded: { run1_1: true },
      }),
    );

    expect(result.current).toHaveLength(5); // As tested in previous nested tests
    expect(result.current[0].runDateAndNestInfo?.isParent).toBe(true);
    expect(result.current[0].runDateAndNestInfo?.expanderOpen).toBe(true);
  });
});
