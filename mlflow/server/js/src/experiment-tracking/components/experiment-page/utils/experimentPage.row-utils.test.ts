import {
  shouldUseNewRunRowsVisibilityModel,
  shouldEnableToggleIndividualRunsInGroups,
  shouldUseRunRowsVisibilityMap,
} from '../../../../common/utils/FeatureUtils';
import { fromPairs } from 'lodash';
import Utils from '../../../../common/utils/Utils';
import { RUNS_VISIBILITY_MODE } from '../models/ExperimentPageUIState';
import { RunGroupingAggregateFunction, RunGroupingMode, RunRowVisibilityControl } from './experimentPage.row-types';
import { SingleRunData, prepareRunsGridData } from './experimentPage.row-utils';

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual('../../../../common/utils/FeatureUtils'),
  shouldUseNewRunRowsVisibilityModel: jest.fn().mockImplementation(() => false),
  shouldUseRunRowsVisibilityMap: jest.fn().mockImplementation(() => false),
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
      jest.mocked(shouldUseNewRunRowsVisibilityModel).mockImplementation(() => false);
      jest.mocked(shouldEnableToggleIndividualRunsInGroups).mockImplementation(() => individualRunsInGroupsFlagValue);
    });

    test('it creates proper row set for runs grouped by a tag', () => {
      const runsGridData = prepareRunsGridData({
        ...commonPrepareRunsGridDataParams,
        groupBy: {
          aggregateFunction: RunGroupingAggregateFunction.Min,
          groupByKeys: [
            {
              mode: RunGroupingMode.Tag,
              groupByData: 'testtag1',
            },
          ],
        },
      });

      // We expect 7 rows - 4 groups and 3 runs. Ungrouped runs are hidden by default.
      expect(runsGridData).toHaveLength(7);

      // We expect first group to be expanded by default
      expect(runsGridData[0].groupParentInfo).toEqual(
        expect.objectContaining({
          aggregateFunction: 'min',
          expanderOpen: true,
          groupId: 'tag.testtag1.testval1',
          groupingValues: [
            {
              groupByData: 'testtag1',
              mode: 'tag',
              value: 'testval1',
            },
          ],
          isRemainingRunsGroup: false,
          aggregatedMetricData: {
            met1: {
              key: 'met1',
              maxStep: 0,
              value: 111.123456789,
            },
          },
          aggregatedParamData: {
            p1: {
              key: 'p1',
              maxStep: 0,
              value: 123,
            },
          },
          runUuids: ['run1_1'],
        }),
      );
      // Next, we expect the first run to be a child of the first group
      expect(runsGridData[1].groupParentInfo).toBeUndefined();
      expect(runsGridData[1].runUuid).toBe('run1_1');

      // Similar for 2nd and 3rd group
      expect(runsGridData[2].groupParentInfo).toEqual(
        expect.objectContaining({
          groupId: 'tag.testtag1.testval2',
          aggregatedMetricData: {
            met1: {
              key: 'met1',
              maxStep: 0,
              value: 222,
            },
          },
          runUuids: ['run1_2'],
        }),
      );
      expect(runsGridData[3].groupParentInfo).toBeUndefined();
      expect(runsGridData[3].runUuid).toBe('run1_2');
      expect(runsGridData[4].groupParentInfo).toEqual(
        expect.objectContaining({
          groupId: 'tag.testtag1.testval3',
          runUuids: ['run1_3'],
        }),
      );
      expect(runsGridData[5].groupParentInfo).toBeUndefined();
      expect(runsGridData[5].runUuid).toBe('run1_3');

      // In the end, we expect the last group with ungrouped runs to be collapsed by default
      expect(runsGridData[6].groupParentInfo).toEqual(
        expect.objectContaining({
          groupId: 'tag.testtag1',
          isRemainingRunsGroup: true,
          expanderOpen: false,
          runUuids: ['run1_4', 'run2_1', 'run2_2'],
        }),
      );
    });

    test('it creates proper row dataset for runs grouped by a tag when expanded configuration is provided', () => {
      const runsGridData = prepareRunsGridData({
        ...commonPrepareRunsGridDataParams,
        groupBy: {
          aggregateFunction: RunGroupingAggregateFunction.Min,
          groupByKeys: [
            {
              mode: RunGroupingMode.Tag,
              groupByData: 'testtag1',
            },
          ],
        },
        // Contract all groups but expand "remaining runs" group
        groupsExpanded: {
          'tag.testtag1.testval1': false,
          'tag.testtag1.testval2': false,
          'tag.testtag1.testval3': false,
          'tag.testtag1': true,
        },
      });
      expect(runsGridData).toHaveLength(7);
      expect(runsGridData[0].groupParentInfo?.expanderOpen).toBe(false);
      expect(runsGridData[1].groupParentInfo?.expanderOpen).toBe(false);
      expect(runsGridData[2].groupParentInfo?.expanderOpen).toBe(false);
      expect(runsGridData[3].groupParentInfo?.expanderOpen).toBe(true);
      expect(runsGridData[4].runUuid).toEqual('run1_4');
      expect(runsGridData[5].runUuid).toEqual('run2_1');
      expect(runsGridData[6].runUuid).toEqual('run2_2');
    });

    test('it properly hoists pinned runs within a certain group', () => {
      const runsGridData = prepareRunsGridData({
        ...commonPrepareRunsGridDataParams,
        groupBy: {
          aggregateFunction: RunGroupingAggregateFunction.Min,
          groupByKeys: [
            {
              mode: RunGroupingMode.Tag,
              groupByData: 'testtag1',
            },
          ],
        },
        groupsExpanded: {
          'tag.testtag1.testval1': false,
          'tag.testtag1.testval2': false,
          'tag.testtag1.testval3': false,
          'tag.testtag1': true,
        },
        // Pin a single run
        runsPinned: ['run2_2'],
      });
      expect(runsGridData).toHaveLength(7);
      expect(runsGridData[3].groupParentInfo?.expanderOpen).toBe(true);
      // Expect pinned run to be hoisted to the top of the group
      expect(runsGridData[4].runUuid).toEqual('run2_2');
      expect(runsGridData[5].runUuid).toEqual('run1_4');
      expect(runsGridData[6].runUuid).toEqual('run2_1');
    });

    test('it properly hoists pinned group runs', () => {
      const runsGridData = prepareRunsGridData({
        ...commonPrepareRunsGridDataParams,
        groupBy: {
          aggregateFunction: RunGroupingAggregateFunction.Min,
          groupByKeys: [
            {
              mode: RunGroupingMode.Tag,
              groupByData: 'testtag1',
            },
          ],
        },
        groupsExpanded: {
          'tag.testtag1.testval1': false,
          'tag.testtag1.testval2': false,
          'tag.testtag1.testval3': false,
          'tag.testtag1': false,
        },
        // Pin two run groups
        runsPinned: ['tag.testtag1.testval2', 'tag.testtag1'],
      });
      expect(runsGridData).toHaveLength(4);
      // Expect pinned groups to be hoisted to the top of the list
      expect(runsGridData[0].rowUuid).toEqual('tag.testtag1.testval2');
      expect(runsGridData[1].rowUuid).toEqual('tag.testtag1');

      // Expect not pinned groups to be at the bottom
      expect(runsGridData[2].rowUuid).toEqual('tag.testtag1.testval1');
      expect(runsGridData[3].rowUuid).toEqual('tag.testtag1.testval3');
    });

    describe.each([
      ['when using runsVisibilityMap UI state', true],
      ['when using legacy runsHidden UI state', false],
    ])('Configurable runs visibility mode  %s', (_, useExplicitRunRowsVisibility) => {
      beforeEach(() => {
        jest.mocked(shouldUseNewRunRowsVisibilityModel).mockImplementation(() => true);
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
    });
  },
);

describe('ExperimentViewRuns row utils, grouped run hierarchy - selecting individual runs', () => {
  beforeEach(() => {
    jest.mocked(shouldUseNewRunRowsVisibilityModel).mockImplementation(() => true);
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
