import Utils from '../../../../common/utils/Utils';
import { prepareRunsGridData } from './experimentPage.row-utils';

const LOGGED_MODEL = { LOGGED_MODEL: true };

// Do not test tag->logged model transformation
Utils.getLoggedModelsFromTags = jest.fn().mockReturnValue([LOGGED_MODEL]);

const MOCK_EXPERIMENTS = [
  {
    experiment_id: '1',
    name: '/Users/john.doe@databricks.com/test-experiment-1',
  },
  {
    experiment_id: '2',
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
    runInfo: { experiment_id: '1', run_uuid: 'run1_1' },
    metrics: [{ key: 'met1', value: '111' }],
    params: [{ key: 'p1', value: '123' }],
    tags: { testtag1: { key: 'testtag1', value: 'testval1' } },
  },
  {
    runInfo: { experiment_id: '1', run_uuid: 'run1_2' },
    metrics: [{ key: 'met1', value: '222' }],
    tags: {
      testtag1: { key: 'testtag1', value: 'testval2' },
      'mlflow.parentRunId': { key: 'mlflow.parentRunId', value: 'run1_1' },
    },
  },
  {
    runInfo: { experiment_id: '1', run_uuid: 'run1_3' },
    tags: {
      testtag1: { key: 'testtag1', value: 'testval3' },
      'mlflow.parentRunId': { key: 'mlflow.parentRunId', value: 'run1_1' },
    },
  },
  {
    runInfo: { experiment_id: '1', run_uuid: 'run1_4' },
    metrics: [
      { key: 'met1', value: '1122' },
      { key: 'met2', value: '2211' },
    ],
    params: [
      { key: 'p1', value: '1234' },
      { key: 'p2', value: '12345' },
    ],
    tags: { 'mlflow.parentRunId': { key: 'mlflow.parentRunId', value: 'run1_2' } },
  },
  {
    runInfo: { experiment_id: '2', run_uuid: 'run2_1' },
  },
  {
    runInfo: { experiment_id: '2', run_uuid: 'run2_2' },
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
  runUuidsMatchingFilter: MOCK_RUN_DATA.map((r) => r.runInfo.run_uuid),
};

describe('ExperimentViewRuns row utils', () => {
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
    expect(runsGridData.map((runData) => runData.runInfo)).toEqual(
      MOCK_RUN_DATA.map((d) => d.runInfo),
    );

    // Assert model for the second run (index #1)
    expect(runsGridData[1].models.registeredModels).toEqual(
      expect.arrayContaining([expect.objectContaining(MOCK_MODEL_MAP.run1_2[0])]),
    );
    expect(runsGridData[1].models.loggedModels).toEqual(expect.arrayContaining([LOGGED_MODEL]));

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
    expect(runsGridData[0]['$$$metric$$$-met1']).toEqual('111');
    expect(runsGridData[0]['$$$metric$$$-met2']).toEqual('-');
    expect(runsGridData[0]['$$$tag$$$-testtag1']).toEqual('testval1');

    // Assert run #2 KV data
    expect(runsGridData[1]['$$$param$$$-p1']).toEqual('-');
    expect(runsGridData[1]['$$$param$$$-p2']).toEqual('-');
    expect(runsGridData[1]['$$$metric$$$-met1']).toEqual('222');
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
    expect(runsGridData[3]['$$$metric$$$-met1']).toEqual('1122');
    expect(runsGridData[3]['$$$metric$$$-met2']).toEqual('2211');
    expect(runsGridData[3]['$$$tag$$$-testtag1']).toEqual('-');

    // Assert run #5 KV data
    expect(runsGridData[4]['$$$param$$$-p1']).toEqual('-');
    expect(runsGridData[4]['$$$param$$$-p2']).toEqual('-');
    expect(runsGridData[4]['$$$metric$$$-met1']).toEqual('-');
    expect(runsGridData[4]['$$$metric$$$-met2']).toEqual('-');
    expect(runsGridData[4]['$$$tag$$$-testtag1']).toEqual('-');
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
    expect(runsGridData[0].runDateAndNestInfo.isParent).toEqual(true);
    expect(runsGridData[0].runDateAndNestInfo.hasExpander).toEqual(true);
    expect(runsGridData[0].runDateAndNestInfo.childrenIds).toEqual(['run1_2', 'run1_3']);
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
    expect(runsGridData[0].runDateAndNestInfo.isParent).toEqual(true);
    expect(runsGridData[0].runDateAndNestInfo.hasExpander).toEqual(true);
    expect(runsGridData[0].runDateAndNestInfo.expanderOpen).toEqual(true);
    expect(runsGridData[0].runDateAndNestInfo.childrenIds).toEqual(['run1_2', 'run1_3']);

    expect(runsGridData[1].runDateAndNestInfo.isParent).toEqual(true);
    expect(runsGridData[1].runDateAndNestInfo.hasExpander).toEqual(true);
    expect(runsGridData[1].runDateAndNestInfo.expanderOpen).toEqual(false);
    expect(runsGridData[1].runDateAndNestInfo.childrenIds).toEqual(['run1_4']);

    expect(runsGridData[2].runDateAndNestInfo.isParent).toEqual(false);
    expect(runsGridData[2].runDateAndNestInfo.hasExpander).toEqual(false);
    expect(runsGridData[2].runDateAndNestInfo.expanderOpen).toEqual(false);
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
    expect(runsGridData[0].runDateAndNestInfo.isParent).toEqual(true);
    expect(runsGridData[0].runDateAndNestInfo.hasExpander).toEqual(true);
    expect(runsGridData[0].runDateAndNestInfo.expanderOpen).toEqual(true);
    expect(runsGridData[0].runDateAndNestInfo.childrenIds).toEqual(['run1_2', 'run1_3']);

    expect(runsGridData[1].runDateAndNestInfo.isParent).toEqual(true);
    expect(runsGridData[1].runDateAndNestInfo.hasExpander).toEqual(true);
    expect(runsGridData[1].runDateAndNestInfo.expanderOpen).toEqual(true);
    expect(runsGridData[1].runDateAndNestInfo.childrenIds).toEqual(['run1_4']);
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
          runInfo: { experiment_id: '1', run_uuid: 'run1_1' },
        },
        {
          runInfo: { experiment_id: '1', run_uuid: 'run1_2' },
          tags: {
            'mlflow.parentRunId': { key: 'mlflow.parentRunId', value: 'run1_4' },
          },
        },
        {
          runInfo: { experiment_id: '1', run_uuid: 'run1_3' },
          tags: {
            'mlflow.parentRunId': { key: 'mlflow.parentRunId', value: 'run1_2' },
          },
        },
        {
          runInfo: { experiment_id: '1', run_uuid: 'run1_4' },
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
          runInfo: { experiment_id: '1', run_uuid: 'run1_1' },
        },
        {
          runInfo: { experiment_id: '1', run_uuid: 'run1_2' },
          tags: {
            'mlflow.parentRunId': { key: 'mlflow.parentRunId', value: 'run1_1' },
          },
        },
        {
          runInfo: { experiment_id: '1', run_uuid: 'run1_3' },
        },
        {
          runInfo: { experiment_id: '1', run_uuid: 'run1_4' },
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
      runsGridData.map(({ runUuid, pinned, runDateAndNestInfo: { isParent } }) => ({
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
          runInfo: { experiment_id: '1', run_uuid: 'run1_1' },
        },
        {
          runInfo: { experiment_id: '1', run_uuid: 'run1_2' },
        },
        {
          runInfo: { experiment_id: '1', run_uuid: 'run1_3' },
        },
        {
          runInfo: { experiment_id: '1', run_uuid: 'run1_4' },
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
