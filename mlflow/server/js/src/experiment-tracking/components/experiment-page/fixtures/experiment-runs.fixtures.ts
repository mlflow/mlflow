import type { ExperimentStoreEntities } from '../../../types';

/**
 * Sample snapshot of the store with experiment and runs.
 *
 * Three experiments:
 * - "123456789" with 5 runs:
 *   - runs 1-4 are active
 *   - run 5 is deleted
 *   - runs 1-4 have metrics "met1", "met2" and "met3"
 *   - runs 1-3 have tags "testtag1" and "testtag2"
 *   - run 4 have tags "testtag1", "testtag2" and "testag3"
 *   - all runs have params "p1", "p2" and "p3"
 * - "654321" with one run:
 *   - active state, one metric "met1", no params
 * - "789" without runs
 * - "3210" with one run:
 *   - active state
 *   - metrics "met1" and ""
 *   - tags "testtag1" and "\t"
 *   - params "p1" and "\n"
 */
export const EXPERIMENT_RUNS_MOCK_STORE: { entities: ExperimentStoreEntities } = {
  entities: {
    artifactRootUriByRunUuid: {},
    runInputsOutputsByUuid: {},
    artifactsByRunUuid: {},
    sampledMetricsByRunUuid: {},
    modelByName: {},
    colorByRunUuid: {},
    runUuidsMatchingFilter: [],
    runDatasetsByUuid: {
      experiment123456789_run1: [
        {
          dataset: {
            digest: 'abc',
            name: 'dataset_train',
            profile: '{}',
            schema: '{}',
            source: '{}',
            sourceType: 'local',
          },
          tags: [{ key: 'mlflow.data.context', value: 'training' } as any],
        },
        {
          dataset: {
            digest: '123',
            name: 'dataset_eval',
            profile: '{}',
            schema: '{}',
            source: '{}',
            sourceType: 'local',
          },
          tags: [{ key: 'mlflow.data.context', value: 'eval' } as any],
        },
      ],
    },
    experimentsById: {
      '123456789': {
        experimentId: '123456789',
        name: '/Users/john.doe@databricks.com/test-experiment',
        artifactLocation: 'dbfs:/databricks/mlflow-tracking/123456789',
        lifecycleStage: 'active',
        lastUpdateTime: 1654502190803,
        creationTime: 1654502190803,
        tags: [
          { key: 'mlflow.ownerId', value: '987654321' },
          {
            key: 'mlflow.experiment.sourceName',
            value: '/Users/john.doe@databricks.com/test-experiment',
          },
          { key: 'mlflow.ownerId', value: '987654321' },
          { key: 'mlflow.ownerEmail', value: 'john.doe@databricks.com' },
          { key: 'mlflow.experimentType', value: 'NOTEBOOK' },
        ],
        allowedActions: ['MODIFIY_PERMISSION', 'DELETE', 'RENAME'],
      },
      '654321': {
        experimentId: '654321',
        name: '/Users/john.doe@databricks.com/test-experiment',
        artifactLocation: 'dbfs:/databricks/mlflow-tracking/654321',
        lifecycleStage: 'active',
        lastUpdateTime: 1654502190603,
        creationTime: 1654502190603,
        tags: [],
        allowedActions: ['MODIFIY_PERMISSION', 'DELETE', 'RENAME'],
      },
      '789': {
        experimentId: '789',
        name: '/Users/john.doe@databricks.com/test-experiment',
        artifactLocation: 'dbfs:/databricks/mlflow-tracking/789',
        lifecycleStage: 'active',
        lastUpdateTime: 1000502190603,
        creationTime: 1000502190603,
        tags: [],
        allowedActions: ['MODIFIY_PERMISSION', 'DELETE', 'RENAME'],
      },
      '3210': {
        experimentId: '3210',
        name: '/Users/john.doe@databricks.com/test-experiment',
        artifactLocation: 'dbfs:/databricks/mlflow-tracking/3210',
        lifecycleStage: 'active',
        lastUpdateTime: 1000502190604,
        creationTime: 1000502190604,
        tags: [],
        allowedActions: ['MODIFIY_PERMISSION', 'DELETE', 'RENAME'],
      },
    },
    runInfosByUuid: {
      experiment123456789_run1: {
        runUuid: 'experiment123456789_run1',
        runName: 'experiment123456789_run1',
        experimentId: '123456789',
        status: 'FINISHED',
        startTime: 1660116336860,
        endTime: 1660116337489,
        artifactUri: 'dbfs:/databricks/mlflow-tracking/123456789/experiment123456789_run1/artifacts',
        lifecycleStage: 'active',
      },
      experiment123456789_run2: {
        runUuid: 'experiment123456789_run2',
        runName: 'experiment123456789_run2',
        experimentId: '123456789',
        status: 'FINISHED',
        startTime: 1660116265829,
        endTime: 1660116266518,
        artifactUri: 'dbfs:/databricks/mlflow-tracking/123456789/experiment123456789_run2/artifacts',
        lifecycleStage: 'active',
      },
      experiment123456789_run3: {
        runUuid: 'experiment123456789_run3',
        runName: 'experiment123456789_run3',
        experimentId: '123456789',
        status: 'FINISHED',
        startTime: 1660116197855,
        endTime: 1660116198498,
        artifactUri: 'dbfs:/databricks/mlflow-tracking/123456789/experiment123456789_run3/artifacts',
        lifecycleStage: 'active',
      },
      experiment123456789_run4: {
        runUuid: 'experiment123456789_run4',
        runName: 'experiment123456789_run4',
        experimentId: '123456789',
        status: 'FINISHED',
        startTime: 1660116194167,
        endTime: 1660116194802,
        artifactUri: 'dbfs:/databricks/mlflow-tracking/123456789/experiment123456789_run4/artifacts',
        lifecycleStage: 'active',
      },
      experiment123456789_run5: {
        runUuid: 'experiment123456789_run5',
        runName: 'experiment123456789_run5',
        experimentId: '123456789',
        status: 'FINISHED',
        startTime: 1660116194167,
        endTime: 1660116194802,
        artifactUri: 'dbfs:/databricks/mlflow-tracking/123456789/experiment123456789_run5/artifacts',
        lifecycleStage: 'deleted',
      },
      experiment654321_run1: {
        runUuid: 'experiment654321_run1',
        runName: 'experiment654321_run1',
        experimentId: '654321',
        status: 'FINISHED',
        startTime: 1660116161320,
        endTime: 1660116194039,
        artifactUri: 'dbfs:/databricks/mlflow-tracking/654321/experiment654321_run1/artifacts',
        lifecycleStage: 'active',
      },
      experiment3210_run1: {
        runUuid: 'experiment3210_run1',
        runName: 'experiment3210_run1',
        experimentId: '3210',
        status: 'FINISHED',
        startTime: 1660116161321,
        endTime: 1660116194042,
        artifactUri: 'dbfs:/databricks/mlflow-tracking/3210/experiment3210_run1/artifacts',
        lifecycleStage: 'active',
      },
    },
    runInfoOrderByUuid: [
      'experiment123456789_run1',
      'experiment123456789_run2',
      'experiment123456789_run3',
      'experiment123456789_run4',
      'experiment123456789_run5',
      'experiment654321_run1',
      'experiment3210_run1',
    ],
    metricsByRunUuid: {},
    imagesByRunUuid: {},
    latestMetricsByRunUuid: {
      experiment123456789_run1: {
        met1: {
          key: 'met1',
          value: 255,
          timestamp: 1000,
          step: 0,
        },
        met2: {
          key: 'met2',
          value: 180,
          timestamp: 1000,
          step: 0,
        },
        met3: {
          key: 'met3',
          value: 333,
          timestamp: 1000,
          step: 0,
        },
      },
      experiment123456789_run2: {
        met1: {
          key: 'met1',
          value: 55,
          timestamp: 1000,
          step: 0,
        },
        met2: {
          key: 'met2',
          value: 80,
          timestamp: 1000,
          step: 0,
        },
        met3: {
          key: 'met3',
          value: 133,
          timestamp: 1000,
          step: 0,
        },
      },
      experiment123456789_run3: {
        met1: {
          key: 'met1',
          value: 5,
          timestamp: 1000,
          step: 0,
        },
        met2: {
          key: 'met2',
          value: 10,
          timestamp: 1000,
          step: 0,
        },
        met3: {
          key: 'met3',
          value: 33,
          timestamp: 1000,
          step: 0,
        },
      },
      experiment123456789_run4: {
        met1: {
          key: 'met1',
          value: 5,
          timestamp: 1000,
          step: 0,
        },
        met2: {
          key: 'met2',
          value: 10,
          timestamp: 1000,
          step: 0,
        },
        met3: {
          key: 'met3',
          value: 33,
          timestamp: 1000,
          step: 0,
        },
      },
      experiment654321_run1: {
        met1: {
          key: 'met1',
          value: 5,
          timestamp: 1000,
          step: 0,
        },
      },
      experiment3210_run1: {
        met1: {
          key: 'met1',
          value: 2,
          timestamp: 1000,
          step: 0,
        },
        '': {
          key: '',
          value: 0,
          timestamp: 1000,
          step: 0,
        },
      },
    },
    minMetricsByRunUuid: {},
    maxMetricsByRunUuid: {},
    paramsByRunUuid: {
      experiment123456789_run1: {
        p1: {
          key: 'p1',
          value: '12',
        },
        p2: {
          key: 'p2',
          value: '17',
        },
        p3: {
          key: 'p3',
          value: '57',
        },
      },
      experiment123456789_run2: {
        p1: {
          key: 'p1',
          value: '11',
        },
        p2: {
          key: 'p2',
          value: '16',
        },
        p3: {
          key: 'p3',
          value: '56',
        },
      },
      experiment123456789_run3: {
        p1: {
          key: 'p1',
          value: '10',
        },
        p2: {
          key: 'p2',
          value: '15',
        },
        p3: {
          key: 'p3',
          value: '55',
        },
      },
      experiment123456789_run4: {
        p1: {
          key: 'p1',
          value: '10',
        },
        p2: {
          key: 'p2',
          value: '15',
        },
        p3: {
          key: 'p3',
          value: '55',
        },
      },
      experiment123456789_run5: {
        p1: {
          key: 'p1',
          value: '10',
        },
        p2: {
          key: 'p2',
          value: '15',
        },
        p3: {
          key: 'p3',
          value: '55',
        },
      },
      experiment654321_run1: {},
      experiment3210_run1: {
        p1: {
          key: 'p1',
          value: '',
        },
        '\n': {
          key: '\n',
          value: '0',
        },
      },
    },
    tagsByRunUuid: {
      experiment123456789_run1: {
        testtag1: {
          key: 'testtag1',
          value: 'value1',
        },
        testtag2: {
          key: 'testtag2',
          value: 'value2',
        },
      },
      experiment123456789_run2: {
        testtag1: {
          key: 'testtag1',
          value: 'value1_2',
        },
        testtag2: {
          key: 'testtag2',
          value: 'value2_2',
        },
      },
      experiment123456789_run3: {
        testtag1: {
          key: 'testtag1',
          value: 'value1_3',
        },
        testtag2: {
          key: 'testtag2',
          value: 'value2_3',
        },
      },
      experiment123456789_run4: {
        testtag1: {
          key: 'testtag1',
          value: 'value1_4',
        },
        testtag2: {
          key: 'testtag2',
          value: 'value2_4',
        },
        testtag3: {
          key: 'testtag3',
          value: 'value3',
        },
        'tag with a space': {
          key: 'tag with a space',
          value: 'value3',
        },
      },
      experiment654321_run1: {
        testtag1: {
          key: 'testtag1',
          value: 'value1_5',
        },
        testtag2: {
          key: 'testtag2',
          value: 'value2_5',
        },
      },
      experiment3210_run1: {
        testtag1: {
          key: 'testtag1',
          value: '',
        },
        '\t': {
          key: '\t',
          value: 'value1',
        },
      },
    },
    experimentTagsByExperimentId: {
      '123456789': {
        'mlflow.ownerId': {
          key: 'mlflow.ownerId',
          value: '987654321',
        },
        'mlflow.experiment.sourceName': {
          key: 'mlflow.experiment.sourceName',
          value: '/Users/john.doe@databricks.com/test-experiment',
        },
        'mlflow.ownerEmail': {
          key: 'mlflow.ownerEmail',
          value: 'john.doe@databricks.com',
        },
        'mlflow.experimentType': {
          key: 'mlflow.experimentType',
          value: 'NOTEBOOK',
        },
      },
    },
    modelVersionsByRunUuid: {
      experiment123456789_run4: [
        {
          name: 'test_model',
          creation_timestamp: 1234,
          current_stage: '',
          last_updated_timestamp: 1234,
          run_id: 'experiment123456789_run4',
          source: 'notebook',
          status: 'active',
          user_id: '123',
          version: '1',
        },
      ],
    },
    modelVersionsByModel: {},
    datasetsByExperimentId: {
      123456789: [
        {
          experiment_id: '123456789',
          name: 'dataset_train',
          digest: 'abc',
        },
        {
          experiment_id: '123456789',
          name: 'dataset_eval',
          digest: '123',
        },
      ],
    },
  },
};

/**
 * Object mapping runUuids to mock metric history for each run.
 * This is used to test generating aggregate history for run groups,
 * for example in `createValueAggregatedMetricHistory()`.
 *
 * The "base" metric for run 1 is [5,4,3,2,1] and [0,1,2,3,5] for run 2.
 * The "metric" metric is `base - 2` for run 1, and `base + 2` for run 2.
 */

export const MOCK_RUN_UUIDS_TO_HISTORY_MAP = {
  '11ab92332f8c4ed28cac10fbfb8e0ecc': {
    runUuid: '11ab92332f8c4ed28cac10fbfb8e0ecc',
    metric: {
      metricsHistory: [
        {
          key: 'metric',
          value: 3,
          timestamp: 1706499312509,
          step: 0,
          run_id: '11ab92332f8c4ed28cac10fbfb8e0ecc',
        },
        {
          key: 'metric',
          value: 2,
          timestamp: 1706499312755,
          step: 1,
          run_id: '11ab92332f8c4ed28cac10fbfb8e0ecc',
        },
        {
          key: 'metric',
          value: 1,
          timestamp: 1706499312952,
          step: 2,
          run_id: '11ab92332f8c4ed28cac10fbfb8e0ecc',
        },
        {
          key: 'metric',
          value: 0,
          timestamp: 1706499313139,
          step: 3,
          run_id: '11ab92332f8c4ed28cac10fbfb8e0ecc',
        },
        {
          key: 'metric',
          value: -1,
          timestamp: 1706499313326,
          step: 4,
          run_id: '11ab92332f8c4ed28cac10fbfb8e0ecc',
        },
      ],
      loading: false,
      refreshing: false,
    },
    base: {
      metricsHistory: [
        {
          key: 'base',
          value: 5,
          timestamp: 1706499312374,
          step: 0,
          run_id: '11ab92332f8c4ed28cac10fbfb8e0ecc',
        },
        {
          key: 'base',
          value: 4,
          timestamp: 1706499312614,
          step: 1,
          run_id: '11ab92332f8c4ed28cac10fbfb8e0ecc',
        },
        {
          key: 'base',
          value: 3,
          timestamp: 1706499312863,
          step: 2,
          run_id: '11ab92332f8c4ed28cac10fbfb8e0ecc',
        },
        {
          key: 'base',
          value: 2,
          timestamp: 1706499313042,
          step: 3,
          run_id: '11ab92332f8c4ed28cac10fbfb8e0ecc',
        },
        {
          key: 'base',
          value: 1,
          timestamp: 1706499313238,
          step: 4,
          run_id: '11ab92332f8c4ed28cac10fbfb8e0ecc',
        },
      ],
      loading: false,
      refreshing: false,
    },
  },
  a9b89d3b2bf54d9ba8ae539dbffa9a4c: {
    runUuid: 'a9b89d3b2bf54d9ba8ae539dbffa9a4c',
    metric: {
      metricsHistory: [
        {
          key: 'metric',
          value: 3,
          timestamp: 1706499310846,
          step: 0,
          run_id: 'a9b89d3b2bf54d9ba8ae539dbffa9a4c',
        },
        {
          key: 'metric',
          value: 4,
          timestamp: 1706499311022,
          step: 1,
          run_id: 'a9b89d3b2bf54d9ba8ae539dbffa9a4c',
        },
        {
          key: 'metric',
          value: 5,
          timestamp: 1706499311240,
          step: 2,
          run_id: 'a9b89d3b2bf54d9ba8ae539dbffa9a4c',
        },
        {
          key: 'metric',
          value: 6,
          timestamp: 1706499311435,
          step: 3,
          run_id: 'a9b89d3b2bf54d9ba8ae539dbffa9a4c',
        },
        {
          key: 'metric',
          value: 7,
          timestamp: 1706499311609,
          step: 4,
          run_id: 'a9b89d3b2bf54d9ba8ae539dbffa9a4c',
        },
      ],
      loading: false,
      refreshing: false,
    },
    base: {
      metricsHistory: [
        {
          key: 'base',
          value: 1,
          timestamp: 1706499310738,
          step: 0,
          run_id: 'a9b89d3b2bf54d9ba8ae539dbffa9a4c',
        },
        {
          key: 'base',
          value: 2,
          timestamp: 1706499310937,
          step: 1,
          run_id: 'a9b89d3b2bf54d9ba8ae539dbffa9a4c',
        },
        {
          key: 'base',
          value: 3,
          timestamp: 1706499311144,
          step: 2,
          run_id: 'a9b89d3b2bf54d9ba8ae539dbffa9a4c',
        },
        {
          key: 'base',
          value: 4,
          timestamp: 1706499311337,
          step: 3,
          run_id: 'a9b89d3b2bf54d9ba8ae539dbffa9a4c',
        },
        {
          key: 'base',
          value: 5,
          timestamp: 1706499311522,
          step: 4,
          run_id: 'a9b89d3b2bf54d9ba8ae539dbffa9a4c',
        },
      ],
      loading: false,
      refreshing: false,
    },
  },
};
