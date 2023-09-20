import {
  Experiment,
  ExperimentTag,
  Metric,
  Param,
  RunInfo,
  RunTag,
} from '../../../sdk/MlflowMessages';
import { ExperimentStoreEntities } from '../../../types';
import { hydrateImmutableRecord } from './fixture.utils';

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
 */
export const EXPERIMENT_RUNS_MOCK_STORE: { entities: ExperimentStoreEntities } = {
  entities: {
    modelByName: {},
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
            source_type: 'local',
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
            source_type: 'local',
          },
          tags: [{ key: 'mlflow.data.context', value: 'eval' } as any],
        },
      ],
    },
    experimentsById: {
      '123456789': hydrateImmutableRecord(Experiment)({
        experiment_id: '123456789',
        name: '/Users/john.doe@databricks.com/test-experiment',
        artifact_location: 'dbfs:/databricks/mlflow-tracking/123456789',
        lifecycle_stage: 'active',
        last_update_time: 1654502190803,
        creation_time: 1654502190803,
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
        allowed_actions: ['MODIFIY_PERMISSION', 'DELETE', 'RENAME'],
      }),
      '654321': hydrateImmutableRecord(Experiment)({
        experiment_id: '654321',
        name: '/Users/john.doe@databricks.com/test-experiment',
        artifact_location: 'dbfs:/databricks/mlflow-tracking/654321',
        lifecycle_stage: 'active',
        last_update_time: 1654502190603,
        creation_time: 1654502190603,
        tags: [],
        allowed_actions: ['MODIFIY_PERMISSION', 'DELETE', 'RENAME'],
      }),
      '789': hydrateImmutableRecord(Experiment)({
        experiment_id: '789',
        name: '/Users/john.doe@databricks.com/test-experiment',
        artifact_location: 'dbfs:/databricks/mlflow-tracking/789',
        lifecycle_stage: 'active',
        last_update_time: 1000502190603,
        creation_time: 1000502190603,
        tags: [],
        allowed_actions: ['MODIFIY_PERMISSION', 'DELETE', 'RENAME'],
      }),
    },
    runInfosByUuid: {
      experiment123456789_run1: hydrateImmutableRecord(RunInfo)({
        run_uuid: 'experiment123456789_run1',
        run_name: 'experiment123456789_run1',
        experiment_id: '123456789',
        status: 'FINISHED',
        start_time: 1660116336860,
        end_time: 1660116337489,
        artifact_uri:
          'dbfs:/databricks/mlflow-tracking/123456789/experiment123456789_run1/artifacts',
        lifecycle_stage: 'active',
      }),
      experiment123456789_run2: hydrateImmutableRecord(RunInfo)({
        run_uuid: 'experiment123456789_run2',
        run_name: 'experiment123456789_run2',
        experiment_id: '123456789',
        status: 'FINISHED',
        start_time: 1660116265829,
        end_time: 1660116266518,
        artifact_uri:
          'dbfs:/databricks/mlflow-tracking/123456789/experiment123456789_run2/artifacts',
        lifecycle_stage: 'active',
      }),
      experiment123456789_run3: hydrateImmutableRecord(RunInfo)({
        run_uuid: 'experiment123456789_run3',
        run_name: 'experiment123456789_run3',
        experiment_id: '123456789',
        status: 'FINISHED',
        start_time: 1660116197855,
        end_time: 1660116198498,
        artifact_uri:
          'dbfs:/databricks/mlflow-tracking/123456789/experiment123456789_run3/artifacts',
        lifecycle_stage: 'active',
      }),
      experiment123456789_run4: hydrateImmutableRecord(RunInfo)({
        run_uuid: 'experiment123456789_run4',
        run_name: 'experiment123456789_run4',
        experiment_id: '123456789',
        status: 'FINISHED',
        start_time: 1660116194167,
        end_time: 1660116194802,
        artifact_uri:
          'dbfs:/databricks/mlflow-tracking/123456789/experiment123456789_run4/artifacts',
        lifecycle_stage: 'active',
      }),
      experiment123456789_run5: hydrateImmutableRecord(RunInfo)({
        run_uuid: 'experiment123456789_run5',
        run_name: 'experiment123456789_run5',
        experiment_id: '123456789',
        status: 'FINISHED',
        start_time: 1660116194167,
        end_time: 1660116194802,
        artifact_uri:
          'dbfs:/databricks/mlflow-tracking/123456789/experiment123456789_run5/artifacts',
        lifecycle_stage: 'deleted',
      }),
      experiment654321_run1: hydrateImmutableRecord(RunInfo)({
        run_uuid: 'experiment654321_run1',
        run_name: 'experiment654321_run1',
        experiment_id: '654321',
        status: 'FINISHED',
        start_time: 1660116161320,
        end_time: 1660116194039,
        artifact_uri: 'dbfs:/databricks/mlflow-tracking/654321/experiment654321_run1/artifacts',
        lifecycle_stage: 'active',
      }),
    },
    metricsByRunUuid: {},
    latestMetricsByRunUuid: {
      experiment123456789_run1: {
        met1: hydrateImmutableRecord(Metric)({
          key: 'met1',
          value: 255,
          timestamp: 1000,
          step: 0,
        }),
        met2: hydrateImmutableRecord(Metric)({
          key: 'met2',
          value: 180,
          timestamp: 1000,
          step: 0,
        }),
        met3: hydrateImmutableRecord(Metric)({
          key: 'met3',
          value: 333,
          timestamp: 1000,
          step: 0,
        }),
      },
      experiment123456789_run2: {
        met1: hydrateImmutableRecord(Metric)({
          key: 'met1',
          value: 55,
          timestamp: 1000,
          step: 0,
        }),
        met2: hydrateImmutableRecord(Metric)({
          key: 'met2',
          value: 80,
          timestamp: 1000,
          step: 0,
        }),
        met3: hydrateImmutableRecord(Metric)({
          key: 'met3',
          value: 133,
          timestamp: 1000,
          step: 0,
        }),
      },
      experiment123456789_run3: {
        met1: hydrateImmutableRecord(Metric)({
          key: 'met1',
          value: 5,
          timestamp: 1000,
          step: 0,
        }),
        met2: hydrateImmutableRecord(Metric)({
          key: 'met2',
          value: 10,
          timestamp: 1000,
          step: 0,
        }),
        met3: hydrateImmutableRecord(Metric)({
          key: 'met3',
          value: 33,
          timestamp: 1000,
          step: 0,
        }),
      },
      experiment123456789_run4: {
        met1: hydrateImmutableRecord(Metric)({
          key: 'met1',
          value: 5,
          timestamp: 1000,
          step: 0,
        }),
        met2: hydrateImmutableRecord(Metric)({
          key: 'met2',
          value: 10,
          timestamp: 1000,
          step: 0,
        }),
        met3: hydrateImmutableRecord(Metric)({
          key: 'met3',
          value: 33,
          timestamp: 1000,
          step: 0,
        }),
      },
      experiment654321_run1: {
        met1: hydrateImmutableRecord(Metric)({
          key: 'met1',
          value: 5,
          timestamp: 1000,
          step: 0,
        }),
      },
    },
    minMetricsByRunUuid: {},
    maxMetricsByRunUuid: {},
    paramsByRunUuid: {
      experiment123456789_run1: {
        p1: hydrateImmutableRecord(Param)({
          key: 'p1',
          value: '12',
        }),
        p2: hydrateImmutableRecord(Param)({
          key: 'p2',
          value: '17',
        }),
        p3: hydrateImmutableRecord(Param)({
          key: 'p3',
          value: '57',
        }),
      },
      experiment123456789_run2: {
        p1: hydrateImmutableRecord(Param)({
          key: 'p1',
          value: '11',
        }),
        p2: hydrateImmutableRecord(Param)({
          key: 'p2',
          value: '16',
        }),
        p3: hydrateImmutableRecord(Param)({
          key: 'p3',
          value: '56',
        }),
      },
      experiment123456789_run3: {
        p1: hydrateImmutableRecord(Param)({
          key: 'p1',
          value: '10',
        }),
        p2: hydrateImmutableRecord(Param)({
          key: 'p2',
          value: '15',
        }),
        p3: hydrateImmutableRecord(Param)({
          key: 'p3',
          value: '55',
        }),
      },
      experiment123456789_run4: {
        p1: hydrateImmutableRecord(Param)({
          key: 'p1',
          value: '10',
        }),
        p2: hydrateImmutableRecord(Param)({
          key: 'p2',
          value: '15',
        }),
        p3: hydrateImmutableRecord(Param)({
          key: 'p3',
          value: '55',
        }),
      },
      experiment123456789_run5: {
        p1: hydrateImmutableRecord(Param)({
          key: 'p1',
          value: '10',
        }),
        p2: hydrateImmutableRecord(Param)({
          key: 'p2',
          value: '15',
        }),
        p3: hydrateImmutableRecord(Param)({
          key: 'p3',
          value: '55',
        }),
      },
      experiment654321_run1: {},
    },
    tagsByRunUuid: {
      experiment123456789_run1: {
        testtag1: hydrateImmutableRecord(RunTag)({
          key: 'testtag1',
          value: 'value1',
        }),
        testtag2: hydrateImmutableRecord(RunTag)({
          key: 'testtag2',
          value: 'value2',
        }),
      },
      experiment123456789_run2: {
        testtag1: hydrateImmutableRecord(RunTag)({
          key: 'testtag1',
          value: 'value1_2',
        }),
        testtag2: hydrateImmutableRecord(RunTag)({
          key: 'testtag2',
          value: 'value2_2',
        }),
      },
      experiment123456789_run3: {
        testtag1: hydrateImmutableRecord(RunTag)({
          key: 'testtag1',
          value: 'value1_3',
        }),
        testtag2: hydrateImmutableRecord(RunTag)({
          key: 'testtag2',
          value: 'value2_3',
        }),
      },
      experiment123456789_run4: {
        testtag1: hydrateImmutableRecord(RunTag)({
          key: 'testtag1',
          value: 'value1_4',
        }),
        testtag2: hydrateImmutableRecord(RunTag)({
          key: 'testtag2',
          value: 'value2_4',
        }),
        testtag3: hydrateImmutableRecord(RunTag)({
          key: 'testtag3',
          value: 'value3',
        }),
        'tag with a space': hydrateImmutableRecord(RunTag)({
          key: 'tag with a space',
          value: 'value3',
        }),
      },
      experiment654321_run1: {
        testtag1: hydrateImmutableRecord(RunTag)({
          key: 'testtag1',
          value: 'value1_5',
        }),
        testtag2: hydrateImmutableRecord(RunTag)({
          key: 'testtag2',
          value: 'value2_5',
        }),
      },
    },
    experimentTagsByExperimentId: {
      '123456789': {
        'mlflow.ownerId': hydrateImmutableRecord(ExperimentTag)({
          key: 'mlflow.ownerId',
          value: '987654321',
        }),
        'mlflow.experiment.sourceName': hydrateImmutableRecord(ExperimentTag)({
          key: 'mlflow.experiment.sourceName',
          value: '/Users/john.doe@databricks.com/test-experiment',
        }),
        'mlflow.ownerEmail': hydrateImmutableRecord(ExperimentTag)({
          key: 'mlflow.ownerEmail',
          value: 'john.doe@databricks.com',
        }),
        'mlflow.experimentType': hydrateImmutableRecord(ExperimentTag)({
          key: 'mlflow.experimentType',
          value: 'NOTEBOOK',
        }),
      },
    },
    modelVersionsByRunUuid: {
      experiment123456789_run4: [
        {
          name: 'test_model',
          creation_timestamp: 1234,
          current_stage: '',
          email_subscription_status: 'active',
          last_updated_timestamp: 1234,
          permission_level: '',
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
          context: 'training',
        },
        {
          experiment_id: '123456789',
          name: 'dataset_eval',
          digest: '123',
          context: 'eval',
        },
      ],
    },
  },
};
