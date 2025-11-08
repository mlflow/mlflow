import { renderHook, waitFor } from '../../../common/utils/TestUtils.react18';
import type {
  UseGetRunQueryResponseInputs,
  UseGetRunQueryResponseOutputs,
} from '../../components/run-page/hooks/useGetRunQuery';
import { useCombinedRunInputsOutputsModels } from './useCombinedRunInputsOutputsModels';
import type { RunInfoEntity } from '../../types';
import { LoggedModelStatusProtoEnum } from '../../types';

describe('useCombinedRunInputsOutputsModels', () => {
  const generateTestModel = (modelId: string) => ({
    info: {
      artifact_uri: `dbfs:/databricks/mlflow/${modelId}`,
      creation_timestamp_ms: 1728322600000,
      last_updated_timestamp_ms: 1728322600000,
      source_run_id: 'run-id-1',
      creator_id: 'test@test.com',
      experiment_id: 'test-experiment',
      model_id: modelId,
      model_type: 'Agent',
      name: `${modelId}-name`,
      tags: [],
      status_message: 'Ready',
      status: LoggedModelStatusProtoEnum.LOGGED_MODEL_READY,
      registrations: [],
    },
    data: {
      metrics: [
        {
          key: 'metric-test',
          value: 10,
          run_id: 'test-run-id',
        },
        {
          key: 'unrelated-metric',
          value: 10,
          run_id: 'unrelated-run-id',
        },
      ],
    },
  });

  const testRunInfo: RunInfoEntity = {
    runUuid: 'test-run-id',
    experimentId: 'experiment-id',
    startTime: 0,
    endTime: 1,
    artifactUri: '',
    lifecycleStage: 'active',
    status: 'FINISHED',
    runName: 'run-name',
  };

  const renderTestHook = (...params: Parameters<typeof useCombinedRunInputsOutputsModels>) =>
    renderHook(() => useCombinedRunInputsOutputsModels(...params));

  it('returns an empty array if no inputs or outputs are provided', () => {
    const { result } = renderTestHook();
    expect(result.current.models).toEqual([]);
  });

  it('returns logged models with direction and filtered metrics', async () => {
    const inputs: UseGetRunQueryResponseInputs = {
      __typename: 'MlflowRunInputs',
      datasetInputs: null,
      modelInputs: [
        { __typename: 'MlflowModelInput', modelId: 'input-model-1' },
        { __typename: 'MlflowModelInput', modelId: 'input-model-2' },
      ],
    };

    const outputs: UseGetRunQueryResponseOutputs = {
      __typename: 'MlflowRunOutputs',
      modelOutputs: [
        { __typename: 'MlflowModelOutput', modelId: 'output-model-1', step: '0' },
        { __typename: 'MlflowModelOutput', modelId: 'output-model-2', step: '0' },
      ],
    };

    const testModels = [
      generateTestModel('input-model-1'),
      generateTestModel('input-model-2'),
      generateTestModel('output-model-1'),
      generateTestModel('output-model-2'),
    ];

    const { result } = renderTestHook(inputs, outputs, testRunInfo, testModels);

    // Wait for models to be fetched
    await waitFor(() => {
      expect(result.current.models.length).toBe(4);
    });

    // Assert IDs
    const loggedModelIds = result.current.models.map((model) => model.info?.model_id);
    expect(loggedModelIds).toContain('input-model-1');
    expect(loggedModelIds).toContain('input-model-2');
    expect(loggedModelIds).toContain('output-model-1');
    expect(loggedModelIds).toContain('output-model-2');

    // Assert proper directions
    expect(result.current.models.find((model) => model.info?.model_id === 'input-model-1')?.direction).toBe('input');
    expect(result.current.models.find((model) => model.info?.model_id === 'input-model-2')?.direction).toBe('input');
    expect(result.current.models.find((model) => model.info?.model_id === 'output-model-1')?.direction).toBe('output');
    expect(result.current.models.find((model) => model.info?.model_id === 'output-model-2')?.direction).toBe('output');

    // Assert that only related metrics are present in the result
    const metrics = result.current.models.find((model) => model.info?.model_id === 'input-model-1')?.data?.metrics;

    expect(metrics?.length).toEqual(1);
    expect(metrics?.map((metric) => metric.key)).toContain('metric-test');
  });
});
