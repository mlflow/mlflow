import type {
  ExperimentEntity,
  GetRunApiResponse,
  RunInfoEntity,
  SearchRunsApiResponse,
  GetExperimentApiResponse,
  SearchExperimentsApiResponse,
  RunDatasetWithTags,
  RunEntity,
  RunModelEntity,
} from '../types';

const transformRunInfoEntity = (snakeCaseRunInfoEntity: any): RunInfoEntity => ({
  ...snakeCaseRunInfoEntity,
  artifactUri: snakeCaseRunInfoEntity.artifact_uri,
  endTime: snakeCaseRunInfoEntity.end_time,
  experimentId: snakeCaseRunInfoEntity.experiment_id,
  lifecycleStage: snakeCaseRunInfoEntity.lifecycle_stage,
  runUuid: snakeCaseRunInfoEntity.run_uuid,
  runName: snakeCaseRunInfoEntity.run_name,
  startTime: snakeCaseRunInfoEntity.start_time,
});

const transformDatasetInputEntity = (snakeCaseDatasetInputEntity: any): RunDatasetWithTags => {
  if (!snakeCaseDatasetInputEntity || !snakeCaseDatasetInputEntity.dataset) {
    return snakeCaseDatasetInputEntity;
  }
  return {
    ...snakeCaseDatasetInputEntity,
    dataset: {
      ...snakeCaseDatasetInputEntity.dataset,
      sourceType: snakeCaseDatasetInputEntity.dataset.source_type,
    },
  };
};

const transformModelEntity = (snakeCaseModelEntity: any): RunModelEntity => {
  return {
    modelId: snakeCaseModelEntity.model_id,
  };
};

const transformRunInputsEntity = (snakeCaseInputsData: any): RunEntity['inputs'] => {
  if (!snakeCaseInputsData) {
    return snakeCaseInputsData;
  }

  const datasetInputs = snakeCaseInputsData.dataset_inputs
    ?.map((input: any) => transformDatasetInputEntity(input))
    .filter(Boolean);

  const modelInputs = snakeCaseInputsData.model_inputs
    ?.map((input: any) => transformModelEntity(input))
    .filter(Boolean);

  return {
    ...(datasetInputs?.length ? { datasetInputs } : {}),
    ...(modelInputs?.length ? { modelInputs } : {}),
  };
};

const transformRunOutputsEntity = (snakeCaseOutputsData: any): RunEntity['outputs'] => {
  if (!snakeCaseOutputsData) {
    return snakeCaseOutputsData;
  }

  const modelOutputs = snakeCaseOutputsData.model_outputs
    ?.map((output: any) => transformModelEntity(output))
    .filter(Boolean);

  return {
    ...(modelOutputs?.length ? { modelOutputs } : {}),
  };
};

const transformExperimentEntity = (snakeCaseExperimentEntity: any): Omit<ExperimentEntity, 'allowedActions'> => ({
  ...snakeCaseExperimentEntity,
  artifactLocation: snakeCaseExperimentEntity.artifact_location,
  creationTime: snakeCaseExperimentEntity.creation_time,
  experimentId: snakeCaseExperimentEntity.experiment_id,
  lastUpdateTime: snakeCaseExperimentEntity.last_update_time,
  lifecycleStage: snakeCaseExperimentEntity.lifecycle_stage,
  tags: snakeCaseExperimentEntity.tags ?? [],
});

export const transformGetRunResponse = (originalResponse: any): GetRunApiResponse => {
  if (!originalResponse || !originalResponse.run || !originalResponse.run.info) {
    return originalResponse;
  }
  return {
    ...originalResponse,
    run: {
      ...originalResponse.run,
      info: transformRunInfoEntity(originalResponse.run.info),
      inputs: transformRunInputsEntity(originalResponse.run.inputs),
    },
  };
};

export const transformSearchRunsResponse = (originalResponse: any): SearchRunsApiResponse => {
  if (!originalResponse || !originalResponse.runs) {
    return originalResponse;
  }

  return {
    ...originalResponse,
    runs: originalResponse.runs.map((run: any) => ({
      ...run,
      info: transformRunInfoEntity(run.info),
      inputs: transformRunInputsEntity(run.inputs),
      outputs: transformRunOutputsEntity(run.outputs),
    })),
  };
};

export const transformSearchExperimentsResponse = (originalResponse: any): SearchExperimentsApiResponse => {
  if (!originalResponse || !originalResponse.experiments) {
    return originalResponse;
  }
  return {
    ...originalResponse,
    experiments: originalResponse.experiments.map((experiment: any) => transformExperimentEntity(experiment)),
  };
};

export const transformGetExperimentResponse = (originalResponse: any): GetExperimentApiResponse => {
  if (!originalResponse || !originalResponse.experiment) {
    return originalResponse;
  }
  return {
    ...originalResponse,
    experiment: transformExperimentEntity(originalResponse.experiment),
  };
};
