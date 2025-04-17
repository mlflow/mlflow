export const mockExperiment = (eid: string, name: string) => {
  return { experimentId: eid, name: name, allowedActions: [] };
};

export const mockRunInfo = (
  run_id: string,
  experiment_id = undefined,
  artifact_uri = undefined,
  lifecycle_stage = undefined,
) => {
  return {
    runUuid: run_id,
    experimentId: experiment_id,
    artifactUri: artifact_uri,
    lifecycleStage: lifecycle_stage,
  };
};
