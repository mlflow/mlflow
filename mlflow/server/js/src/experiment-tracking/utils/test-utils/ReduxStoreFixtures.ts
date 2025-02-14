/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { RunTag, Metric } from '../../sdk/MlflowMessages';

export const mockExperiment = (eid: any, name: any) => {
  return { experimentId: eid, name: name, allowedActions: [] };
};

export const mockRunInfo = (
  run_id: any,
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
