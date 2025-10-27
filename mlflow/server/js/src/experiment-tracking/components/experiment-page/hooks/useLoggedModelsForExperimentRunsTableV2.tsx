import { useMemo } from 'react';
import type { LoggedModelProto, RunInfoEntity, RunInputsType, RunOutputsType } from '../../../types';
import { useGetLoggedModelsQuery } from '../../../hooks/logged-models/useGetLoggedModelsQuery';
import { uniq } from 'lodash';

/**
 * Custom hook to fetch logged models for experiment runs table.
 * It processes run data to extract model IDs and fetches the corresponding logged models.
 *
 * Contrary to V1, this version uses runs' inputs and outputs to determine the model IDs,
 * instead of getting all models logged in the experiment.
 */
export const useLoggedModelsForExperimentRunsTableV2 = ({
  runData,
  enabled,
}: {
  runData: {
    runInfo: RunInfoEntity;
    inputs?: RunInputsType;
    outputs?: RunOutputsType;
  }[];
  enabled?: boolean;
}) => {
  const modelIdsByRunId = useMemo(() => {
    if (!enabled) {
      return {};
    }
    const modelIdsMap: Record<string, string[]> = {};
    for (const { runInfo, inputs, outputs } of runData) {
      const runId = runInfo.runUuid;
      const inputModelIds = inputs?.modelInputs?.map((input) => input.modelId) || [];
      const outputModelIds = outputs?.modelOutputs?.map((output) => output.modelId) || [];
      const allModelIds = [...inputModelIds, ...outputModelIds];
      if (runId && allModelIds.length > 0) {
        modelIdsMap[runId] = uniq(allModelIds); // Ensure unique model IDs per run
      }
    }
    return modelIdsMap;
  }, [runData, enabled]);

  const modelIds = useMemo(() => {
    if (!enabled) {
      return [];
    }
    // Unique model IDs across all runs with no repeats
    return uniq(Object.values(modelIdsByRunId).flat());
  }, [modelIdsByRunId, enabled]);

  const loggedModelsData = useGetLoggedModelsQuery(
    {
      modelIds,
    },
    {
      enabled: enabled && modelIds.length > 0,
    },
  );

  const loggedModelsByRunId = useMemo(() => {
    if (!loggedModelsData.data) {
      return {};
    }
    return Object.entries(modelIdsByRunId).reduce<Record<string, LoggedModelProto[]>>((acc, [runId, modelIds]) => {
      acc[runId] = loggedModelsData.data?.filter((model) => modelIds.includes(model.info?.model_id || '')) ?? [];
      return acc;
    }, {});
  }, [modelIdsByRunId, loggedModelsData.data]);

  return loggedModelsByRunId;
};
