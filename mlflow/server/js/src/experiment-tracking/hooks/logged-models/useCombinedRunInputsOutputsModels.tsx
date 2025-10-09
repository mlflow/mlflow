import { compact, uniq, uniqBy } from 'lodash';
import { useMemo } from 'react';
import type {
  UseGetRunQueryResponseInputs,
  UseGetRunQueryResponseOutputs,
  UseGetRunQueryResponseRunInfo,
} from '../../components/run-page/hooks/useGetRunQuery';
import type { LoggedModelProto, RunInfoEntity } from '../../types';

type LoggedModelProtoWithRunDirection = LoggedModelProto & { direction: 'input' | 'output'; step?: string };

const filterMetricsByMatchingRunId = (runUuid?: string | null) => (loggedModel: LoggedModelProtoWithRunDirection) => {
  if (loggedModel.data?.metrics) {
    return {
      ...loggedModel,
      data: {
        ...loggedModel.data,
        metrics: loggedModel.data.metrics.filter((metric) => !runUuid || metric.run_id === runUuid),
      },
    };
  }
  return loggedModel;
};

export const useCombinedRunInputsOutputsModels = (
  inputs?: UseGetRunQueryResponseInputs,
  outputs?: UseGetRunQueryResponseOutputs,
  runInfo?: RunInfoEntity | UseGetRunQueryResponseRunInfo,
  loggedModelsV3?: LoggedModelProto[],
) => {
  const inputLoggedModels = useMemo(() => {
    const inputModelIds = compact(uniq(inputs?.modelInputs?.map((modelInput) => modelInput.modelId)));
    return inputModelIds.map<LoggedModelProtoWithRunDirection | undefined>((model_id) => {
      const model = loggedModelsV3?.find((model) => model.info?.model_id === model_id);
      if (!model) return undefined;
      return { ...model, direction: 'input' as const };
    });
  }, [inputs?.modelInputs, loggedModelsV3]);

  const outputLoggedModels = useMemo(() => {
    const outputModelIds = compact(uniq(outputs?.modelOutputs?.map((modelOutput) => modelOutput.modelId)));
    return outputModelIds.map<LoggedModelProtoWithRunDirection | undefined>((model_id) => {
      const model = loggedModelsV3?.find((model) => model.info?.model_id === model_id);

      const correspondingOutputEntry = outputs?.modelOutputs?.find(({ modelId }) => modelId === model?.info?.model_id);

      if (!model) return undefined;
      return { ...model, direction: 'output' as const, step: correspondingOutputEntry?.step ?? undefined };
    });
  }, [outputs?.modelOutputs, loggedModelsV3]);

  const modelsWithDirection = useMemo(() => {
    return (
      uniqBy(
        compact([...inputLoggedModels, ...outputLoggedModels]).map(filterMetricsByMatchingRunId(runInfo?.runUuid)),
        (modelData) => modelData.info?.model_id,
      ) ?? []
    );
  }, [inputLoggedModels, outputLoggedModels, runInfo]);

  return { models: modelsWithDirection };
};
