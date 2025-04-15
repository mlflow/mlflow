import { compact, uniq, uniqBy } from 'lodash';
import { useMemo } from 'react';
import type {
  UseGetRunQueryResponseInputs,
  UseGetRunQueryResponseOutputs,
  UseGetRunQueryResponseRunInfo,
} from '../../components/run-page/hooks/useGetRunQuery';
import type { LoggedModelProto, RunInfoEntity } from '../../types';
import { useGetLoggedModelQueries } from './useGetLoggedModelQuery';

type LoggedModelProtoWithRunDirection = LoggedModelProto & { direction: 'input' | 'output' };

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
) => {
  const inputModelIds = compact(uniq(inputs?.modelInputs?.map((modelInput) => modelInput.modelId)));
  const outputModelIds = compact(uniq(outputs?.modelOutputs?.map((modelOutput) => modelOutput.modelId)));
  const inputModelQueries = useGetLoggedModelQueries(inputModelIds);
  const outputModelQueries = useGetLoggedModelQueries(outputModelIds);

  const inputLoggedModels = useMemo(() => {
    return inputModelQueries.map<LoggedModelProtoWithRunDirection | undefined>((query) => {
      if (!query.data?.model) return undefined;
      return { ...query.data?.model, direction: 'input' as const };
    });
  }, [inputModelQueries]);

  const outputLoggedModels = useMemo(() => {
    return outputModelQueries.map<LoggedModelProtoWithRunDirection | undefined>((query) => {
      if (!query.data?.model) return undefined;
      return { ...query.data?.model, direction: 'output' as const };
    });
  }, [outputModelQueries]);

  const models = useMemo(() => {
    return (
      uniqBy(
        compact([...inputLoggedModels, ...outputLoggedModels]).map(filterMetricsByMatchingRunId(runInfo?.runUuid)),
        (modelData) => modelData.info?.model_id,
      ) ?? []
    );
  }, [inputLoggedModels, outputLoggedModels, runInfo]);

  const errors = [...inputModelQueries, ...outputModelQueries].map((query) => query.error).filter(Boolean);

  const isLoading = [...inputModelQueries, ...outputModelQueries].some((query) => query.isLoading);

  return { models, errors, isLoading };
};
