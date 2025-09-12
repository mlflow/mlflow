import { useMemo } from 'react';
import { useGetLoggedModelsQuery } from '../../../hooks/logged-models/useGetLoggedModelsQuery';
import type { UseGetRunQueryResponseInputs, UseGetRunQueryResponseOutputs } from '../../run-page/hooks/useGetRunQuery';
import { compact, isEmpty, uniq } from 'lodash';

export const useLoggedModelsForExperimentRunV2 = ({
  runInputs,
  runOutputs,
  enabled = true,
}: {
  runInputs?: UseGetRunQueryResponseInputs;
  runOutputs?: UseGetRunQueryResponseOutputs;
  enabled?: boolean;
}) => {
  const modelIds = useMemo(() => {
    const inputs = runInputs?.modelInputs ?? [];
    const outputs = runOutputs?.modelOutputs ?? [];
    const allModels = [...inputs, ...outputs];
    const modelIds = uniq(compact(allModels.map(({ modelId }) => modelId)));

    if (isEmpty(modelIds)) {
      return undefined;
    }

    return modelIds;
  }, [runInputs, runOutputs]);

  const isHookEnabled = enabled && !isEmpty(modelIds);

  const {
    data: loggedModelsData,
    isLoading,
    error,
  } = useGetLoggedModelsQuery(
    {
      modelIds,
    },
    {
      enabled: isHookEnabled,
    },
  );

  return { models: loggedModelsData, isLoading: isHookEnabled && isLoading, error };
};
