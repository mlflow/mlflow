import { useMemo } from 'react';
import { useSearchLoggedModelsQuery } from '../../../hooks/logged-models/useSearchLoggedModelsQuery';
import type { UseGetRunQueryResponseInputs, UseGetRunQueryResponseOutputs } from '../../run-page/hooks/useGetRunQuery';
import { compact, isEmpty, uniq } from 'lodash';

export const useLoggedModelsForExperimentRun = (
  experimentId: string,
  runId: string,
  runInputs?: UseGetRunQueryResponseInputs,
  runOutputs?: UseGetRunQueryResponseOutputs,
  enabled = true,
) => {
  const searchQuery = useMemo(() => {
    const inputs = runInputs?.modelInputs ?? [];
    const outputs = runOutputs?.modelOutputs ?? [];
    const allModels = [...inputs, ...outputs];
    const modelIds = uniq(compact(allModels.map(({ modelId }) => modelId)));

    if (isEmpty(modelIds)) {
      return undefined;
    }

    return `attributes.model_id IN (${modelIds.map((id) => `'${id}'`).join(',')})`;
  }, [runInputs, runOutputs]);

  const isHookEnabled = enabled && Boolean(searchQuery);

  const {
    data: loggedModelsData,
    isLoading,
    error,
  } = useSearchLoggedModelsQuery(
    { experimentIds: [experimentId], searchQuery },
    {
      enabled: isHookEnabled,
    },
  );

  return {
    // We explicitly check if the hook is supposed to be enabled before returning data,
    // otherwise react-query might erroneously return data from the cache.
    models: isHookEnabled ? loggedModelsData : undefined,
    // Same goes for `isLoading` which sometimes returns `true` despite the hook being disabled.
    isLoading: isHookEnabled && isLoading,
    error,
  };
};
