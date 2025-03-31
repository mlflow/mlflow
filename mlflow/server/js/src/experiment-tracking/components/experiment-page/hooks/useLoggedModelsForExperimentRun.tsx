import { useMemo } from 'react';
import { isExperimentLoggedModelsUIEnabled } from '../../../../common/utils/FeatureUtils';
import { useSearchLoggedModelsQuery } from '../../../hooks/logged-models/useSearchLoggedModelsQuery';

export const useLoggedModelsForExperimentRun = (experimentId: string, runId: string) => {
  const { data: loggedModelsData, isLoading } = useSearchLoggedModelsQuery(
    { experimentIds: [experimentId] },
    {
      enabled: isExperimentLoggedModelsUIEnabled(),
    },
  );

  const models = useMemo(
    () => loggedModelsData?.filter((model) => model.info?.source_run_id === runId) ?? [],
    [loggedModelsData, runId],
  );

  return { models, isLoading: isExperimentLoggedModelsUIEnabled() && isLoading };
};
