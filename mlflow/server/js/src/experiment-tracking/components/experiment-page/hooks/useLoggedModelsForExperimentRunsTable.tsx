import { useMemo } from 'react';
import { isExperimentLoggedModelsUIEnabled } from '../../../../common/utils/FeatureUtils';
import { useSearchLoggedModelsQuery } from '../../../hooks/logged-models/useSearchLoggedModelsQuery';
import { LoggedModelProto } from '../../../types';

export const useLoggedModelsForExperimentRunsTable = (experimentIds: string[]) => {
  const { data: loggedModelsData } = useSearchLoggedModelsQuery(
    { experimentIds },
    {
      enabled: isExperimentLoggedModelsUIEnabled(),
    },
  );

  const loggedModelsByRunId = useMemo(
    () =>
      loggedModelsData?.reduce<Record<string, LoggedModelProto[]>>((acc, model) => {
        const { source_run_id } = model.info ?? {};
        if (!source_run_id) {
          return acc;
        }
        if (!acc[source_run_id]) {
          acc[source_run_id] = [];
        }
        acc[source_run_id].push(model);
        return acc;
      }, {}),
    [loggedModelsData],
  );

  return loggedModelsByRunId;
};
