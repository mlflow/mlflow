import { useMemo } from 'react';
import { useSearchLoggedModelsQuery } from '../../../hooks/logged-models/useSearchLoggedModelsQuery';
import type { LoggedModelProto } from '../../../types';

export const useLoggedModelsForExperimentRunsTable = ({
  experimentIds,
  enabled = true,
}: {
  experimentIds: string[];
  enabled?: boolean;
}) => {
  const { data: loggedModelsData } = useSearchLoggedModelsQuery(
    { experimentIds },
    {
      enabled,
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
