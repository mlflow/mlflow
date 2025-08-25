import { ParagraphSkeleton } from '@databricks/design-system';
import type { NetworkRequestError } from '@databricks/web-shared/errors';
import { useQuery } from '@databricks/web-shared/query-client';

import { ErrorCell } from './ErrorCell';
import { NullCell } from './NullCell';
import { makeRequest } from '../utils/FetchUtils';
import MlflowUtils from '../utils/MlflowUtils';
import { Link } from '../utils/RoutingUtils';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

export const RunName = (props: { experimentId: string; runUuid: string }) => {
  const { experimentId, runUuid } = props;

  const { data, isLoading, error } = useRunName(experimentId, runUuid);

  const runName = data?.runs?.[0]?.info?.run_name;

  if (isLoading) {
    return <ParagraphSkeleton />;
  }

  if (error) {
    return <ErrorCell />;
  }

  if (!runName) {
    return <NullCell />;
  }

  return (
    <Link
      css={{
        display: 'flex',
        maxWidth: '100%',
        textOverflow: 'ellipsis',
        overflow: 'hidden',
        whiteSpace: 'nowrap',
      }}
      to={MlflowUtils.getRunPageRoute(experimentId, runUuid)}
      title={runName}
    >
      {runName}
    </Link>
  );
};

interface RunNameResponse {
  runs: {
    data: {
      tags: { [key: string]: string };
    };
    info: {
      run_name: string;
    };
  }[];
}
const useRunName = (experimentId: string, runUuid: string) => {
  return useQuery<RunNameResponse, NetworkRequestError>({
    queryKey: ['runName', experimentId, runUuid],
    cacheTime: Infinity,
    staleTime: Infinity,
    retry: 1, // limit retries so we don't spam the api
    refetchOnMount: false,
    queryFn: async () => {
      const filter = `run_id IN ('${runUuid}')`;

      const res: RunNameResponse = await makeRequest(getAjaxUrl('ajax-api/2.0/mlflow/runs/search'), 'POST', {
        experiment_ids: [experimentId],
        filter,
      });

      return res;
    },
  });
};
