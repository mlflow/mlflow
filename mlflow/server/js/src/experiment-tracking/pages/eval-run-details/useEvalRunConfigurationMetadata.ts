import { useApolloClient } from '../../../common/utils/graphQLHooks';
import { useQueries, type UseQueryResult } from '@databricks/web-shared/query-client';
import type { GetRun, GetRunVariables } from '../../../graphql/__generated__/graphql';
import { GET_RUN_QUERY } from '../../components/run-page/hooks/useGetRunQuery';
import { getEvalRunConfigurationMetadata, type EvalRunConfigurationMetadata } from './evalRunConfigurationData';

export const useEvalRunConfigurationMetadata = (
  experimentId: string,
  runUuids: string[],
): UseQueryResult<EvalRunConfigurationMetadata>[] => {
  const client = useApolloClient();
  return useQueries({
    queries: runUuids.map((runUuid) => ({
      queryKey: ['eval-run-comparison-configuration', experimentId, runUuid],
      queryFn: async () => {
        const { data } = await client.query<GetRun, GetRunVariables>({
          query: GET_RUN_QUERY,
          variables: { data: { runId: runUuid } },
          fetchPolicy: 'network-only',
        });
        if (data.mlflowGetRun?.apiError || !data.mlflowGetRun?.run) {
          throw new Error('Could not load run configuration');
        }
        return getEvalRunConfigurationMetadata(data.mlflowGetRun.run);
      },
      staleTime: 30000,
      refetchOnWindowFocus: false,
      retry: false,
    })),
  });
};
