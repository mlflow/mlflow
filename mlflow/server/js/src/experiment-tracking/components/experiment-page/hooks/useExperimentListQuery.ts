import { useQuery, QueryFunctionContext, defaultContext } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { MlflowService } from '../../../sdk/MlflowService';
import { useCallback, useContext, useRef, useState } from 'react';
import { SearchExperimentsApiResponse } from '../../../types';

const ExperimentListQueryKeyHeader = 'experiment_list';

type ExperimentListQueryKey = [typeof ExperimentListQueryKeyHeader, { searchFilter?: string; pageToken?: string }];

export const useInvalidateExperimentList = () => {
  const context = useContext(defaultContext);
  return () => {
    context?.invalidateQueries({ queryKey: [ExperimentListQueryKeyHeader] });
  };
};

const queryFn = ({ queryKey }: QueryFunctionContext<ExperimentListQueryKey>) => {
  const [, { searchFilter, pageToken }] = queryKey;
  return MlflowService.searchExperiments({
    filter: searchFilter,
    max_results: 25,
    page_token: pageToken,
  });
};

export const useExperimentListQuery = ({
  searchFilter,
}: {
  searchFilter?: string;
} = {}) => {
  const previousPageTokens = useRef<(string | undefined)[]>([]);

  const [currentPageToken, setCurrentPageToken] = useState<string | undefined>(undefined);

  const queryResult = useQuery<
    SearchExperimentsApiResponse,
    Error,
    SearchExperimentsApiResponse,
    ExperimentListQueryKey
  >([ExperimentListQueryKeyHeader, { searchFilter, pageToken: currentPageToken }], {
    queryFn,
    retry: false,
  });

  const onNextPage = useCallback(() => {
    previousPageTokens.current.push(currentPageToken);
    setCurrentPageToken(queryResult.data?.next_page_token);
  }, [queryResult.data?.next_page_token, currentPageToken]);

  const onPreviousPage = useCallback(() => {
    const previousPageToken = previousPageTokens.current.pop();
    setCurrentPageToken(previousPageToken);
  }, []);

  return {
    data: queryResult.data?.experiments,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    hasNextPage: queryResult.data?.next_page_token !== undefined,
    hasPreviousPage: Boolean(currentPageToken),
    onNextPage,
    onPreviousPage,
    refetch: queryResult.refetch,
  };
};
