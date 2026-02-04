import { useCallback } from 'react';
import { useSearchParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';

export const SELECTED_DATASET_ID_QUERY_PARAM_KEY = 'selectedDatasetId';

/**
 * Query param-powered hook that returns the selected dataset ID.
 * Used to persist dataset selection in the URL for the datasets tab,
 * enabling shareable URLs.
 */
export const useSelectedDatasetBySearchParam = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedDatasetId = searchParams.get(SELECTED_DATASET_ID_QUERY_PARAM_KEY) ?? undefined;

  const setSelectedDatasetId = useCallback(
    (datasetId: string | undefined) => {
      setSearchParams(
        (params) => {
          if (!datasetId) {
            params.delete(SELECTED_DATASET_ID_QUERY_PARAM_KEY);
            return params;
          }
          params.set(SELECTED_DATASET_ID_QUERY_PARAM_KEY, datasetId);
          return params;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const clearSelectedDatasetId = useCallback(() => {
    setSearchParams(
      (params) => {
        params.delete(SELECTED_DATASET_ID_QUERY_PARAM_KEY);
        return params;
      },
      { replace: true },
    );
  }, [setSearchParams]);

  return [selectedDatasetId, setSelectedDatasetId, clearSelectedDatasetId] as const;
};
