import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { fetchAPI, getAjaxUrl } from '../../common/utils/FetchUtils';
import type { GetEndpointResponse } from '../types';

export const useEndpointByNameQuery = (name: string | undefined) => {
  return useQuery(['gateway_endpoint_by_name', name], {
    queryFn: () => {
      const params = new URLSearchParams();
      params.append('name', name!);
      return fetchAPI(
        getAjaxUrl(`ajax-api/3.0/mlflow/gateway/endpoints/get?${params.toString()}`),
      ) as Promise<GetEndpointResponse>;
    },
    retry: false,
    enabled: Boolean(name),
  });
};
