import { fetchAPI, getAjaxUrl } from '../../common/utils/FetchUtils';
import { useQuery } from '../../common/utils/reactQueryHooks';
import type { TelemetryConfig } from '../worker/types';

export const useTelemetryConfig = () =>
  useQuery<TelemetryConfig>(['ui-telemetry-config'], {
    queryFn: () => fetchAPI(getAjaxUrl('ajax-api/3.0/mlflow/ui-telemetry')),
    retry: false,
  });
