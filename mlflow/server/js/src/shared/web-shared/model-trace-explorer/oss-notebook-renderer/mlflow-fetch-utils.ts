import { fetchAPI } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import type { ModelTrace, ModelTraceData } from '../ModelTrace.types';
import { getAjaxUrl } from '../ModelTraceExplorer.request.utils';

// returns ModelTrace if the request is successful, otherwise returns an error message
export async function getTraceArtifact(requestId: string): Promise<ModelTrace | string> {
  try {
    const jsonData = await fetchAPI(getAjaxUrl(`ajax-api/2.0/mlflow/get-trace-artifact?request_id=${requestId}`));

    // successful request containing span data
    if (jsonData.spans) {
      return {
        info: {
          request_id: requestId,
        },
        data: jsonData as ModelTraceData,
      };
    }

    if (jsonData.error_code) {
      return jsonData.message;
    }

    return 'Unknown error occurred';
  } catch (e) {
    if (e instanceof Error) {
      return e.message;
    }

    if (typeof e === 'string') {
      return e;
    }

    return 'Unknown error occurred';
  }
}
