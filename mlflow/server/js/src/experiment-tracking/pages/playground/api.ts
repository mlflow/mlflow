import { fetchOrFail, getAjaxUrl } from '../../../common/utils/FetchUtils';
import type { ChatCompletionRequest, ChatCompletionResponse } from './types';

// fetchOrFail throws a NetworkRequestError with `.response` attached and the
// body still unread. Walk the body to surface the upstream provider's actual
// error string in place of the predefined error class default
// (e.g. "Internal server error"). FastAPI emits `{detail: ...}`; MLflow's
// translate_http_exception emits either `{detail: "<string>"}` (AIGatewayException)
// or `{detail: {error_code, message}}` (MlflowException). Some upstream/proxy
// paths use `{error: {message}}` or a top-level `{message}`.
const enrichErrorFromResponseBody = async (e: unknown): Promise<unknown> => {
  const response = (e as { response?: Response }).response;
  if (!response || response.bodyUsed) return e;
  try {
    const body = await response.json();
    const extracted =
      (typeof body?.detail === 'object' && typeof body.detail?.message === 'string' && body.detail.message) ||
      (typeof body?.detail === 'string' && body.detail) ||
      (typeof body?.message === 'string' && body.message) ||
      (typeof body?.error === 'object' && typeof body.error?.message === 'string' && body.error.message) ||
      undefined;
    if (extracted && e instanceof Error) {
      e.message = extracted;
    }
  } catch {
    // body unparseable — keep the predefined default
  }
  return e;
};

export const PlaygroundApi = {
  chatCompletion: async (request: ChatCompletionRequest): Promise<ChatCompletionResponse> => {
    try {
      const response = await fetchOrFail(getAjaxUrl('gateway/mlflow/v1/chat/completions'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });
      return (await response.json()) as ChatCompletionResponse;
    } catch (e) {
      throw await enrichErrorFromResponseBody(e);
    }
  },
};
