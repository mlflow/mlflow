/**
 * Service layer for Assistant Agent REST API calls.
 *
 * Streaming transports live in ./transports/* and are re-exported below so existing
 * `from './AssistantService'` imports keep resolving every symbol.
 */

import type { AssistantConfig, AssistantConfigUpdate, HealthCheckResult, InstallSkillsResponse } from './types';
import { API_BASE } from './transports/shared';
import { fetchAPI, getAjaxUrl, getDefaultHeaders } from '@mlflow/mlflow/src/common/utils/FetchUtils';

// Streaming transports (re-exported so callers can keep importing from './AssistantService').
export { sendMessageStream, createEventSource, resumeStream } from './transports/eventSourceTransport';
export { streamChatViaFetch } from './transports/fetchStreamTransport';
export * from './transports/shared';

/**
 * Check if a provider is healthy (CLI installed and authenticated).
 * Returns { ok: true } on success, or { ok: false, error, status } if not set up.
 * Status codes: 412 = CLI not installed, 401 = not authenticated, 404 = provider not found
 */
export const checkProviderHealth = async (provider: string): Promise<HealthCheckResult> => {
  try {
    await fetchAPI(getAjaxUrl(`${API_BASE}/providers/${provider}/health`));
    return { ok: true };
  } catch (error: any) {
    return { ok: false, error: error.message || 'Unknown error', status: error.status };
  }
};

/**
 * Get the assistant configuration.
 */
export const getConfig = async (): Promise<AssistantConfig> => {
  return await fetchAPI(getAjaxUrl(`${API_BASE}/config`));
};

/**
 * Update the assistant configuration.
 * Pass null for a project to remove it.
 */
export const updateConfig = async (config: AssistantConfigUpdate): Promise<AssistantConfig> => {
  return await fetchAPI(getAjaxUrl(`${API_BASE}/config`), {
    method: 'PUT',
    body: JSON.stringify(config),
  });
};

/**
 * Cancel an active session by terminating the backend process.
 */
export const cancelSession = async (sessionId: string): Promise<{ message: string }> => {
  return await fetchAPI(getAjaxUrl(`${API_BASE}/sessions/${sessionId}`), {
    method: 'PATCH',
    body: JSON.stringify({ status: 'cancelled' }),
  });
};

export const listProviderModels = async (provider: string, baseUrl: string, apiKey?: string): Promise<string[]> => {
  // api_key is sent as an X-API-Key header (not a query param) so the
  // bearer token doesn't end up in access logs, browser history, or
  // referer headers.
  const params = new URLSearchParams({ base_url: baseUrl });
  const url = `${API_BASE}/providers/${encodeURIComponent(provider)}/models?${params.toString()}`;
  const headers = {
    ...getDefaultHeaders(document.cookie),
    ...(apiKey ? { 'X-API-Key': apiKey } : {}),
  };
  const response = await fetch(url, { headers });
  if (!response.ok) {
    const data = await response.json();
    throw new Error(data.detail || `Failed to list models for provider '${provider}': ${response.statusText}`);
  }
  const data = await response.json();
  return data.models as string[];
};

/**
 * Install skills from the MLflow skills repository.
 * Returns { installed_skills, skills_directory } on success.
 * Throws with error.status for:
 *   412 = git not installed
 *   500 = clone failed
 */
export const installSkills = async (
  type: 'global' | 'project' | 'custom',
  customPath?: string,
  experimentId?: string,
): Promise<InstallSkillsResponse> => {
  return await fetchAPI(getAjaxUrl(`${API_BASE}/skills/install`), {
    method: 'POST',
    body: JSON.stringify({
      type,
      custom_path: customPath,
      experiment_id: experimentId,
    }),
  });
};
