/**
 * Service layer for Assistant Agent API calls.
 */

import type {
  MessageRequest,
  ToolUseInfo,
  ToolResultInfo,
  AssistantConfig,
  AssistantConfigUpdate,
  HealthCheckResult,
  InstallSkillsResponse,
} from './types';
import { fetchAPI, getAjaxUrl, getDefaultHeaders } from '@mlflow/mlflow/src/common/utils/FetchUtils';

const API_BASE = getAjaxUrl('ajax-api/3.0/mlflow/assistant');

/** Tool-result `content` is string | list[dict] | null on the wire; collapse to a string. */
const normalizeToolResultContent = (content: unknown): string => {
  if (content == null) return '';
  if (typeof content === 'string') return content;
  return JSON.stringify(content, null, 2);
};

/**
 * Process a content block array from an assistant response, emitting text, tool-use,
 * and tool-result blocks in order so the transcript preserves their sequence.
 */
export const processContentBlocks = (
  content: any[],
  onMessage: (text: string) => void,
  onToolUse?: (tools: ToolUseInfo[]) => void,
  onToolResult?: (result: ToolResultInfo) => void,
): void => {
  for (const block of content) {
    if ('text' in block && block.text) {
      onMessage(block.text);
    } else if (block.tool_use_id) {
      // ToolResultBlock: carries the output for a previously-streamed tool call.
      onToolResult?.({
        toolUseId: block.tool_use_id,
        content: normalizeToolResultContent(block.content),
        isError: Boolean(block.is_error),
      });
    } else if (block.name && block.input) {
      // TextBlock-less ToolUseBlock (claude_code & openai_compatible both shape it this way).
      onToolUse?.([
        {
          id: block.id,
          name: block.name,
          description: block.input?.description,
          input: block.input,
        },
      ]);
    }
  }
};

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
 * Create an EventSource for streaming responses.
 */
export const createEventSource = (sessionId: string): EventSource => {
  return new EventSource(`${API_BASE}/sessions/${sessionId}/stream`);
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

export interface SendMessageStreamCallbacks {
  onMessage: (text: string) => void;
  onError: (error: string) => void;
  onDone: () => void;
  onStatus?: (status: string) => void;
  onSessionId?: (sessionId: string) => void;
  onToolUse?: (tools: ToolUseInfo[]) => void;
  onToolResult?: (result: ToolResultInfo) => void;
  onInterrupted?: () => void;
  onUsage?: (usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    total_cost_usd?: number | null;
  }) => void;
}

export interface SendMessageStreamResult {
  eventSource: EventSource | null;
}

/**
 * Send a message and get the response stream via SSE.
 * First POSTs to /message to initiate, then connects to SSE endpoint.
 * Returns the EventSource so caller can close it if needed (e.g., on cancel).
 */
export const sendMessageStream = async (
  request: MessageRequest,
  callbacks: SendMessageStreamCallbacks,
): Promise<SendMessageStreamResult> => {
  const { onMessage, onError, onDone, onStatus, onSessionId, onToolUse, onToolResult, onInterrupted, onUsage } =
    callbacks;

  try {
    // Step 1: POST the message to initiate processing
    // eslint-disable-next-line no-restricted-globals -- See go/spog-fetch
    const response = await fetch(`${API_BASE}/message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getDefaultHeaders(document.cookie),
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.text();
      onError(`Failed to send message: ${error}`);
      return { eventSource: null };
    }

    // Step 2: Get the session_id from the response
    const result = await response.json();
    const sessionId = result.session_id;

    if (!sessionId) {
      onError('No session_id returned from server');
      return { eventSource: null };
    }

    // Notify caller of the session ID
    onSessionId?.(sessionId);

    // Step 3: Connect to the SSE endpoint to receive the stream
    const eventSource = createEventSource(sessionId);

    // Listen for 'message' events (contains assistant's response)
    eventSource.addEventListener('message', (event) => {
      try {
        const data = JSON.parse(event.data);
        // Backend sends: {"message": {"role": "assistant", "content": "..."}}
        if (data.message && data.message.content) {
          const content = data.message.content;
          // Handle string content
          if (typeof content === 'string') {
            onMessage(content);
          } else if (Array.isArray(content)) {
            // Handle ContentBlock array (TextBlock, ThinkingBlock, etc.)
            processContentBlocks(content, onMessage, onToolUse, onToolResult);
          }
        }
      } catch (err) {
        // fail silently
      }
    });

    // Listen for 'stream_event' events (streaming updates)
    eventSource.addEventListener('stream_event', (event) => {
      try {
        const data = JSON.parse(event.data);
        // Backend sends: {"event": {...}}
        if (data.event) {
          // Handle different stream event types
          if (data.event.type === 'content_delta' && data.event.delta?.text) {
            onMessage(data.event.delta.text);
          } else if (data.event.type === 'status') {
            onStatus?.(data.event.status);
          } else if (data.event.type === 'usage' && data.event.usage) {
            onUsage?.(data.event.usage);
          }
        }
      } catch (err) {
        // fail silently
      }
    });

    // Listen for 'done' event (completion)
    eventSource.addEventListener('done', () => {
      onDone();
      eventSource.close();
    });

    // Listen for 'interrupted' event (cancelled by user)
    eventSource.addEventListener('interrupted', () => {
      onInterrupted?.();
      eventSource.close();
    });

    // Listen for 'error' event
    eventSource.addEventListener('error', (event) => {
      // Check if it's a network error or an error event with data
      if (event.type === 'error' && (event as MessageEvent).data) {
        try {
          const data = JSON.parse((event as MessageEvent).data);
          onError(data.error || 'Unknown error');
        } catch {
          onError('Connection error');
        }
      } else if (eventSource.readyState === EventSource.CLOSED) {
        // Connection closed - this can happen after cancel, don't report as error
        return;
      } else {
        onError('Connection error');
      }
      eventSource.close();
    });

    return { eventSource };
  } catch (error) {
    onError(error instanceof Error ? error.message : 'Unknown error');
    return { eventSource: null };
  }
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
