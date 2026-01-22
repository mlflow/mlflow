/**
 * Service layer for Assistant Agent API calls.
 */

import type { MessageRequest, ToolUseInfo, AssistantConfig, HealthCheckResult, InstallSkillsResponse } from './types';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

const API_BASE = getAjaxUrl('ajax-api/3.0/mlflow/assistant');

/**
 * Process content block array from assistant response.
 * Extracts text or tool uses and calls appropriate callbacks.
 */
const processContentBlocks = (
  content: any[],
  onMessage: (text: string) => void,
  onToolUse?: (tools: ToolUseInfo[]) => void,
): void => {
  // Extract text from TextBlock items
  const text = content
    .filter((block: any) => 'text' in block)
    .map((block: any) => block.text)
    .join('');

  if (text) {
    // Clear tools and show text when assistant is responding
    onToolUse?.([]);
    onMessage(text);
    return;
  }

  // Only show tool uses when there's no text response yet
  const toolUses = content
    .filter((block: any) => block.name && block.input && !block.tool_use_id)
    .map((block: any) => ({
      id: block.id,
      name: block.name,
      description: block.input?.description,
      input: block.input,
    }));
  if (toolUses.length > 0 && onToolUse) {
    onToolUse(toolUses);
  }
};

/**
 * Check if a provider is healthy (CLI installed and authenticated).
 * Returns { ok: true } on success, or { ok: false, error, status } if not set up.
 * Status codes: 412 = CLI not installed, 401 = not authenticated, 404 = provider not found
 */
export const checkProviderHealth = async (provider: string): Promise<HealthCheckResult> => {
  const response = await fetch(`${API_BASE}/providers/${provider}/health`);
  if (response.ok) {
    return { ok: true };
  }
  const data = await response.json();
  return { ok: false, error: data.detail || 'Unknown error', status: response.status };
};

/**
 * Get the assistant configuration.
 */
export const getConfig = async (): Promise<AssistantConfig> => {
  const response = await fetch(`${API_BASE}/config`);
  if (!response.ok) {
    throw new Error(`Failed to get config: ${response.statusText}`);
  }
  return response.json();
};

/**
 * Update the assistant configuration.
 */
export const updateConfig = async (config: Partial<AssistantConfig>): Promise<AssistantConfig> => {
  const response = await fetch(`${API_BASE}/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!response.ok) {
    throw new Error(`Failed to update config: ${response.statusText}`);
  }
  return response.json();
};

/**
 * Create an EventSource for streaming responses.
 */
export const createEventSource = (sessionId: string): EventSource => {
  return new EventSource(`${API_BASE}/sessions/${sessionId}/stream`);
};

/**
 * Send a message and get the response stream via SSE.
 * First POSTs to /message to initiate, then connects to SSE endpoint.
 */
export const sendMessageStream = async (
  request: MessageRequest,
  onMessage: (text: string) => void,
  onError: (error: string) => void,
  onDone: () => void,
  onStatus?: (status: string) => void,
  onSessionId?: (sessionId: string) => void,
  onToolUse?: (tools: ToolUseInfo[]) => void,
): Promise<void> => {
  try {
    // Step 1: POST the message to initiate processing
    const response = await fetch(`${API_BASE}/message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.text();
      onError(`Failed to send message: ${error}`);
      return;
    }

    // Step 2: Get the session_id from the response
    const result = await response.json();
    const sessionId = result.session_id;

    if (!sessionId) {
      onError('No session_id returned from server');
      return;
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
            processContentBlocks(content, onMessage, onToolUse);
          }
        }
      } catch (err) {
        console.error('Failed to parse message event:', err);
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
          }
        }
      } catch (err) {
        console.error('Failed to parse stream_event:', err);
      }
    });

    // Listen for 'done' event (completion)
    eventSource.addEventListener('done', (event) => {
      try {
        const data = JSON.parse(event.data);
        // Backend sends: {"result": null, "session_id": "..."}
        onToolUse?.([]);
        onDone();
        eventSource.close();
      } catch (err) {
        console.error('Failed to parse done event:', err);
        onToolUse?.([]);
        onDone();
        eventSource.close();
      }
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
        onError('Connection closed');
      } else {
        onError('Connection error');
      }
      eventSource.close();
    });
  } catch (error) {
    onError(error instanceof Error ? error.message : 'Unknown error');
  }
};

/**
 * Install skills from the MLflow skills repository.
 * Returns { installed_skills, skills_directory } on success.
 * Throws with error.status for:
 *   412 = git not installed
 *   500 = clone failed
 */
export const installSkills = async (skillsLocation: string): Promise<InstallSkillsResponse> => {
  const response = await fetch(`${API_BASE}/skills/install`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ skills_location: skillsLocation }),
  });
  if (!response.ok) {
    const data = await response.json();
    const error = new Error(data.detail || 'Failed to install skills');
    (error as any).status = response.status;
    throw error;
  }
  return response.json();
};
