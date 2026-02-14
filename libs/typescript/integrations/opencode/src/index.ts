/**
 * MLflow Tracing Plugin for OpenCode
 *
 * This plugin listens for session.idle events and creates MLflow traces
 * directly using the mlflow-tracing TypeScript SDK.
 *
 * Usage:
 *   1. Install: npm install @mlflow/opencode mlflow-tracing
 *   2. Add to opencode.json: { "plugin": ["@mlflow/opencode"] }
 *   3. Set environment variables:
 *      export MLFLOW_TRACKING_URI=http://localhost:5000
 *      export MLFLOW_EXPERIMENT_ID=123
 *   4. Run opencode normally - tracing happens automatically
 */

import type { Plugin, PluginInput, Hooks } from '@opencode-ai/plugin';
import {
  init,
  startSpan,
  flushTraces,
  SpanType,
  SpanAttributeKey,
  TraceMetadataKey,
  InMemoryTraceManager,
} from 'mlflow-tracing';

// Track processed turns to avoid duplicate traces
const processedTurns = new Map<string, number>();

// Silent plugin - no console output to avoid TUI interference
const DEBUG = process.env.MLFLOW_OPENCODE_DEBUG === 'true';

// Constants
const NANOSECONDS_PER_MS = 1e6;
const MAX_PREVIEW_LENGTH = 1000;
const MESSAGE_ROLE_USER = 'user';
const MESSAGE_ROLE_ASSISTANT = 'assistant';
const PART_TYPE_TEXT = 'text';
const PART_TYPE_TOOL = 'tool';

// SDK initialization state
let initialized = false;

interface ApiResponse<T> {
  data?: T;
  error?: unknown;
}

interface SessionClient {
  get: (params: { path: { id: string } }) => Promise<ApiResponse<unknown>>;
  messages: (params: {
    path: { id: string };
    query: { limit: number };
  }) => Promise<ApiResponse<unknown[]>>;
}

interface PluginClient {
  session: SessionClient;
}

interface MessagePart {
  type: string;
  text?: string;
  tool?: string;
  callID?: string;
  state?: {
    status?: string;
    input?: Record<string, unknown>;
    output?: string;
    error?: string;
    title?: string;
    time?: {
      start?: number;
      end?: number;
    };
  };
}

interface MessageInfo {
  role?: string;
  modelID?: string;
  providerID?: string;
  tokens?: {
    input?: number;
    output?: number;
    reasoning?: number;
    cache?: {
      read?: number;
      write?: number;
    };
  };
  time?: {
    created?: number;
    completed?: number;
  };
}

interface Message {
  info?: MessageInfo;
  parts?: MessagePart[];
}

interface SessionInfo {
  directory?: string;
  title?: string;
}

/**
 * Initialize the MLflow tracing SDK if not already initialized.
 * Requires MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID environment variables.
 */
function ensureInitialized(): boolean {
  if (initialized) {
    return true;
  }

  const trackingUri = process.env.MLFLOW_TRACKING_URI;
  const experimentId = process.env.MLFLOW_EXPERIMENT_ID;

  if (!trackingUri) {
    if (DEBUG) {
      console.error('[mlflow] MLFLOW_TRACKING_URI not set, skipping initialization');
    }
    return false;
  }

  if (!experimentId) {
    if (DEBUG) {
      console.error('[mlflow] MLFLOW_EXPERIMENT_ID not set, skipping initialization');
    }
    return false;
  }

  try {
    init({ trackingUri, experimentId });
    initialized = true;
    if (DEBUG) {
      console.error('[mlflow] SDK initialized successfully');
    }
    return true;
  } catch (error) {
    if (DEBUG) {
      console.error('[mlflow] Failed to initialize SDK:', error);
    }
    return false;
  }
}

/**
 * Convert timestamp in milliseconds to nanoseconds
 */
function timestampToNs(timestamp: number | undefined): number | undefined {
  return timestamp != null ? Math.floor(timestamp * NANOSECONDS_PER_MS) : undefined;
}

/**
 * Extract the user prompt from messages (find last user message text)
 */
function extractUserPrompt(messages: Message[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.info?.role === MESSAGE_ROLE_USER) {
      const parts = msg.parts || [];
      for (const part of parts) {
        if (part.type === PART_TYPE_TEXT && part.text) {
          return part.text;
        }
      }
    }
  }
  return '';
}

/**
 * Extract the assistant response from messages (find last assistant text)
 */
function extractAssistantResponse(messages: Message[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.info?.role === MESSAGE_ROLE_ASSISTANT) {
      const parts = msg.parts || [];
      for (let j = parts.length - 1; j >= 0; j--) {
        const part = parts[j];
        if (part.type === PART_TYPE_TEXT && part.text?.trim()) {
          return part.text;
        }
      }
    }
  }
  return '';
}

/**
 * Find the index of the last user message
 */
function findLastUserMessageIndex(messages: Message[]): number | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].info?.role === MESSAGE_ROLE_USER) {
      return i;
    }
  }
  return null;
}

/**
 * Reconstruct conversation messages for LLM input history
 */
function reconstructConversationMessages(
  messages: Message[],
  endIdx: number,
): Array<{ role: string; content: string; tool_call_id?: string }> {
  const result: Array<{ role: string; content: string; tool_call_id?: string }> = [];

  for (let i = 0; i < endIdx; i++) {
    const msg = messages[i];
    const role = msg.info?.role;
    const parts = msg.parts || [];

    if (role === MESSAGE_ROLE_USER) {
      const textParts = parts
        .filter((p) => p.type === PART_TYPE_TEXT && p.text)
        .map((p) => p.text || '');
      if (textParts.length > 0) {
        result.push({ role: 'user', content: textParts.join('\n') });
      }
    } else if (role === MESSAGE_ROLE_ASSISTANT) {
      const textParts = parts
        .filter((p) => p.type === PART_TYPE_TEXT && p.text)
        .map((p) => p.text || '');
      if (textParts.length > 0) {
        result.push({ role: 'assistant', content: textParts.join('\n') });
      }

      // Add tool results as tool messages
      for (const part of parts) {
        if (part.type === PART_TYPE_TOOL && part.state?.status === 'completed') {
          result.push({
            role: 'tool',
            tool_call_id: part.callID,
            content: part.state.output || '',
          });
        }
      }
    }
  }

  return result;
}

/**
 * Build token usage object for LLM spans
 */
function buildTokenUsage(tokens: MessageInfo['tokens']): Record<string, number> | null {
  if (!tokens) {
    return null;
  }

  const inputTokens = tokens.input || 0;
  const outputTokens = tokens.output || 0;
  const reasoningTokens = tokens.reasoning || 0;

  const usage: Record<string, number> = {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: inputTokens + outputTokens + reasoningTokens,
  };

  // Add cache info if available
  const cache = tokens.cache;
  if (cache) {
    if (cache.read) {
      usage.cache_read_tokens = cache.read;
    }
    if (cache.write) {
      usage.cache_write_tokens = cache.write;
    }
  }

  return usage;
}

/**
 * Create LLM and tool spans as children of the parent span
 */
function createLlmAndToolSpans(
  parentSpan: ReturnType<typeof startSpan>,
  messages: Message[],
  startIdx: number,
): void {
  let llmCallNum = 0;

  for (let i = startIdx; i < messages.length; i++) {
    const msg = messages[i];
    if (msg.info?.role !== MESSAGE_ROLE_ASSISTANT) {
      continue;
    }

    const parts = msg.parts || [];
    const modelId = msg.info?.modelID || 'unknown';
    const providerId = msg.info?.providerID || 'unknown';
    const tokens = msg.info?.tokens;

    // Get timing from message
    const timeInfo = msg.info?.time || {};
    const createdNs = timestampToNs(timeInfo.created);
    const completedNs = timestampToNs(timeInfo.completed);

    // Check for text and tool content
    const textParts = parts.filter((p) => p.type === PART_TYPE_TEXT);
    const toolParts = parts.filter((p) => p.type === PART_TYPE_TOOL);

    // Create LLM span for text responses
    if (textParts.length > 0) {
      llmCallNum++;
      const conversationMessages = reconstructConversationMessages(messages, i);
      const textContent = textParts.map((p) => p.text || '').join('\n');

      const llmSpan = startSpan({
        name: `llm_call_${llmCallNum}`,
        parent: parentSpan,
        spanType: SpanType.LLM,
        startTimeNs: createdNs,
        inputs: {
          model: `${providerId}/${modelId}`,
          messages: conversationMessages,
        },
        attributes: {
          model: modelId,
          provider: providerId,
        },
      });

      // Set token usage
      const tokenUsage = buildTokenUsage(tokens);
      if (tokenUsage) {
        llmSpan.setAttribute(SpanAttributeKey.TOKEN_USAGE, tokenUsage);
      }

      llmSpan.setOutputs({ response: textContent });
      llmSpan.end({ endTimeNs: completedNs });
    }

    // Create tool spans
    for (const toolPart of toolParts) {
      const state = toolPart.state || {};
      const toolName = toolPart.tool || 'unknown';
      const callId = toolPart.callID || '';

      // Get tool timing
      const toolTime = state.time || {};
      const toolStartNs = timestampToNs(toolTime.start);
      const toolEndNs = timestampToNs(toolTime.end);

      const toolSpan = startSpan({
        name: `tool_${toolName}`,
        parent: parentSpan,
        spanType: SpanType.TOOL,
        startTimeNs: toolStartNs,
        inputs: state.input || {},
        attributes: {
          tool_name: toolName,
          tool_id: callId,
          status: state.status || 'unknown',
        },
      });

      // Set output based on status
      if (state.status === 'completed') {
        toolSpan.setOutputs({
          result: state.output || '',
          title: state.title || '',
        });
      } else if (state.status === 'error') {
        toolSpan.setOutputs({
          error: state.error || 'Unknown error',
        });
      }

      toolSpan.end({ endTimeNs: toolEndNs });
    }
  }
}

/**
 * Process a session and create MLflow traces
 */
async function processSession(
  sessionId: string,
  sessionInfo: SessionInfo,
  messages: Message[],
  directory: string,
): Promise<void> {
  if (!messages || messages.length === 0) {
    if (DEBUG) {
      console.error('[mlflow] Empty messages list, skipping');
    }
    return;
  }

  const lastUserIdx = findLastUserMessageIndex(messages);
  if (lastUserIdx == null) {
    if (DEBUG) {
      console.error('[mlflow] No user message found in session');
    }
    return;
  }

  const userPrompt = extractUserPrompt(messages);
  if (!userPrompt) {
    if (DEBUG) {
      console.error('[mlflow] No user prompt text found');
    }
    return;
  }

  if (DEBUG) {
    console.error(`[mlflow] Creating trace for session: ${sessionId}`);
  }

  // Get timing from the FIRST message in this batch
  const firstMsg = messages[0];
  const firstMsgTime = firstMsg.info?.time || {};
  const createdNs = timestampToNs(firstMsgTime.created);

  // Get end time from the LAST message
  const lastMsg = messages[messages.length - 1];
  const lastMsgTime = lastMsg.info?.time || {};
  const updatedNs = timestampToNs(lastMsgTime.completed || lastMsgTime.created);

  // Create parent span for the conversation
  const parentSpan = startSpan({
    name: 'opencode_conversation',
    inputs: { prompt: userPrompt },
    startTimeNs: createdNs,
    spanType: SpanType.AGENT,
  });

  // Create child spans for LLM calls and tools
  createLlmAndToolSpans(parentSpan, messages, lastUserIdx + 1);

  // Get final response for preview
  const finalResponse = extractAssistantResponse(messages);

  // Set trace metadata
  try {
    const traceManager = InMemoryTraceManager.getInstance();
    const trace = traceManager.getTrace(parentSpan.traceId);
    if (trace) {
      trace.info.requestPreview = userPrompt.slice(0, MAX_PREVIEW_LENGTH);
      if (finalResponse) {
        trace.info.responsePreview = finalResponse.slice(0, MAX_PREVIEW_LENGTH);
      }
      trace.info.traceMetadata = {
        ...trace.info.traceMetadata,
        [TraceMetadataKey.TRACE_SESSION]: sessionId,
        [TraceMetadataKey.TRACE_USER]: process.env.USER || '',
        'mlflow.trace.working_directory': sessionInfo.directory || directory,
        'mlflow.trace.session_title': sessionInfo.title || '',
      };
    }
  } catch (error) {
    if (DEBUG) {
      console.error('[mlflow] Failed to update trace metadata:', error);
    }
  }

  // End parent span
  parentSpan.setOutputs({
    response: finalResponse || 'Conversation completed',
    status: 'completed',
  });
  parentSpan.end({ endTimeNs: updatedNs });

  // Flush traces to MLflow
  await flushTraces();

  if (DEBUG) {
    console.error(`[mlflow] Created trace: ${parentSpan.traceId}`);
  }
}

/**
 * MLflow tracing plugin for OpenCode.
 * Automatically traces conversations to MLflow when sessions become idle.
 */
export const MLflowTracingPlugin: Plugin = (input: PluginInput): Promise<Hooks> => {
  const client = (input as { client: PluginClient }).client;
  const directory = (input as { directory: string }).directory;

  return Promise.resolve({
    event: async ({ event }: { event: { type: string; properties?: Record<string, unknown> } }) => {
      // Only process session.idle events
      if (event.type !== 'session.idle') {
        return;
      }

      const sessionID = (event.properties as { sessionID?: string } | undefined)?.sessionID;
      if (!sessionID) {
        return;
      }

      // Initialize SDK on first use
      if (!ensureInitialized()) {
        return;
      }

      try {
        // Fetch session info and messages using the SDK client
        const sessionResult = await client.session.get({
          path: { id: sessionID },
        });
        if (!sessionResult.data) {
          if (DEBUG) {
            console.error('[mlflow] Failed to fetch session:', sessionID);
          }
          return;
        }

        const messagesResult = await client.session.messages({
          path: { id: sessionID },
          query: { limit: 1000 },
        });
        if (!messagesResult.data) {
          if (DEBUG) {
            console.error('[mlflow] Failed to fetch messages:', sessionID);
          }
          return;
        }

        // Check if we've already processed this exact turn
        const allMessages = messagesResult.data as Message[];
        const messageCount = allMessages.length;
        const lastProcessedCount = processedTurns.get(sessionID) ?? 0;

        if (messageCount <= lastProcessedCount) {
          return;
        }

        // Get only the NEW messages since last trace
        const newMessages = allMessages.slice(lastProcessedCount);
        processedTurns.set(sessionID, messageCount);

        // Clean up old entries to prevent memory leak
        if (processedTurns.size > 50) {
          const keys = Array.from(processedTurns.keys());
          for (let i = 0; i < keys.length - 50; i++) {
            processedTurns.delete(keys[i]);
          }
        }

        // Process the session
        await processSession(sessionID, sessionResult.data as SessionInfo, newMessages, directory);
      } catch (error) {
        if (DEBUG) {
          console.error('[mlflow] Error processing session:', error);
        }
      }
    },
  });
};

export default MLflowTracingPlugin;
