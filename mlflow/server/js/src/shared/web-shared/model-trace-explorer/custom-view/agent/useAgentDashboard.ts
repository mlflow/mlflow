import { fetchOrFail } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import type { A2uiMessage } from '@a2ui/web_core/v0_9';

import { type AgentTraceData, buildAgentMessages } from './buildAgentPrompt';
import { validateAndPrepareMessages } from './validateA2uiMessages';
import { useMutation } from '../../../query-client/queryClient';

// The unified, OpenAI-compatible gateway chat completions route (same one the
// gateway "Try it" panel posts to). The endpoint is selected via the `model`
// field in the body.
const getChatCompletionsUrl = (): string => {
  const origin = typeof window !== 'undefined' ? window.location.origin : '';
  return `${origin}/gateway/mlflow/v1/chat/completions`;
};

// Pulls the JSON payload out of the model's reply, tolerating ```json fences,
// bare ``` fences, or a raw JSON body.
const extractJsonText = (content: string): string => {
  const trimmed = content.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced?.[1]) {
    return fenced[1].trim();
  }
  return trimmed;
};

type ChatCompletionResponse = {
  choices?: { message?: { content?: string } }[];
};

export type GenerateParams = {
  instruction: string;
  endpointName: string;
  surfaceId: string;
  catalogId: string;
  data: AgentTraceData;
};

/**
 * Generates an A2UI dashboard block via an LLM. Builds the prompt, posts to the
 * selected gateway endpoint, extracts the model's JSON, then validates and
 * normalizes it into a processor-ready `A2uiMessage[]` (host-owned surface id).
 * Throws a descriptive Error on any failure so the UI can surface it.
 */
export const useAgentDashboard = () => {
  const mutation = useMutation<A2uiMessage[], Error, GenerateParams>({
    mutationFn: async ({ instruction, endpointName, surfaceId, catalogId, data }) => {
      const messages = buildAgentMessages({ instruction, data });

      // Bound the request so a slow/hung gateway surfaces an error instead of
      // spinning indefinitely. Large traces make the prompt big and generation
      // slow, so allow a generous window before giving up.
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120_000);
      let response: Response;
      try {
        response = await fetchOrFail(getChatCompletionsUrl(), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: endpointName, messages }),
          signal: controller.signal,
        });
      } catch (error) {
        if (controller.signal.aborted) {
          throw new Error('The request timed out after 120s. The trace may be too large — try a more specific instruction.');
        }
        throw error;
      } finally {
        clearTimeout(timeoutId);
      }

      const body = (await response.json()) as ChatCompletionResponse;
      const content = body.choices?.[0]?.message?.content;
      if (!content || typeof content !== 'string') {
        throw new Error('The model returned an empty response.');
      }

      let parsed: unknown;
      try {
        parsed = JSON.parse(extractJsonText(content));
      } catch {
        throw new Error('The model did not return valid JSON.');
      }

      const result = validateAndPrepareMessages(parsed, { surfaceId, catalogId });
      if (!result.ok) {
        throw new Error(result.error);
      }
      return result.messages;
    },
  });

  return {
    generate: mutation.mutateAsync,
    isLoading: mutation.isLoading,
    error: mutation.error ?? undefined,
    reset: mutation.reset,
  };
};
