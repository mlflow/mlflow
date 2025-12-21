// Type definitions for OpenTelemetry GenAI semantic convention input messages
// Schema reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-input-messages.json

export type OtelBasePart = { type: string } & Record<string, unknown>;

export type OtelTextPart = OtelBasePart & { type: 'text'; content: string };

export type OtelToolCallRequestPart = OtelBasePart & {
  type: 'tool_call' | 'function_call';
  id?: string | null;
  name: string;
  arguments?: unknown;
};

export type OtelToolCallResponsePart = OtelBasePart & {
  type: 'tool_call_response';
  id?: string | null;
  response: unknown;
};

export type SupportedOtelPart = OtelTextPart | OtelToolCallRequestPart | OtelToolCallResponsePart;

export type OtelGenAIMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool';
  name?: string;
  parts: SupportedOtelPart[];
};
