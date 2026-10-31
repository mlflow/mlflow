import { isNil, isString } from 'lodash';

import type { ModelTraceChatTool } from '../ModelTrace.types';
import { isModelTraceChatTool } from '../ModelTraceExplorer.utils';

/**
 * Framework-agnostic extraction of tool *definitions* from span request inputs.
 *
 * Unlike chat messages (normalized per-format in the sibling chat-utils modules), tool
 * definitions share a recognizable shape across providers: an object carrying a string name and a
 * JSON-schema-ish parameters object under one of a few well-known keys. Detecting that shape —
 * rather than dispatching on the framework — means any integration's request format is supported,
 * including ones we have never seen:
 *
 *  - OpenAI-compatible (OpenAI, Mistral, DeepSeek, Groq, LiteLLM, the MLflow gateway, ...):
 *      { type: 'function', function: { name, description?, parameters? } }
 *  - Anthropic: { name, description?, input_schema }
 *  - Google Gemini: { function_declarations: [{ name, description?, parameters }] }
 *  - Amazon Bedrock Converse: { toolConfig: { tools: [{ toolSpec: { name, inputSchema: { json } } }] } }
 *  - PydanticAI: { model_request_parameters: { function_tools: [{ name, description?, parameters_json_schema }] } }
 *
 * Tool *calls* never match: they carry serialized `arguments` / `args` rather than a schema
 * object, and the matcher requires a schema (or the explicit OpenAI `function` envelope).
 */

type UnknownRecord = Record<string, unknown>;

const MAX_SEARCH_DEPTH = 8;
const MAX_TOOLS = 50;

// Keys different providers use for a tool's JSON-schema parameters. NB: a bare `schema` key is
// deliberately excluded — response_format envelopes ({ json_schema: { name, schema } }) would
// otherwise be misdetected as tools.
const SCHEMA_KEYS = ['parameters', 'input_schema', 'parameters_json_schema', 'inputSchema', 'json_schema'];

const asRecord = (value: unknown): UnknownRecord | null =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as UnknownRecord) : null;

const isSchemaLike = (value: unknown): boolean => {
  const record = asRecord(value);
  if (!record) {
    return false;
  }
  return (
    asRecord(record['properties']) !== null ||
    record['type'] === 'object' ||
    Array.isArray(record['anyOf']) ||
    Array.isArray(record['oneOf']) ||
    Array.isArray(record['allOf'])
  );
};

const getParametersSchema = (record: UnknownRecord): UnknownRecord | null => {
  for (const key of SCHEMA_KEYS) {
    const value = record[key];
    if (isSchemaLike(value)) {
      return asRecord(value);
    }
    // Bedrock nests the schema one level deeper: inputSchema: { json: { ... } }
    const nestedJson = asRecord(value)?.['json'];
    if (isSchemaLike(nestedJson)) {
      return asRecord(nestedJson);
    }
  }
  return null;
};

const buildTool = (name: unknown, description: unknown, schema: UnknownRecord | null): ModelTraceChatTool | null => {
  if (!isString(name) || !name) {
    return null;
  }
  const fn = {
    name,
    ...(isString(description) && description ? { description } : {}),
  };
  if (schema) {
    const candidate = {
      type: 'function' as const,
      function: { ...fn, parameters: schema as unknown as NonNullable<ModelTraceChatTool['function']['parameters']> },
    };
    if (isModelTraceChatTool(candidate)) {
      return candidate;
    }
    // Schema present but malformed for the UI's tool renderer — keep the tool, drop the schema.
  }
  return { type: 'function', function: fn };
};

const tryExtractToolDefinition = (record: UnknownRecord): ModelTraceChatTool | null => {
  // OpenAI-style envelope: { function: { name, ... } }. A serialized `arguments` field means it
  // is a tool call, not a definition.
  const fn = asRecord(record['function']);
  if (fn && isString(fn['name']) && fn['name'] && isNil(fn['arguments'])) {
    return buildTool(fn['name'], fn['description'], getParametersSchema(fn));
  }

  // Direct style: { name, <schema key>, description? } — Anthropic tools, Gemini function
  // declarations, Bedrock toolSpec, PydanticAI function_tools, and similar. Requiring a schema
  // keeps named non-tool objects (e.g. chat messages with a `name`) from matching, and the
  // `arguments`/`args` guard excludes tool calls.
  if (isString(record['name']) && record['name'] && isNil(record['arguments']) && isNil(record['args'])) {
    const schema = getParametersSchema(record);
    if (schema) {
      return buildTool(record['name'], record['description'], schema);
    }
  }

  return null;
};

const visit = (value: unknown, depth: number, tools: ModelTraceChatTool[], seenNames: Set<string>): void => {
  if (isNil(value) || depth > MAX_SEARCH_DEPTH || tools.length >= MAX_TOOLS) {
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((item) => visit(item, depth + 1, tools, seenNames));
    return;
  }
  const record = asRecord(value);
  if (!record) {
    return;
  }
  const tool = tryExtractToolDefinition(record);
  if (tool) {
    if (!seenNames.has(tool.function.name)) {
      seenNames.add(tool.function.name);
      tools.push(tool);
    }
    // A matched definition is a leaf — don't descend into its schema.
    return;
  }
  Object.values(record).forEach((item) => visit(item, depth + 1, tools, seenNames));
};

/**
 * Extracts the tool definitions available in a span's request inputs, normalized to the OpenAI
 * function-tool shape the trace UI renders. Returns undefined when the inputs carry no tools.
 * De-duplicated by tool name; search depth and tool count are bounded.
 */
export const normalizeToolDefinitions = (inputs: unknown): ModelTraceChatTool[] | undefined => {
  const tools: ModelTraceChatTool[] = [];
  visit(inputs, 0, tools, new Set());
  return tools.length > 0 ? tools : undefined;
};
