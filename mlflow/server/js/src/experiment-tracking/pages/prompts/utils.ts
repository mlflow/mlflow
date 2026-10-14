import type { KeyValueEntity } from '@mlflow/mlflow/src/common/types';
import { parseJSONSafe } from '@mlflow/mlflow/src/common/utils/TagUtils';
import { isEqual } from 'lodash';
import type {
  ChatPromptMessage,
  EditableSchemaProperty,
  PromptModelConfig,
  PromptModelConfigFormData,
  RegisteredPrompt,
  RegisteredPromptVersion,
  SchemaProperty,
  SchemaScalarType,
} from './types';
import { SCHEMA_SCALAR_TYPES } from './types';
import { MLFLOW_LINKED_PROMPTS_TAG } from '../../constants';

export const REGISTERED_PROMPT_CONTENT_TAG_KEY = 'mlflow.prompt.text';
// Tag key used to store the run ID associated with a single prompt version
export const REGISTERED_PROMPT_SOURCE_RUN_ID = 'mlflow.prompt.run_id';
// Tak key used to store comma-separated run IDs associated with a prompt
export const REGISTERED_PROMPT_SOURCE_RUN_IDS = 'mlflow.prompt.run_ids';
export const IS_PROMPT_TAG_NAME = 'mlflow.prompt.is_prompt';
export const IS_PROMPT_TAG_VALUE = 'true';
export const PROMPT_TYPE_TEXT = 'text' as const;
export const PROMPT_TYPE_CHAT = 'chat' as const;
export const PROMPT_TYPE_TAG_KEY = '_mlflow_prompt_type';
// Tag key used to store comma-separated experiment IDs associated with a prompt
export const PROMPT_EXPERIMENT_IDS_TAG_KEY = '_mlflow_experiment_ids';
// Tag key used to store model config as JSON string (must match backend)
export const PROMPT_MODEL_CONFIG_TAG_KEY = '_mlflow_prompt_model_config';
// Tag key used to store response format / structured output schema (must match backend)
export const RESPONSE_FORMAT_TAG_KEY = '_mlflow_prompt_response_format';

export const MODEL_CONFIG_FIELD_LABELS: Record<Exclude<keyof PromptModelConfig, 'extra_params'>, string> = {
  provider: 'Provider',
  model_name: 'Model',
  temperature: 'Temperature',
  max_tokens: 'Max Tokens',
  top_p: 'Top P',
  top_k: 'Top K',
  frequency_penalty: 'Frequency Penalty',
  presence_penalty: 'Presence Penalty',
  stop_sequences: 'Stop Sequences',
} as const;

// Query parameter name for specifying prompt version in URLs
export const PROMPT_VERSION_QUERY_PARAM = 'promptVersion';

export type PromptsTableMetadata = {
  onEditTags: (editedEntity: RegisteredPrompt) => void;
  experimentId?: string;
};
export type PromptsVersionsTableMetadata = {
  showEditAliasesModal: (versionNumber: string) => void;
  aliasesByVersion: Record<string, string[]>;
  registeredPrompt: RegisteredPrompt;
};

export enum PromptVersionsTableMode {
  PREVIEW = 'preview',
  COMPARE = 'compare',
  TRACES = 'traces',
}

export const getPromptContentTagValue = (promptVersion: RegisteredPromptVersion) => {
  return promptVersion?.tags?.find((tag) => tag.key === REGISTERED_PROMPT_CONTENT_TAG_KEY)?.value;
};

export const isChatPrompt = (promptVersion?: RegisteredPromptVersion): boolean => {
  const tagValue = promptVersion?.tags?.find((tag) => tag.key === PROMPT_TYPE_TAG_KEY)?.value;
  return tagValue === PROMPT_TYPE_CHAT;
};

const isPromptChatMessage = (value: any): value is ChatPromptMessage => {
  return value && typeof value === 'object' && typeof value.role === 'string' && typeof value.content === 'string';
};

const isPromptChatMessages = (value: unknown): value is ChatPromptMessage[] => {
  return Array.isArray(value) && value.every((item) => isPromptChatMessage(item));
};

export const getChatPromptMessagesFromValue = (value?: string) => {
  if (!value) {
    return undefined;
  }

  const parsedValue = parseJSONSafe(value);
  if (isPromptChatMessages(parsedValue)) {
    return parsedValue;
  }

  return undefined;
};

export const parseLinkedPromptsFromRunTags = (
  tags: Record<string, KeyValueEntity>,
): { name: string; version: string }[] => {
  const linkedPromptsTag = tags[MLFLOW_LINKED_PROMPTS_TAG];
  if (!linkedPromptsTag || !linkedPromptsTag.value) {
    return [];
  }

  try {
    const promptVersions = JSON.parse(linkedPromptsTag.value);
    if (Array.isArray(promptVersions)) {
      return promptVersions.filter(
        (prompt: any) => prompt && typeof prompt.name === 'string' && typeof prompt.version === 'string',
      );
    }
  } catch (error) {
    // do nothing, just don't display any linked prompts
  }

  return [];
};

// Re-export from shared utility for backwards compatibility
export { buildSearchFilterClause } from '@mlflow/mlflow/src/common/utils/SearchUtils';

/**
 * Parse model config from tag value (JSON string).
 * Returns undefined if tag doesn't exist or JSON parsing fails.
 */
export const getModelConfigFromTags = (tags?: KeyValueEntity[]): PromptModelConfig | undefined => {
  const configTag = tags?.find((tag) => tag.key === PROMPT_MODEL_CONFIG_TAG_KEY);
  if (!configTag?.value) {
    return undefined;
  }

  try {
    return JSON.parse(configTag.value) as PromptModelConfig;
  } catch (error) {
    return undefined;
  }
};

/**
 * Convert form data to backend model config format.
 * Returns undefined if no fields have values.
 */
export const formDataToModelConfig = (formData: PromptModelConfigFormData): PromptModelConfig | undefined => {
  const hasAnyValue = Object.values(formData).some((v) => Boolean(v));
  if (!hasAnyValue) {
    return undefined;
  }

  const config: PromptModelConfig = {};

  const provider = formData.provider?.trim();
  if (provider) {
    config.provider = provider;
  }
  const modelName = formData.modelName?.trim();
  if (modelName) {
    config.model_name = modelName;
  }
  if (formData.temperature?.trim()) {
    const temp = parseFloat(formData.temperature);
    if (!isNaN(temp)) config.temperature = temp;
  }
  if (formData.maxTokens?.trim()) {
    const tokens = parseInt(formData.maxTokens, 10);
    if (!isNaN(tokens)) config.max_tokens = tokens;
  }
  if (formData.topP?.trim()) {
    const topP = parseFloat(formData.topP);
    if (!isNaN(topP)) config.top_p = topP;
  }
  if (formData.topK?.trim()) {
    const topK = parseInt(formData.topK, 10);
    if (!isNaN(topK)) config.top_k = topK;
  }
  if (formData.frequencyPenalty?.trim()) {
    const freqPenalty = parseFloat(formData.frequencyPenalty);
    if (!isNaN(freqPenalty)) config.frequency_penalty = freqPenalty;
  }
  if (formData.presencePenalty?.trim()) {
    const presPenalty = parseFloat(formData.presencePenalty);
    if (!isNaN(presPenalty)) config.presence_penalty = presPenalty;
  }
  if (formData.stopSequences?.trim()) {
    // Split by comma and trim each value
    const sequences = formData.stopSequences
      .split(',')
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    if (sequences.length > 0) config.stop_sequences = sequences;
  }

  return Object.keys(config).length > 0 ? config : undefined;
};

/**
 * Convert backend model config to form data format.
 * Returns empty object if config is undefined.
 */
export const modelConfigToFormData = (config?: PromptModelConfig): PromptModelConfigFormData => {
  if (!config) {
    return {};
  }

  return {
    provider: config.provider ?? '',
    modelName: config.model_name ?? '',
    temperature: config.temperature !== undefined ? String(config.temperature) : '',
    maxTokens: config.max_tokens !== undefined ? String(config.max_tokens) : '',
    topP: config.top_p !== undefined ? String(config.top_p) : '',
    topK: config.top_k !== undefined ? String(config.top_k) : '',
    frequencyPenalty: config.frequency_penalty !== undefined ? String(config.frequency_penalty) : '',
    presencePenalty: config.presence_penalty !== undefined ? String(config.presence_penalty) : '',
    stopSequences: config.stop_sequences ? config.stop_sequences.join(', ') : '',
  };
};

/**
 * Validate model config form values.
 * Returns an object with field names as keys and error messages as values.
 */
export const validateModelConfig = (formData: PromptModelConfigFormData): Record<string, string> => {
  const errors: Record<string, string> = {};

  if (formData['temperature']?.trim()) {
    const temp = parseFloat(formData['temperature']);
    if (isNaN(temp) || temp < 0) {
      errors['temperature'] = 'Temperature must be a number >= 0';
    }
  }

  if (formData['maxTokens']?.trim()) {
    const tokens = parseInt(formData['maxTokens'], 10);
    if (isNaN(tokens) || tokens <= 0) {
      errors['maxTokens'] = 'Max tokens must be an integer > 0';
    }
  }

  if (formData['topP']?.trim()) {
    const topP = parseFloat(formData['topP']);
    if (isNaN(topP) || topP < 0 || topP > 1) {
      errors['topP'] = 'Top P must be a number between 0 and 1';
    }
  }

  if (formData['topK']?.trim()) {
    const topK = parseInt(formData['topK'], 10);
    if (isNaN(topK) || topK <= 0) {
      errors['topK'] = 'Top K must be an integer > 0';
    }
  }

  if (formData['frequencyPenalty']?.trim()) {
    const freqPenalty = parseFloat(formData['frequencyPenalty']);
    if (isNaN(freqPenalty) || freqPenalty < -2 || freqPenalty > 2) {
      errors['frequencyPenalty'] = 'Frequency penalty must be a number between -2 and 2';
    }
  }

  if (formData['presencePenalty']?.trim()) {
    const presPenalty = parseFloat(formData['presencePenalty']);
    if (isNaN(presPenalty) || presPenalty < -2 || presPenalty > 2) {
      errors['presencePenalty'] = 'Presence penalty must be a number between -2 and 2';
    }
  }

  return errors;
};

/**
 * Parse response format (structured output schema) from version tags.
 * Returns the raw JSON string if the tag exists, or undefined if not.
 */
export const getResponseFormatFromTags = (tags?: KeyValueEntity[]): string | undefined => {
  const tag = tags?.find((t) => t.key === RESPONSE_FORMAT_TAG_KEY);
  return tag?.value || undefined;
};

/**
 * Validate that a string is valid JSON and parses to an object (JSON schema).
 * Returns { valid: true } or { valid: false, error: string }.
 */
export const validateResponseFormatJson = (value: string): { valid: boolean; error?: string } => {
  const trimmed = value?.trim();
  if (!trimmed) {
    return { valid: true };
  }
  try {
    const parsed = JSON.parse(trimmed);
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return {
        valid: false,
        error: 'Structured output must be a JSON object (e.g. a JSON schema).',
      };
    }
    return { valid: true };
  } catch (e) {
    const message = e instanceof Error ? e.message : 'Invalid JSON';
    return { valid: false, error: message };
  }
};

let idCounter = 0;
export const nextSchemaPropertyId = () => `prop_${Date.now()}_${idCounter++}`;

// Keywords read by name are listed so they can be accessed with dot notation; all stay `unknown`.
interface JsonSchema {
  type?: unknown;
  items?: unknown;
  enum?: unknown;
  properties?: unknown;
  required?: unknown;
  propertyOrdering?: unknown;
  [keyword: string]: unknown;
}

const isPlainObject = (value: unknown): value is JsonSchema =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

// Keywords the visual editor owns; everything else on a definition is carried through on merge.
const EDITOR_OWNED_KEYWORDS = ['type', 'items', 'enum'];

const UNSUPPORTED_KEYWORDS = ['$ref', 'oneOf', 'anyOf', 'allOf', 'not'];

const ARRAY_ONLY_KEYWORDS = [
  'minItems',
  'maxItems',
  'uniqueItems',
  'contains',
  'minContains',
  'maxContains',
  'prefixItems',
  'unevaluatedItems',
];

// Annotations describe the field itself, so they stay on the property when it becomes an array.
const ANNOTATION_KEYWORDS = ['description', 'title', 'deprecated', 'readOnly', 'writeOnly', '$comment'];

const STRING_KEYWORDS = ['minLength', 'maxLength', 'pattern', 'format', 'contentEncoding', 'contentMediaType'];
const NUMERIC_KEYWORDS = ['minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf', 'format'];
// Keywords holding a value of the property's own type.
const VALUE_KEYWORDS = ['default', 'examples', 'const'];

const ENUM_VALUE_JS_TYPES = ['string', 'number', 'boolean'];

// Stricter than `Number()`, which also accepts hex, binary and blank strings.
const JSON_NUMBER_PATTERN = /^-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?$/;

const keywordsForType = (type: unknown): string[] => {
  if (type === 'string') return STRING_KEYWORDS;
  if (type === 'number' || type === 'integer') return NUMERIC_KEYWORDS;
  return [];
};

// Keywords that no longer fit once a value changes type; number <-> integer keep their constraints.
const staleTypeKeywords = (from: unknown, to: unknown): string[] => {
  if (from === undefined || from === to) return [];
  const fromKeywords = keywordsForType(from);
  return [...(keywordsForType(to) === fromKeywords ? [] : fromKeywords), ...VALUE_KEYWORDS];
};

// Value keywords holding anything the edited enum no longer allows; enum values are scalars, so nesting is flattened.
const valuesOutsideEnum = (def: JsonSchema, allowed: unknown[]): string[] =>
  VALUE_KEYWORDS.filter(
    (keyword) => keyword in def && ![def[keyword]].flat(Infinity).every((v) => allowed.includes(v)),
  );

const omitKeywords = (def: JsonSchema, keywords: string[]): JsonSchema =>
  Object.fromEntries(Object.entries(def).filter(([key]) => !keywords.includes(key)));

const pickKeywords = (def: JsonSchema, keywords: string[]): JsonSchema =>
  Object.fromEntries(Object.entries(def).filter(([key]) => keywords.includes(key)));

const isScalarType = (type: unknown): type is SchemaScalarType =>
  SCHEMA_SCALAR_TYPES.includes(type as SchemaScalarType);

const isRepresentableValue = (def: unknown): def is JsonSchema => {
  if (!isPlainObject(def)) return false;
  if (UNSUPPORTED_KEYWORDS.some((keyword) => keyword in def)) return false;
  if (Array.isArray(def.enum)) {
    return def.enum.every((value) => ENUM_VALUE_JS_TYPES.includes(typeof value));
  }
  return isScalarType(def.type);
};

const readOnlyLabelOf = (def: unknown): string => {
  if (!isPlainObject(def)) return 'unsupported';
  const keyword = UNSUPPORTED_KEYWORDS.find((k) => k in def);
  if (keyword) return keyword;
  if (typeof def.type === 'string') return def.type;
  if (Array.isArray(def.type)) return def.type.join(' | ');
  return 'unsupported';
};

const enumValueTypeOf = (def: JsonSchema): SchemaScalarType => {
  if (isScalarType(def.type)) return def.type;
  const values: unknown[] = Array.isArray(def.enum) ? def.enum : [];
  if (values.length > 0 && values.every((value) => typeof value === 'boolean')) return 'boolean';
  if (values.length > 0 && values.every((value) => typeof value === 'number')) {
    return values.every((value) => Number.isInteger(value)) ? 'integer' : 'number';
  }
  return 'string';
};

/**
 * Whether the visual editor can be opened on a schema at all. It edits the `properties` map of a
 * single object schema; anything else (an OpenAI `response_format` wrapper, a `$ref` or combinator
 * at the root, an array schema) would show no fields and get `properties` attached at the wrong level.
 */
export const canEditSchemaVisually = (schemaText: string): boolean => {
  if (!schemaText?.trim()) return true;
  const parsed = parseJSONSafe(schemaText);
  if (!isPlainObject(parsed)) return false;
  if (Object.keys(parsed).length === 0) return true;
  if (UNSUPPORTED_KEYWORDS.some((keyword) => keyword in parsed)) return false;
  if (parsed.type !== undefined && parsed.type !== 'object') return false;
  return parsed.type === 'object' || isPlainObject(parsed.properties);
};

const toSchemaProperty = (id: string, name: string, def: unknown, required: boolean): SchemaProperty => {
  const common = { id, name, originalName: name, required };
  const container = isPlainObject(def) && def.type === 'array' ? def : undefined;
  const value = container ? (container.items ?? {}) : def;
  const unsupportedContainer = container !== undefined && UNSUPPORTED_KEYWORDS.some((k) => k in container);

  if (unsupportedContainer || !isRepresentableValue(value)) {
    return { ...common, kind: 'readOnly', label: readOnlyLabelOf(def) };
  }
  const isArray = container !== undefined;
  if (Array.isArray(value.enum)) {
    return {
      ...common,
      kind: 'editable',
      type: 'enum',
      isArray,
      enumValues: value.enum.map(String),
      enumValueType: enumValueTypeOf(value),
    };
  }
  return { ...common, kind: 'editable', type: value.type as SchemaScalarType, isArray, enumValues: [] };
};

/**
 * Parse a JSON schema string into a flat list of properties for the visual editor. Returns an
 * empty list if the schema is empty or fails to parse. Definitions outside the editor's model
 * become read-only rows.
 */
export const jsonSchemaToProperties = (schemaText: string): SchemaProperty[] => {
  const schema = schemaText?.trim() ? parseJSONSafe(schemaText) : undefined;
  if (!isPlainObject(schema)) return [];

  const props = isPlainObject(schema.properties) ? schema.properties : {};
  const required: unknown[] = Array.isArray(schema.required) ? schema.required : [];
  const names = Object.keys(props);
  // Gemini's `propertyOrdering` may list only some properties, so it only sorts them.
  const ordering = (Array.isArray(schema.propertyOrdering) ? schema.propertyOrdering : []).filter(
    (name): name is string => typeof name === 'string' && names.includes(name),
  );
  const order = [...ordering, ...names.filter((name) => !ordering.includes(name))];

  return order.map((name) => toSchemaProperty(nextSchemaPropertyId(), name, props[name], required.includes(name)));
};

// Re-emit enum values in their original JSON type; undefined when a value no longer fits it.
const coerceEnumValues = (values: string[], valueType: SchemaScalarType): JsonSchema | undefined => {
  if (valueType === 'boolean') {
    const coerced = values.map((value) => (value === 'true' ? true : value === 'false' ? false : undefined));
    return coerced.every((value) => value !== undefined) ? { type: 'boolean', enum: coerced } : undefined;
  }
  if (valueType === 'number' || valueType === 'integer') {
    const coerced = values.map(Number);
    const fits = values.every(
      (value, index) =>
        JSON_NUMBER_PATTERN.test(value) &&
        Number.isFinite(coerced[index]) &&
        (valueType === 'number' || Number.isInteger(coerced[index])),
    );
    return fits ? { type: valueType, enum: coerced } : undefined;
  }
  return undefined;
};

const editedValueDefinition = (property: EditableSchemaProperty): JsonSchema => {
  if (property.type !== 'enum') {
    return { type: property.type };
  }
  const valueType = property.enumValueType ?? 'string';
  // JSON Schema requires unique `enum` values and rejects an empty array.
  const values = Array.from(new Set(property.enumValues.map((value) => value.trim()).filter((value) => value)));
  if (values.length === 0) {
    return { type: valueType };
  }
  return coerceEnumValues(values, valueType) ?? { type: 'string', enum: values };
};

const withTypeFirst = (def: JsonSchema): JsonSchema => ({ type: def.type, ...def });

const mergeDefinition = (property: EditableSchemaProperty, sourceDef: JsonSchema | undefined): JsonSchema => {
  const edited = editedValueDefinition(property);
  if (!sourceDef) {
    return property.isArray ? { type: 'array', items: edited } : edited;
  }

  const sourceIsArray = sourceDef.type === 'array';
  const sourceValue = sourceIsArray ? (isPlainObject(sourceDef.items) ? sourceDef.items : {}) : sourceDef;
  const staleKeywords = staleTypeKeywords(sourceValue.type, edited.type);
  const outsideEnum = (def: JsonSchema) => (Array.isArray(edited.enum) ? valuesOutsideEnum(def, edited.enum) : []);
  const valueExtras = omitKeywords(sourceValue, [
    ...EDITOR_OWNED_KEYWORDS,
    ...staleKeywords,
    ...outsideEnum(sourceValue),
  ]);
  // The container's own `default` / `examples` are arrays of the old items, so they only survive an unchanged array.
  const staleContainerKeywords =
    property.isArray && staleKeywords.length === 0 ? outsideEnum(sourceDef) : VALUE_KEYWORDS;
  const containerExtras = sourceIsArray
    ? omitKeywords(sourceDef, [...EDITOR_OWNED_KEYWORDS, ...staleContainerKeywords])
    : {};

  if (property.isArray) {
    // Scalar -> array: value constraints move into `items`, annotations stay on the property.
    const annotations = sourceIsArray ? {} : pickKeywords(valueExtras, ANNOTATION_KEYWORDS);
    const itemExtras = sourceIsArray ? valueExtras : omitKeywords(valueExtras, ANNOTATION_KEYWORDS);
    return withTypeFirst({
      ...containerExtras,
      ...annotations,
      type: 'array',
      items: withTypeFirst({ ...itemExtras, ...edited }),
    });
  }
  return withTypeFirst({ ...valueExtras, ...omitKeywords(containerExtras, ARRAY_ONLY_KEYWORDS), ...edited });
};

const isUnedited = (property: EditableSchemaProperty, sourceDef: JsonSchema): boolean => {
  const parsed = toSchemaProperty(property.id, property.name, sourceDef, property.required);
  return (
    parsed.kind === 'editable' &&
    parsed.type === property.type &&
    parsed.isArray === property.isArray &&
    (property.type !== 'enum' || isEqual(parsed.enumValues, property.enumValues))
  );
};

/**
 * Convert the editor's properties back into a JSON schema string, merged onto `baseSchema` (the
 * parsed schema the editor was opened with) so that everything the flat editor does not model is
 * preserved. Properties with an empty name are skipped.
 */
export const propertiesToJsonSchema = (properties: SchemaProperty[], baseSchema?: unknown): string => {
  const base = isPlainObject(baseSchema) ? baseSchema : {};
  const baseProperties = isPlainObject(base.properties) ? base.properties : {};
  const validProps = properties.filter((p) => p.name.trim());

  const definitionFor = (property: SchemaProperty): unknown => {
    // By original name, so a rename keeps its keywords and a new row never inherits a deleted namesake's.
    const source = property.originalName === undefined ? undefined : baseProperties[property.originalName];
    if (property.kind === 'readOnly') {
      return source === undefined ? {} : source;
    }
    const sourceDef = isPlainObject(source) ? source : undefined;
    // Untouched rows go out verbatim so that editing one field never normalizes another.
    if (sourceDef && isUnedited(property, sourceDef)) {
      return sourceDef;
    }
    return mergeDefinition(property, sourceDef);
  };

  const names = validProps.map((p) => p.name.trim());
  // Duplicate names collapse into one property, and JSON Schema rejects repeated `required` entries.
  const requiredNames = Array.from(new Set(validProps.filter((p) => p.required).map((p) => p.name.trim())));

  const schema: JsonSchema = { type: 'object', ...base };
  schema.properties = Object.fromEntries(validProps.map((p, index) => [names[index], definitionFor(p)]));
  if (requiredNames.length > 0) {
    schema.required = requiredNames;
  } else {
    delete schema.required;
  }
  // Kept in sync when the source schema already used it; never introduced.
  if (Array.isArray(base.propertyOrdering)) {
    schema.propertyOrdering = names;
  }

  return JSON.stringify(schema, null, 2);
};
