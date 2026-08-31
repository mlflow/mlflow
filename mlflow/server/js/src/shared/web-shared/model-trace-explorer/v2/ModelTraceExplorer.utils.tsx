import {
  isNil,
  omitBy,
  mapValues,
  isArray,
  isString,
  isNumber,
  isBoolean,
  escapeRegExp,
  map,
  every,
  has,
  compact,
  keyBy,
  isEmpty,
} from 'lodash';

import Utils from '../../../../common/utils/Utils';

import type {
  SearchMatch,
  ModelTrace,
  ModelTraceSpan,
  ModelTraceSpanNode,
  ModelTraceChatMessage,
  ModelTraceChatInput,
  ModelTraceChatTool,
  ModelTraceChatToolParamProperty,
  SpanFilterState,
  ModelTraceSpanV4,
  ModelTraceSpanV3,
  ModelTraceSpanV2,
  ModelTraceInfoV3,
  Assessment,
  RetrieverDocument,
  ModelTraceEvent,
} from './ModelTrace.types';
import { ModelSpanType, ModelIconType, MLFLOW_TRACE_SCHEMA_VERSION_KEY, type SpanCostInfo } from './ModelTrace.types';
import { ModelTraceExplorerIcon } from './ModelTraceExplorerIcon';
import { parseJSONSafe } from '../TagUtils';
import { normalizeAnthropicChatInput, normalizeAnthropicChatOutput } from '../chat-utils/anthropic';
import { normalizeAutogenChatInput, normalizeAutogenChatOutput } from '../chat-utils/autogen';
import { normalizeBedrockChatInput, normalizeBedrockChatOutput } from '../chat-utils/bedrock';
import { normalizeGeminiChatInput, normalizeGeminiChatOutput } from '../chat-utils/gemini';
import {
  normalizeOpenAIChatInput,
  normalizeOpenAIChatResponse,
  normalizeOpenAIResponsesInput,
  normalizeOpenAIResponsesOutput,
  normalizeOpenAIAgentInput,
  normalizeOpenAIAgentOutput,
  normalizeOpenAIResponsesStreamingOutput,
} from '../chat-utils/openai';
import { normalizeLangchainChatInput, normalizeLangchainChatResult } from '../chat-utils/langchain';
import { normalizeLlamaIndexChatInput, normalizeLlamaIndexChatResponse } from '../chat-utils/llamaindex';
import { normalizeDspyChatInput, normalizeDspyChatOutput } from '../chat-utils/dspy';
import { normalizeVercelAIChatInput, normalizeVercelAIChatOutput } from '../chat-utils/vercelai';
import { isOtelGenAIChatMessage, normalizeOtelGenAIChatMessage } from '../chat-utils/otel';
import { normalizePydanticAIChatInput, normalizePydanticAIChatOutput } from '../chat-utils/pydanticai';
import { getSpanAttribute } from '../spanAttributeReader';
import { decodeOtelAnyValue, isOtelAnyValue } from '../../genai-traces-table/utils/TraceUtils';
import { normalizeMistralChatInput, normalizeMistralChatOutput } from '../chat-utils/mistral';
import {
  normalizeVoltAgentChatInput,
  normalizeVoltAgentChatOutput,
  synthesizeVoltAgentChatMessages,
} from '../chat-utils/voltagent';
import { prettyPrintChatMessage, prettyPrintToolCall } from '../formatChatMessage';
import {
  isModelTraceChatMessage,
  isModelTraceChatResponse,
  isModelTraceChoices,
  isModelTraceToolCall,
  isRawModelTraceChatMessage,
} from '../modelTraceChatMessageTypes';
import { getSpanExceptionCount, getSpanExceptionEvents } from '../spanExceptionEvents';
import { getSpanTokenUsage } from './ModelTraceTokenUsage.utils';
import {
  ASSESSMENT_SESSION_METADATA_KEY,
  CHUNK_INDEX_KEY,
  COST_METADATA_KEY,
  MLFLOW_SPAN_OUTPUT_KEY,
  SPAN_ATTRIBUTE_COST_KEY,
  SPAN_ATTRIBUTE_LINKED_GATEWAY_TRACE_ID_KEY,
  SPAN_ATTRIBUTE_MODEL_KEY,
  SPAN_ATTRIBUTE_TIME_TO_FIRST_TOKEN_MS_KEY,
  TOKEN_USAGE_METADATA_KEY,
} from '../constants';
export {
  isV4TraceId,
  createTraceV4SerializedLocation,
  parseTraceV4SerializedLocation,
  createTraceV4LongIdentifier,
  parseV4TraceId,
  parseV4TraceIdToObject,
} from '../traceV4Identifiers';

export const FETCH_TRACE_INFO_QUERY_KEY = 'model-trace-info-v3';

export const displayErrorNotification = (errorMessage: string): void => {
  Utils.displayGlobalErrorNotification(errorMessage);
};

export const displaySuccessNotification = (successMessage: string): void => {
  Utils.displayGlobalInfoNotification(successMessage);
};

export function getIconTypeForSpan(spanType: ModelSpanType | string): ModelIconType {
  switch (spanType) {
    case ModelSpanType.LLM:
      return ModelIconType.MODELS;
    case ModelSpanType.CHAIN:
      return ModelIconType.CHAIN;
    case ModelSpanType.AGENT:
      return ModelIconType.AGENT;
    case ModelSpanType.TOOL:
      return ModelIconType.WRENCH;
    case ModelSpanType.CHAT_MODEL:
      return ModelIconType.MODELS;
    case ModelSpanType.RETRIEVER:
      return ModelIconType.SEARCH;
    case ModelSpanType.PARSER:
      return ModelIconType.CODE;
    case ModelSpanType.EMBEDDING:
      return ModelIconType.NUMBERS;
    case ModelSpanType.RERANKER:
      return ModelIconType.SORT;
    case ModelSpanType.MEMORY:
      return ModelIconType.SAVE;
    case ModelSpanType.FUNCTION:
      return ModelIconType.FUNCTION;
    case ModelSpanType.UNKNOWN:
      return ModelIconType.UNKNOWN;
    default:
      return ModelIconType.FUNCTION;
  }
}

export function getDisplayNameForSpanType(spanType: ModelSpanType | string): string {
  switch (spanType) {
    case ModelSpanType.LLM:
      return 'LLM';
    case ModelSpanType.CHAIN:
      return 'Chain';
    case ModelSpanType.AGENT:
      return 'Agent';
    case ModelSpanType.TOOL:
      return 'Tool';
    case ModelSpanType.CHAT_MODEL:
      return 'Chat model';
    case ModelSpanType.RETRIEVER:
      return 'Retriever';
    case ModelSpanType.PARSER:
      return 'Parser';
    case ModelSpanType.EMBEDDING:
      return 'Embedding';
    case ModelSpanType.RERANKER:
      return 'Reranker';
    case ModelSpanType.MEMORY:
      return 'Memory';
    case ModelSpanType.FUNCTION:
      return 'Function';
    case ModelSpanType.UNKNOWN:
      return 'Unknown';
    default:
      return spanType;
  }
}

export function tryDeserializeAttribute(value: string): any {
  try {
    return JSON.parse(value);
  } catch (e) {
    return value;
  }
}

export const getMatchesFromEvent = (span: ModelTraceSpanNode, searchFilter: string): SearchMatch[] => {
  const events = span.events;
  if (!events) {
    return [];
  }

  const matches: SearchMatch[] = [];
  events.forEach((event, index) => {
    const attributes = event.attributes;

    if (!attributes) {
      return;
    }

    Object.keys(attributes).forEach((attribute) => {
      const isKeyMatch = attribute.toLowerCase().includes(searchFilter);
      const key = getEventAttributeKey(event.name, index, attribute);

      if (isKeyMatch) {
        matches.push({
          span,
          section: 'events',
          key,
          isKeyMatch: true,
          matchIndex: 0,
        });
      }

      // event values can be arbitrary JSON
      const value = JSON.stringify(attributes[attribute]).toLowerCase();
      const numValueMatches = value.split(searchFilter).length - 1;
      for (let i = 0; i < numValueMatches; i++) {
        matches.push({
          span,
          section: 'events',
          key,
          isKeyMatch: false,
          matchIndex: i,
        });
      }
    });
  });

  return matches;
};

/**
 * This function extracts all the matches from a span based on the search filter,
 * and appends some necessary metadata that is necessary for the jump-to-search
 * function.
 */
export const getMatchesFromSpan = (span: ModelTraceSpanNode, searchFilter: string): SearchMatch[] => {
  // if search filter is empty, don't generate matches
  // because there will be nothing to highlight anyway
  if (!searchFilter) {
    return [];
  }

  const matches: SearchMatch[] = [];

  const sections = {
    inputs: span?.inputs,
    outputs: span?.outputs,
    attributes: span?.attributes,
    events: span?.events,
  };

  map(sections, (section: any, label: 'inputs' | 'outputs' | 'attributes' | 'events') => {
    if (label === 'events') {
      matches.push(...getMatchesFromEvent(span, searchFilter));
      return;
    }

    const sectionList = createListFromObject(section);
    sectionList.forEach((item) => {
      // NOTE: this ignores the fact that there might be multiple matches in a key
      // for example, if the key is "aaaaa", and the search filter is "a". However,
      // implementing support for this case would make the code needlessly complex.
      // If we receive feedback that this is a problem, we can revisit this.
      const isKeyMatch = item.key.toLowerCase().includes(searchFilter);
      if (isKeyMatch) {
        matches.push({
          span: span,
          section: label,
          key: item.key,
          isKeyMatch: true,
          matchIndex: 0,
        });
      }

      const numValueMatches = item.value.toLowerCase().split(searchFilter).length - 1;
      for (let i = 0; i < numValueMatches; i++) {
        matches.push({
          span: span,
          section: label,
          key: item.key,
          isKeyMatch: false,
          matchIndex: i,
        });
      }
    });
  });
  return matches;
};

export function searchTree(
  rootNode: ModelTraceSpanNode,
  searchFilter: string,
  spanFilterState: SpanFilterState,
): {
  filteredTreeNodes: ModelTraceSpanNode[];
  matches: SearchMatch[];
} {
  const searchFilterLowercased = searchFilter.toLowerCase().trim();
  const allSpanTypesSelected = Object.values(spanFilterState.spanTypeDisplayState).every(
    (shouldDisplay) => shouldDisplay,
  );
  // if there is no search filter and all span types
  // are selected, then we don't have to do any filtering.
  if (searchFilterLowercased === '' && allSpanTypesSelected) {
    return {
      filteredTreeNodes: [rootNode],
      matches: [],
    };
  }

  const children = rootNode.children ?? [];
  const filteredChildren: ModelTraceSpanNode[] = [];
  const matches: SearchMatch[] = [];
  children.forEach((child) => {
    const { filteredTreeNodes: childNodes, matches: childMatches } = searchTree(
      child,
      searchFilterLowercased,
      spanFilterState,
    );

    filteredChildren.push(...childNodes);
    matches.push(...childMatches);
  });

  const spanName = ((rootNode.title as string) ?? '').toLowerCase();
  const spanMatches = getMatchesFromSpan(rootNode, searchFilterLowercased);

  // check if the span passes the text and type filters
  const nodeMatchesSearch = spanMatches.length > 0 || spanName.includes(searchFilterLowercased);
  const spanTypeIsDisplayed = rootNode.type ? spanFilterState.spanTypeDisplayState[rootNode.type] : true;
  const nodePassesSpanFilters = nodeMatchesSearch && spanTypeIsDisplayed;

  const hasMatchingChild = filteredChildren.length > 0;
  const hasException = getSpanExceptionCount(rootNode) > 0;

  const nodeShouldBeDisplayed =
    nodePassesSpanFilters ||
    // the `showParents` and `showExceptions` flags override the
    // search filters, so we always show the node if they pass
    (spanFilterState.showParents && hasMatchingChild) ||
    (spanFilterState.showExceptions && hasException);

  if (nodeShouldBeDisplayed) {
    return {
      filteredTreeNodes: [{ ...rootNode, children: filteredChildren }],
      matches: spanMatches.concat(matches),
    };
  }

  // otherwise cut the span out of the tree by returning the children directly
  return {
    filteredTreeNodes: filteredChildren,
    matches,
  };
}

export function searchTreeBySpanId(
  rootNode: ModelTraceSpanNode | null,
  selectedSpanId?: string,
): ModelTraceSpanNode | undefined {
  if (isNil(selectedSpanId) || isNil(rootNode)) {
    return undefined;
  }

  if (rootNode.key === selectedSpanId) {
    return rootNode;
  }

  const children = rootNode.children ?? [];
  for (const child of children) {
    const matchedNode = searchTreeBySpanId(child, selectedSpanId);
    if (matchedNode) {
      return matchedNode;
    }
  }

  return undefined;
}

const getChatMessagesFromSpan = (
  messagesAttributeValue: any,
  inputs: any,
  outputs: any,
  messageFormat?: string,
  children?: ModelTraceSpanNode[],
): ModelTraceChatMessage[] | undefined => {
  // if the `mlflow.chat.messages` attribute is provided
  // and in the correct format, return it as-is
  // we allow content type to be content part list for the `mlflow.chat.messages` attribute
  if (Array.isArray(messagesAttributeValue) && messagesAttributeValue.every(isRawModelTraceChatMessage)) {
    return compact(messagesAttributeValue.map(prettyPrintChatMessage));
  }

  // otherwise, attempt to parse messages from inputs and outputs
  // this is to support rich rendering for older versions of MLflow
  // before the `mlflow.chat.messages` attribute was introduced
  const messagesFromInputs = normalizeConversation(inputs, messageFormat) ?? [];
  const messagesFromOutputs = normalizeConversation(outputs, messageFormat) ?? [];

  // PydanticAI's new_messages() returns complete conversation including user prompt
  if (messageFormat === 'pydantic_ai' && messagesFromOutputs.length > 0) {
    return messagesFromInputs.length > 0 ? messagesFromInputs.concat(messagesFromOutputs) : messagesFromOutputs;
  }

  // For VoltAgent format, synthesize messages from child spans (tool executions)
  // This is necessary because VoltAgent stores tool calls as child TOOL spans
  // rather than inline in the messages array
  if (messageFormat === 'voltagent' && children && children.length > 0) {
    const synthesizedMessages = synthesizeVoltAgentChatMessages(inputs, outputs, children);
    if (synthesizedMessages && synthesizedMessages.length > 0) {
      return synthesizedMessages;
    }
  }

  // When the output is a plain string and inputs parsed as chat messages,
  // wrap the output as an assistant message so the chat UI can render.
  if (messagesFromInputs.length > 0 && messagesFromOutputs.length === 0 && typeof outputs === 'string') {
    return messagesFromInputs.concat([{ role: 'assistant', content: outputs }]);
  }

  // when either input or output is not chat messages, we do not set the chat message field.
  if (messagesFromInputs.length === 0 || messagesFromOutputs.length === 0) {
    return undefined;
  }

  // LangGraph (and similar frameworks) accumulate all messages in the output state,
  // so outputs already contain the input messages as a prefix. Detect this overlap
  // and use only the output messages to avoid duplication.
  if (
    messagesFromOutputs.length >= messagesFromInputs.length &&
    messagesFromInputs.every(
      (msg, i) => msg.role === messagesFromOutputs[i].role && msg.content === messagesFromOutputs[i].content,
    )
  ) {
    return messagesFromOutputs;
  }

  return messagesFromInputs.concat(messagesFromOutputs);
};

// Surfaces how often we receive non-conforming tools so we can spot real data-quality
// regressions instead of silently dropping them from the UI.
const logDroppedChatTools = (total: number, valid: number, source: string): void => {
  if (total > valid) {
    // eslint-disable-next-line no-console -- intentional observability log for malformed chat tools
    console.warn(`[model-trace-explorer] dropped ${total - valid}/${total} malformed chat tools from ${source}`);
  }
};

const getChatToolsFromSpan = (toolsAttributeValue: any, inputs: any): ModelTraceChatTool[] | undefined => {
  // if the `mlflow.chat.tools` attribute is provided, return the tools that
  // are in the correct format. A single non-conforming tool shouldn't drop the
  // entire list from the UI.
  if (Array.isArray(toolsAttributeValue)) {
    const validTools = toolsAttributeValue.filter(isModelTraceChatTool);
    logDroppedChatTools(toolsAttributeValue.length, validTools.length, 'mlflow.chat.tools');
    if (validTools.length > 0) {
      return validTools;
    }
  }

  // otherwise, attempt to parse tools from inputs
  // TODO: support langchain format for tool inputs
  if (Array.isArray(inputs?.tools)) {
    const validTools = inputs.tools.filter(isModelTraceChatTool);
    logDroppedChatTools(inputs.tools.length, validTools.length, 'inputs.tools');
    if (validTools.length > 0) {
      return validTools;
    }
  }

  return undefined;
};

const getCostFromSpan = (costAttributeValue: any): SpanCostInfo | undefined => {
  if (
    costAttributeValue &&
    typeof costAttributeValue === 'object' &&
    'input_cost' in costAttributeValue &&
    'output_cost' in costAttributeValue &&
    'total_cost' in costAttributeValue
  ) {
    return costAttributeValue as SpanCostInfo;
  }
  return undefined;
};

export const normalizeNewSpanData = (
  span: ModelTraceSpan,
  rootStartTime: number,
  rootEndTime: number,
  children: ModelTraceSpanNode[],
  assessmentMap: Record<string, Assessment[]>,
  traceId: string,
): ModelTraceSpanNode => {
  const spanType = tryDeserializeAttribute(getSpanAttribute(span.attributes, 'mlflow.spanType') as string);
  const inputs = tryDeserializeAttribute(getSpanAttribute(span.attributes, 'mlflow.spanInputs') as string);
  const outputs = tryDeserializeAttribute(getSpanAttribute(span.attributes, 'mlflow.spanOutputs') as string);
  const parentId = getModelTraceSpanParentId(span);
  const spanId = getModelTraceSpanId(span);

  const assessments = assessmentMap[spanId] ?? [];
  if (!parentId) {
    // assessments that are not associated with a specific
    // span should be displayed at the root.
    assessments.push(...(assessmentMap[''] ?? []));
  }

  // data that powers the "chat" tab
  const messagesAttributeValue = tryDeserializeAttribute(
    getSpanAttribute(span.attributes, 'mlflow.chat.messages') as string,
  );
  const messageFormat = tryDeserializeAttribute(getSpanAttribute(span.attributes, 'mlflow.message.format') as string);
  const chatMessages = getChatMessagesFromSpan(messagesAttributeValue, inputs, outputs, messageFormat, children);
  const chatTools = getChatToolsFromSpan(
    tryDeserializeAttribute(getSpanAttribute(span.attributes, 'mlflow.chat.tools') as string),
    inputs,
  );

  // Extract model name, cost info, and linked gateway trace ID
  const modelName = tryDeserializeAttribute(getSpanAttribute(span.attributes, SPAN_ATTRIBUTE_MODEL_KEY) as string);
  const cost = getCostFromSpan(
    tryDeserializeAttribute(getSpanAttribute(span.attributes, SPAN_ATTRIBUTE_COST_KEY) as string),
  );
  const linkedGatewayTraceId = tryDeserializeAttribute(
    getSpanAttribute(span.attributes, SPAN_ATTRIBUTE_LINKED_GATEWAY_TRACE_ID_KEY) as string,
  );

  // remove other private mlflow attributes
  const attributes = mapValues(
    omitBy(span.attributes, (_, key) => key.startsWith('mlflow.')),
    (value) => tryDeserializeAttribute(value),
  );
  const events = span.events;
  const start = (Number(getModelTraceSpanStartTime(span)) - rootStartTime) / 1000;
  const end = (Number(getModelTraceSpanEndTime(span) ?? rootEndTime) - rootStartTime) / 1000;

  return {
    title: span.name,
    icon: (
      <ModelTraceExplorerIcon
        type={getIconTypeForSpan(spanType)}
        hasException={getSpanExceptionCount(span) > 0}
        isRootSpan={!parentId}
      />
    ),
    type: spanType,
    key: spanId,
    start,
    end,
    children: children.slice().sort((a, b) => a.start - b.start),
    inputs,
    outputs,
    attributes,
    events,
    chatMessageFormat: messageFormat,
    chatMessages,
    chatTools,
    parentId,
    assessments,
    traceId,
    modelName,
    cost,
    linkedGatewayTraceId,
    tokenUsage: getSpanTokenUsage({ attributes: span.attributes }),
  };
};

const base64ToHex = (base64: string): string => {
  const binaryString = atob(base64);
  const binaryLen = binaryString.length;
  let hex = '';
  for (let i = 0; i < binaryLen; i++) {
    const charCode = binaryString.charCodeAt(i);
    hex += charCode.toString(16).padStart(2, '0');
  }
  return hex;
};

// mlflow span ids are meant to be interpreted as hex strings
export const decodeSpanId = (spanId: string | null | undefined, isV3Span: boolean): string => {
  if (!spanId) {
    return '';
  }

  // v3 span ids are base64 encoded
  // only attempt decoding if the id length is less than 16 chars
  if (isV3Span && spanId.length < 16) {
    try {
      return base64ToHex(spanId);
    } catch (e) {
      // if base64 decoding fails, just return the original spanId
      return spanId;
    }
  }

  // old V2 span ids (pre-March 2025) are in hex with a 0x prefix
  if (spanId.startsWith('0x')) {
    return spanId.slice(2);
  }

  // new V2 span ids have the prefix stripped
  return spanId;
};

export function isV3ModelTraceInfo(
  info:
    | Pick<ModelTraceInfoV3, 'trace_location' | 'trace_id'>
    | ModelTrace['info']
    | { trace?: { trace_info?: ModelTrace['info'] } }
    | undefined,
): info is ModelTraceInfoV3 {
  if (!info) {
    return false;
  }

  return 'trace_location' in info;
}

export function isV4ModelTraceSpan(span: ModelTraceSpan): span is ModelTraceSpanV4 {
  return 'start_time_unix_nano' in span && Array.isArray(span.attributes);
}

export function isV3ModelTraceSpan(span: ModelTraceSpan): span is ModelTraceSpanV3 {
  return 'start_time_unix_nano' in span;
}

export function isV2ModelTraceSpan(span: ModelTraceSpan): span is ModelTraceSpanV2 {
  return 'parent_id' in span;
}

export function getModelTraceSpanId(span: ModelTraceSpan): string {
  return isV3ModelTraceSpan(span) || isV4ModelTraceSpan(span)
    ? decodeSpanId(span.span_id ?? '', true)
    : decodeSpanId(span.context?.span_id ?? '', false);
}

export function getModelTraceSpanParentId(span: ModelTraceSpan): string {
  return isV3ModelTraceSpan(span) || isV4ModelTraceSpan(span)
    ? decodeSpanId(span.parent_span_id ?? '', true)
    : decodeSpanId(span.parent_id ?? '', false);
}

export function getModelTraceSpanStartTime(span: ModelTraceSpan): number {
  return isV3ModelTraceSpan(span) || isV4ModelTraceSpan(span)
    ? Number(span.start_time_unix_nano)
    : Number(span.start_time);
}

export function getModelTraceSpanEndTime(span: ModelTraceSpan): number {
  return isV3ModelTraceSpan(span) || isV4ModelTraceSpan(span) ? Number(span.end_time_unix_nano) : Number(span.end_time);
}

export function getModelTraceId(trace: ModelTrace): string {
  return isV3ModelTraceInfo(trace.info) ? trace.info.trace_id : (trace.info.request_id ?? '');
}

// get the size of the trace in bytes, or null if the size is not available
export function getModelTraceSize(trace: ModelTrace): number | null {
  if (!isV3ModelTraceInfo(trace.info)) {
    return null;
  }

  const size = Number(trace.info?.trace_metadata?.['mlflow.trace.sizeBytes']);
  return !isNil(size) && !isNaN(size) ? size : null;
}

/**
 * Parses a model trace into a single tree starting with the root span.
 * @param trace - The model trace to parse.
 * @returns The tree starting with the root span.
 */
export function parseModelTraceToTree(trace: ModelTrace): ModelTraceSpanNode | null {
  const topLevelSpans = parseModelTraceToTreeWithMultipleRoots(trace);
  if (!topLevelSpans || topLevelSpans.length !== 1) {
    return null;
  }
  return topLevelSpans[0];
}

/**
 * Parses a model trace into a tree of ModelTraceSpanNodes.
 * @param trace - The model trace to parse.
 * @returns The top-level nodes in the trace. This is a single root span when the trace is complete,
 * but can be multiple spans when the trace is in-progress and root span is not yet emitted.
 */
export function parseModelTraceToTreeWithMultipleRoots(trace: ModelTrace): ModelTraceSpanNode[] {
  const traceId = getModelTraceId(trace);
  const rawSpans = trace.trace_data?.spans ?? trace.data.spans;

  // Normalize span attributes to the common format (K/V list to map).
  const spans = rawSpans.map(convertOtelAttributesToMap);
  const spanMap: { [span_id: string]: ModelTraceSpan } = {};
  const relationMap: { [span_id: string]: string[] } = {};

  if (!spans || spans.length === 0) {
    return [];
  }

  spans.forEach((span) => {
    const spanId = getModelTraceSpanId(span);
    spanMap[spanId] = span;
    relationMap[spanId] = [];
  });

  // Populate child relationships only when the parent exists in the partial set
  spans.forEach((span) => {
    const spanId = getModelTraceSpanId(span);
    const parentId = getModelTraceSpanParentId(span);
    relationMap[parentId]?.push(spanId);
  });

  // Compute a global time window for the tree
  const globalStartTime = Math.min(...spans.map((s) => getModelTraceSpanStartTime(s)));
  const globalEndTime = Math.max(...spans.map((s) => getModelTraceSpanEndTime(s)));
  const assessmentMap = getAssessmentMap(trace.info);

  function getSpanNodeFromData(span_id: string): ModelTraceSpanNode {
    const span = spanMap[span_id];
    const children = relationMap[span_id].map(getSpanNodeFromData);

    // not using `isV2Span` here because for legacy reasons,
    // V1 and V2 are rolled into in the same type. "parent_id" is
    // the way we distinguish between the two.
    if (isV3ModelTraceSpan(span) || isV4ModelTraceSpan(span) || 'parent_id' in span) {
      // reusing the same function for v2 and v3 as the changes are small
      return normalizeNewSpanData(span, globalStartTime, globalEndTime, children, assessmentMap, traceId);
    }

    // v1 spans
    const spanType = span.span_type ?? ModelSpanType.UNKNOWN;
    return {
      title: span.name,
      icon: <ModelTraceExplorerIcon type={getIconTypeForSpan(spanType)} />,
      type: spanType as ModelSpanType,
      key: span.context.span_id,
      start: Number(span.start_time) - globalStartTime,
      // default to the end of the root span if the span has no end time.
      // this can happen if an exception was thrown in the span.
      end: Number(span.end_time ?? globalEndTime) - globalStartTime,
      children: children.slice().sort((a, b) => a.start - b.start),
      inputs: span.inputs,
      outputs: span.outputs,
      attributes: span.attributes,
      events: span.events,
      parentId: span.parent_id ?? span.parent_span_id,
      assessments: [],
      traceId,
      tokenUsage: getSpanTokenUsage({ attributes: span.attributes }),
    };
  }

  // While the trace is in-progress, there can be multiple top-level spans.
  const topLevelNodes = new Array<ModelTraceSpanNode>();
  spans.forEach((s) => {
    const parentId = getModelTraceSpanParentId(s);
    if (!parentId || !spanMap[parentId]) {
      topLevelNodes.push(getSpanNodeFromData(getModelTraceSpanId(s)));
    }
  });
  return topLevelNodes;
}

// returns a map of { [span_id: string] : Assessment[] }
export const getAssessmentMap = (traceInfo: ModelTrace['info']): Record<string, Assessment[]> => {
  let assessments: Assessment[] = [];
  if (isV3ModelTraceInfo(traceInfo)) {
    assessments = traceInfo.assessments ?? [];
  } else {
    assessments = getAssessmentsFromTags(traceInfo.tags);
  }

  // hydrate the assessments with the overridden assessment objects
  const assessmentsById = keyBy(assessments, 'assessment_id');
  Object.values(assessmentsById).forEach((assessment) => {
    if (assessment.overrides) {
      const overriddenAssessment = assessmentsById[assessment.overrides];
      assessment.overriddenAssessment = overriddenAssessment;
    }
  });

  assessments = Object.values(assessmentsById);

  // construct the map by reducing over the assessments
  const assessmentMap = assessments.reduce((acc: Record<string, Assessment[]>, assessment: Assessment) => {
    const spanId = assessment.span_id ?? '';
    if (!acc[spanId]) {
      acc[spanId] = [];
    }
    acc[spanId].push(assessment);
    return acc;
  }, {});

  // sort the assessments by last_update_time
  Object.keys(assessmentMap).forEach((spanId) => {
    assessmentMap[spanId].sort(
      (a, b) => new Date(b.last_update_time).getTime() - new Date(a.last_update_time).getTime(),
    );
  });

  return assessmentMap;
};

// parses assessments out from the trace tags
const getAssessmentsFromTags = (tags: ModelTrace['info']['tags']): Assessment[] => {
  if (!tags) {
    return [];
  }

  const tagList = Array.isArray(tags) ? tags : Object.entries(tags).map(([key, value]) => ({ key, value }));

  return tagList.filter(({ key }) => key.startsWith('mlflow.assessment.')).map(({ value }) => JSON.parse(value));
};

// this function attempts to extract the trace version from
// a given source (either request_metadata or tags)
export function findTraceVersionByKey(
  source: { [key: string]: string } | { key: string; value: string }[] | undefined,
): string | undefined {
  if (!source) {
    return undefined;
  }

  if (isArray(source)) {
    return source.find((tag) => tag.key === MLFLOW_TRACE_SCHEMA_VERSION_KEY)?.value;
  }

  return source[MLFLOW_TRACE_SCHEMA_VERSION_KEY];
}

// this function determines whether an object is a ModelTrace by asserting
// that the object has the `data` and `info` fields, and that the
// trace info contains the `mlflow.trace_schema.version` key
export const isModelTrace = (trace: any): trace is ModelTrace => {
  const traceInfo = trace?.info;
  const traceData = trace?.data;
  if (!traceInfo || !traceData || !traceData?.spans) {
    return false;
  }

  // request_metadata is for V2 traces, trace_metadata is for V3 traces
  const metadata = traceInfo?.request_metadata ?? traceInfo?.trace_metadata;
  if (metadata && findTraceVersionByKey(metadata)) {
    return true;
  }

  const tags = traceInfo?.tags;
  if (tags && findTraceVersionByKey(tags)) {
    return true;
  }

  return false;
};

export const createListFromObject = (
  obj: { [key: string]: any } | string[] | string | boolean | number | undefined,
): { key: string; value: string }[] => {
  if (isNil(obj)) {
    return [];
  }

  if (Array.isArray(obj) || isString(obj) || isNumber(obj) || isBoolean(obj)) {
    return [{ key: '', value: JSON.stringify(obj, null, 2) }];
  }

  return Object.entries(obj).map(([key, value]) => {
    return { key, value: JSON.stringify(value, null, 2) };
  });
};

export const getHighlightedSpanComponents = ({
  searchFilter,
  data,
  activeMatchBackgroundColor,
  inactiveMatchBackgroundColor,
  containsActiveMatch,
  activeMatch,
  scrollToActiveMatch,
}: {
  searchFilter: string;
  data: string;
  activeMatchBackgroundColor: string;
  inactiveMatchBackgroundColor: string;
  containsActiveMatch: boolean;
  activeMatch: SearchMatch;
  scrollToActiveMatch: (node: HTMLSpanElement) => void;
}): React.ReactNode[] => {
  // splitting by regex retains the matches in the array,
  // which makes it easier to handle stuff like preserving
  // the original case of the match.
  const regex = new RegExp(`(${escapeRegExp(searchFilter.trim())})`, 'gi');
  const parts = data.split(regex);
  const spans: React.ReactNode[] = [];
  let matchIndex = 0;

  for (let i = 0; i < parts.length; i++) {
    if (parts[i].toLowerCase().includes(searchFilter.toLowerCase().trim())) {
      const isActiveMatch = containsActiveMatch && activeMatch.matchIndex === matchIndex;
      const backgroundColor = isActiveMatch ? activeMatchBackgroundColor : inactiveMatchBackgroundColor;
      const span = (
        <span ref={isActiveMatch ? scrollToActiveMatch : null} key={i} css={{ backgroundColor, scrollMarginTop: 50 }}>
          {parts[i]}
        </span>
      );
      matchIndex++;
      spans.push(span);
    } else {
      spans.push(parts[i]);
    }
  }

  return spans;
};

export const isRetrieverDocument = (document: any): document is RetrieverDocument => {
  return has(document, 'page_content');
};

export const isRenderableRetrieverSpan = (span: ModelTraceSpanNode): boolean => {
  return (
    span.type === ModelSpanType.RETRIEVER &&
    Array.isArray(span.outputs) &&
    span.outputs.length > 0 &&
    every(span.outputs, isRetrieverDocument)
  );
};

export const getEventAttributeKey = (name: string, index: number, attribute: string): string => {
  return `${name}-${index}-${attribute}`;
};

export { getSpanExceptionCount, getSpanExceptionEvents };

export const isModelTraceChatToolParamProperty = (obj: any): obj is ModelTraceChatToolParamProperty => {
  if (isNil(obj)) {
    return false;
  }

  // JSON Schema allows `type` to be a single string or an array of strings
  // (e.g. ["number", "null"] for nullable/union types).
  if (!isNil(obj.type) && !isString(obj.type) && !(Array.isArray(obj.type) && obj.type.every(isString))) {
    return false;
  }

  if (!isNil(obj.description) && !isString(obj.description)) {
    return false;
  }

  if (!isNil(obj.enum) && !Array.isArray(obj.enum)) {
    return false;
  }

  return true;
};

export const isModelTraceChatTool = (obj: any): obj is ModelTraceChatTool => {
  if (isNil(obj) || obj.type !== 'function' || !has(obj, 'function.name')) {
    return false;
  }

  // conditional validation for the `parameters` field
  const parameters = obj.function?.parameters;
  if (!isNil(parameters)) {
    // if `required` is present, it must be a list of strings
    const required = parameters.required;
    if (!isNil(required) && (!Array.isArray(required) || !required.every(isString))) {
      return false;
    }

    const properties = parameters.properties;
    if (!isNil(properties) && !Object.values(properties).every(isModelTraceChatToolParamProperty)) {
      return false;
    }
  }

  return true;
};

export {
  isModelTraceChatMessage,
  isModelTraceChatResponse,
  isModelTraceChoices,
  isModelTraceToolCall,
  isRawModelTraceChatMessage,
};

export const isModelTraceChatInput = (obj: any): obj is ModelTraceChatInput => {
  return (
    obj && Array.isArray(obj.messages) && obj.messages.length > 0 && obj.messages.every(isRawModelTraceChatMessage)
  );
};

/**
 * Attempt to normalize a conversation, return null in case the format is unrecognized.
 * Defaults to checking OpenAI format if not provided, as it is a common case.
 *
 * Supported formats:
 *   1. Langchain chat inputs
 *   2. Langchain chat results
 *   3. OpenAI ChatCompletions inputs
 *   4. OpenAI ChatCompletions responses
 *   5. OpenAI Responses inputs
 *   6. OpenAI Responses output
 *   7. LlamaIndex chat responses
 *   8. DSPy chat inputs
 *   8. DSPy chat outputs
 *   9. Gemini inputs
 *  10. Gemini outputs
 *  11. Anthropic inputs
 *  12. Anthropic outputs
 *  13. OpenAI Agent inputs
 *  14. OpenAI Agent outputs
 *  15. Autogen inputs
 *  16. Autogen outputs
 *  17. Bedrock inputs
 *  18. Bedrock outputs
 *  19. Vercel AI inputs
 *  20. Vercel AI outputs
 *  21. PydanticAI inputs
 *  22. PydanticAI outputs
 */
const normalizeOpenAIFormats = (input: any): ModelTraceChatMessage[] | null =>
  normalizeOpenAIChatInput(input) ??
  normalizeOpenAIChatResponse(input) ??
  normalizeOpenAIResponsesOutput(input) ??
  normalizeOpenAIResponsesStreamingOutput(input);

export const normalizeConversation = (input: any, messageFormat?: string): ModelTraceChatMessage[] | null => {
  // wrap in try/catch to avoid crashing the UI. we're doing a lot of type coercion
  // and formatting, and it's possible that we miss some edge cases. in case of an error,
  // simply return null to signify that the input is not a chat input.
  try {
    // if the input is already in the correct format, return it
    if (Array.isArray(input) && input.length > 0 && input.every(isRawModelTraceChatMessage)) {
      return compact(input.map(prettyPrintChatMessage));
    }

    switch (messageFormat) {
      case 'langchain': {
        const langchainMessages =
          normalizeLangchainChatInput(input) ??
          normalizeLangchainChatResult(input) ??
          // LangChain autolog may serialize messages in OpenAI format
          normalizeOpenAIFormats(input) ??
          normalizeOpenAIResponsesInput(input);
        if (langchainMessages) return langchainMessages;
        break;
      }
      case 'llamaindex':
        const llamaIndexMessages = normalizeLlamaIndexChatInput(input) ?? normalizeLlamaIndexChatResponse(input);
        if (llamaIndexMessages) return llamaIndexMessages;
        break;
      case 'openai':
        const openAIMessages = normalizeOpenAIFormats(input) ?? normalizeOpenAIResponsesInput(input);
        if (openAIMessages) return openAIMessages;
        break;
      case 'dspy':
        const dspyMessages = normalizeDspyChatInput(input) ?? normalizeDspyChatOutput(input);
        if (dspyMessages) return dspyMessages;
        break;
      case 'gemini':
        const geminiMessages = normalizeGeminiChatInput(input) ?? normalizeGeminiChatOutput(input);
        if (geminiMessages) return geminiMessages;
        break;
      case 'anthropic':
        const anthropicMessages = normalizeAnthropicChatInput(input) ?? normalizeAnthropicChatOutput(input);
        if (anthropicMessages) return anthropicMessages;
        break;
      case 'mistral':
        const mistralMessages = normalizeMistralChatInput(input) ?? normalizeMistralChatOutput(input);
        if (mistralMessages) return mistralMessages;
        break;
      case 'openai-agent':
        const openAIAgentMessages = normalizeOpenAIAgentInput(input) ?? normalizeOpenAIAgentOutput(input);
        if (openAIAgentMessages) return openAIAgentMessages;
        break;
      case 'autogen':
        const autogenMessages = normalizeAutogenChatInput(input) ?? normalizeAutogenChatOutput(input);
        if (autogenMessages) return autogenMessages;
        break;
      case 'bedrock':
        const bedrockMessages = normalizeBedrockChatInput(input) ?? normalizeBedrockChatOutput(input);
        if (bedrockMessages) return bedrockMessages;
        break;
      case 'vercel_ai':
        const vercelAIMessages = normalizeVercelAIChatInput(input) ?? normalizeVercelAIChatOutput(input);
        if (vercelAIMessages) return vercelAIMessages;
        break;
      case 'pydantic_ai':
        const pydanticAIMessages = normalizePydanticAIChatInput(input) ?? normalizePydanticAIChatOutput(input);
        if (pydanticAIMessages) return pydanticAIMessages;
        break;
      case 'voltagent':
        const voltAgentMessages = normalizeVoltAgentChatInput(input) ?? normalizeVoltAgentChatOutput(input);
        if (voltAgentMessages) return voltAgentMessages;
        break;
      default:
        const chatMessages =
          normalizeOpenAIFormats(input) ?? normalizeLangchainChatInput(input) ?? normalizeLangchainChatResult(input);
        if (chatMessages) return chatMessages;
        const geminiFallbackMessages = normalizeGeminiChatInput(input) ?? normalizeGeminiChatOutput(input);
        if (geminiFallbackMessages) return geminiFallbackMessages;
        break;
    }

    if (Array.isArray(input) && input.length > 0 && input.every(isOtelGenAIChatMessage)) {
      return compact(input.map(normalizeOtelGenAIChatMessage));
    }

    return null;
  } catch (e) {
    return null;
  }
};

export { prettyPrintChatMessage, prettyPrintToolCall };

export const getDefaultActiveTab = (
  selectedNode: ModelTraceSpanNode | undefined,
): 'content' | 'attributes' | 'events' => {
  if (isNil(selectedNode)) {
    return 'content';
  }

  // Auto-navigate to events tab when span has errors
  if (getSpanExceptionCount(selectedNode) > 0) {
    return 'events';
  }

  const hasPayload = (payload: unknown): boolean => {
    if (isNil(payload)) return false;
    if (typeof payload === 'string' || Array.isArray(payload)) return payload.length > 0;
    if (typeof payload === 'object') return Object.keys(payload).length > 0;
    return true;
  };
  const hasContent =
    hasPayload(selectedNode.inputs) ||
    hasPayload(selectedNode.outputs) ||
    (selectedNode.chatMessages?.length ?? 0) > 0 ||
    (selectedNode.chatTools?.length ?? 0) > 0;
  if (hasContent) {
    return 'content';
  }

  return 'attributes';
};

/**
 * Processes entire model trace and converts any attributes that are in OTEL key-value array format
 * to a map format.
 */
export const convertOtelAttributesToMap = (modelTraceSpan: ModelTraceSpan): ModelTraceSpan => {
  // Complex values (dicts/lists, e.g. `mlflow.spanInputs` / `mlflow.spanOutputs`)
  // arrive as nested `kvlist_value` / `array_value` structures, so decode
  // recursively. A scalar-only unwrap leaves them as raw OTEL AnyValue objects,
  // which breaks chat-message normalization (the pretty conversation view).
  const getValue = (value: any) => (isOtelAnyValue(value) ? decodeOtelAnyValue(value) : value);

  const convertAttributes = (attributes: any) => {
    if (!Array.isArray(attributes)) {
      return attributes;
    }
    return attributes.reduce(
      (acc, attr) => {
        if (!attr.key || !attr.value) {
          return acc;
        }
        return { ...acc, [attr.key]: getValue(attr.value) };
      },
      {} as Record<string, any>,
    );
  };

  return {
    ...modelTraceSpan,
    ...(modelTraceSpan.attributes && { attributes: convertAttributes(modelTraceSpan.attributes) }),
    ...(modelTraceSpan.events && {
      events: modelTraceSpan.events?.map((event) => ({
        ...event,
        attributes: convertAttributes(event.attributes),
      })),
    }),
  };
};

export const getTotalTokens = (traceInfo: ModelTraceInfoV3): number | null => {
  const tokenUsage = traceInfo.trace_metadata?.[TOKEN_USAGE_METADATA_KEY];
  if (!tokenUsage) {
    return null;
  }

  try {
    const parsedTokenUsage = JSON.parse(tokenUsage);
    return parsedTokenUsage?.total_tokens ?? null;
  } catch {
    return null;
  }
};

export const getTraceTokenUsage = (
  traceInfo: ModelTraceInfoV3,
): {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
} => parseJSONSafe(traceInfo?.trace_metadata?.[TOKEN_USAGE_METADATA_KEY] ?? '{}');

export const getTraceCost = (
  traceInfo: ModelTraceInfoV3,
): { input_cost?: number; output_cost?: number; total_cost?: number } =>
  parseJSONSafe(traceInfo?.trace_metadata?.[COST_METADATA_KEY] ?? '{}');

export const getRootSpanTimeToFirstTokenMs = (rootNode: ModelTraceSpanNode | null): number | null => {
  const raw = getSpanAttribute(rootNode?.attributes, SPAN_ATTRIBUTE_TIME_TO_FIRST_TOKEN_MS_KEY);
  // Accept only a finite number or a non-empty numeric string. Guarding the types
  // before coercion avoids Number() silently turning malformed values into finite
  // numbers (e.g. '' / '   ' -> 0, true -> 1), which would render a fabricated
  // metric instead of hiding it.
  if (typeof raw === 'number') {
    return Number.isFinite(raw) ? raw : null;
  }
  if (typeof raw === 'string' && raw.trim() !== '') {
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  }
  return null;
};

export const isSessionLevelAssessment = (assessment: Assessment): boolean => {
  return !isEmpty(assessment.metadata?.[ASSESSMENT_SESSION_METADATA_KEY]);
};

/**
 * Filters the provided assessments to only include those that are at the trace level
 * (i.e., not associated with a specific session) or are IssueReferenceAssessment types.
 */
export const getTraceLevelAssessments = (assessments?: Assessment[]): Assessment[] =>
  assessments?.filter((assessment) => !isSessionLevelAssessment(assessment) || 'issue' in assessment) ?? [];

export const isValidException = (
  event: ModelTraceEvent,
): event is ModelTraceEvent & {
  attributes: {
    'exception.type': string;
    'exception.message': string;
    'exception.stacktrace'?: string;
    'exception.escaped'?: boolean;
  };
} => {
  const stackTraceExists = event?.attributes && 'exception.stacktrace' in event.attributes;
  const stackTraceIsString = typeof event.attributes?.['exception.stacktrace'] === 'string';

  return Boolean(
    event?.attributes &&
    typeof event.attributes['exception.type'] === 'string' &&
    typeof event.attributes['exception.message'] === 'string' &&
    (stackTraceExists ? stackTraceIsString : true),
  );
};

const CHUNK_RELEVANCE_ASSESSMENT_NAMES = ['chunk_relevance', 'retrieval_relevance'];

export const isChunkRelevanceAssessment = (assessment: Assessment): boolean =>
  CHUNK_RELEVANCE_ASSESSMENT_NAMES.includes(assessment.assessment_name);

export const getAssessmentDocumentIndex = (assessment: Assessment): number | undefined => {
  const spanOutputKey = assessment.metadata?.[MLFLOW_SPAN_OUTPUT_KEY] ?? assessment.metadata?.[CHUNK_INDEX_KEY];
  if (isNil(spanOutputKey)) return undefined;
  const index = Number(spanOutputKey);
  return Number.isNaN(index) ? undefined : index;
};

export const buildDocumentRelevanceAssessmentMap = (assessments: Assessment[]): Map<number, Assessment> => {
  const map = new Map<number, Assessment>();
  for (const assessment of assessments) {
    if (!isChunkRelevanceAssessment(assessment)) continue;
    const index = getAssessmentDocumentIndex(assessment);
    if (index !== undefined) map.set(index, assessment);
  }
  return map;
};
