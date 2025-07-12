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
} from 'lodash';

import { ModelSpanType, ModelIconType, MLFLOW_TRACE_SCHEMA_VERSION_KEY } from './ModelTrace.types';
import type {
  SearchMatch,
  ModelTrace,
  ModelTraceSpan,
  ModelTraceSpanNode,
  ModelTraceChatMessage,
  LangchainChatGeneration,
  LangchainBaseMessage,
  ModelTraceChatResponse,
  ModelTraceChatInput,
  LlamaIndexChatResponse,
  LangchainToolCallMessage,
  ModelTraceToolCall,
  ModelTraceChatTool,
  ModelTraceChatToolParamProperty,
  RawModelTraceChatMessage,
  ModelTraceContentType,
  SpanFilterState,
  ModelTraceSpanV3,
  ModelTraceSpanV2,
  ModelTraceInfoV3,
  Assessment,
  RetrieverDocument,
  ModelTraceEvent,
} from './ModelTrace.types';
import { ModelTraceExplorerIcon } from './ModelTraceExplorerIcon';
import type { OpenAIResponsesOutputItem } from './chat-utils/openai.types';

export const getCurrentUser = () => {
  return 'User';
};

export const displayErrorNotification = (errorMessage: string) => {
  // TODO: display error notification in OSS
  return;
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
  const messagesFromInputs = normalizeConversation(inputs) ?? [];
  const messagesFromOutputs = normalizeConversation(outputs) ?? [];

  // when either input or output is not chat messages, we do not set the chat message fiels.
  if (messagesFromInputs.length === 0 || messagesFromOutputs.length === 0) {
    return undefined;
  }

  return messagesFromInputs.concat(messagesFromOutputs);
};

const getChatToolsFromSpan = (toolsAttributeValue: any, inputs: any): ModelTraceChatTool[] | undefined => {
  // if the `mlflow.chat.tools` attribute is provided
  // and in the correct format, return it as-is
  if (Array.isArray(toolsAttributeValue) && toolsAttributeValue.every(isModelTraceChatTool)) {
    return toolsAttributeValue;
  }

  // otherwise, attempt to parse tools from inputs
  // TODO: support langchain format for tool inputs
  if (Array.isArray(inputs?.tools) && inputs?.tools?.every(isModelTraceChatTool)) {
    return inputs.tools;
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
  const spanType = tryDeserializeAttribute(span.attributes?.['mlflow.spanType']);
  const inputs = tryDeserializeAttribute(span.attributes?.['mlflow.spanInputs']);
  const outputs = tryDeserializeAttribute(span.attributes?.['mlflow.spanOutputs']);
  const parentId = getModelTraceSpanParentId(span);
  const spanId = getModelTraceSpanId(span);

  const assessments = assessmentMap[spanId] ?? [];
  if (!parentId) {
    // assessments that are not associated with a specific
    // span should be displayed at the root.
    assessments.push(...(assessmentMap[''] ?? []));
  }

  // data that powers the "chat" tab
  const chatMessages = getChatMessagesFromSpan(
    tryDeserializeAttribute(span.attributes?.['mlflow.chat.messages']),
    inputs,
    outputs,
  );
  const chatTools = getChatToolsFromSpan(tryDeserializeAttribute(span.attributes?.['mlflow.chat.tools']), inputs);

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
    children,
    inputs,
    outputs,
    attributes,
    events,
    chatMessages,
    chatTools,
    parentId,
    assessments,
    traceId,
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
export const decodeSpanId = (spanId: string, isV3Span: boolean): string => {
  if (isV3Span) {
    // v3 span ids are base64 encoded
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

export function isV3ModelTraceInfo(info: ModelTrace['info']): info is ModelTraceInfoV3 {
  return 'trace_metadata' in info;
}

export function isV3ModelTraceSpan(span: ModelTraceSpan): span is ModelTraceSpanV3 {
  return 'start_time_unix_nano' in span;
}

export function isV2ModelTraceSpan(span: ModelTraceSpan): span is ModelTraceSpanV2 {
  return 'parent_id' in span;
}

export function getModelTraceSpanId(span: ModelTraceSpan): string {
  return isV3ModelTraceSpan(span) ? decodeSpanId(span.span_id, true) : decodeSpanId(span.context?.span_id ?? '', false);
}

export function getModelTraceSpanParentId(span: ModelTraceSpan): string {
  return isV3ModelTraceSpan(span) ? decodeSpanId(span.parent_span_id, true) : decodeSpanId(span.parent_id ?? '', false);
}

export function getModelTraceSpanStartTime(span: ModelTraceSpan): number {
  return isV3ModelTraceSpan(span) ? Number(span.start_time_unix_nano) : Number(span.start_time);
}

export function getModelTraceSpanEndTime(span: ModelTraceSpan): number {
  return isV3ModelTraceSpan(span) ? Number(span.end_time_unix_nano) : Number(span.end_time);
}

export function getModelTraceId(trace: ModelTrace): string {
  return isV3ModelTraceInfo(trace.info) ? trace.info.trace_id : trace.info.request_id ?? '';
}

export function parseModelTraceToTree(trace: ModelTrace): ModelTraceSpanNode | null {
  const traceId = getModelTraceId(trace);
  const spans = trace.trace_data?.spans ?? trace.data.spans;
  const spanMap: { [span_id: string]: ModelTraceSpan } = {};
  const relationMap: { [span_id: string]: string[] } = {};

  spans.forEach((span) => {
    const spanId = getModelTraceSpanId(span);
    spanMap[spanId] = span;
    relationMap[spanId] = [];
  });

  spans.forEach((span) => {
    const spanId = getModelTraceSpanId(span);
    const parentId = getModelTraceSpanParentId(span);
    if (parentId) {
      if (!relationMap[parentId]) {
        throw new Error('Tree structure is malformed!');
      }
      relationMap[parentId].push(spanId);
    }
  });

  const rootSpan = spans.find((span) => !getModelTraceSpanParentId(span));
  if (isNil(rootSpan)) {
    return null;
  }

  const rootSpanId = getModelTraceSpanId(rootSpan);
  function getSpanNodeFromData(span_id: string): ModelTraceSpanNode {
    const span = spanMap[span_id];
    // above we return if rootSpan is null, but for some
    // reason typescript thinks it's still nullable here.
    const rootStart = Number(getModelTraceSpanStartTime(rootSpan as ModelTraceSpan));
    const rootEnd = Number(getModelTraceSpanEndTime(rootSpan as ModelTraceSpan));
    const children = relationMap[span_id].map(getSpanNodeFromData);
    const assessmentMap = getAssessmentMap(trace.info);

    // not using `isV2Span` here because for legacy reasons,
    // V1 and V2 are rolled into in the same type. "parent_id" is
    // the way we distinguish between the two.
    if (isV3ModelTraceSpan(span) || 'parent_id' in span) {
      // reusing the same function for v2 and v3 as the changes are small
      return normalizeNewSpanData(span, rootStart, rootEnd, children, assessmentMap, traceId);
    }

    // v1 spans
    const spanType = span.span_type ?? ModelSpanType.UNKNOWN;
    return {
      title: span.name,
      icon: <ModelTraceExplorerIcon type={getIconTypeForSpan(spanType)} />,
      type: spanType as ModelSpanType,
      key: span.context.span_id,
      start: Number(span.start_time) - rootStart,
      // default to the end of the root span if the span has no end time.
      // this can happen if an exception was thrown in the span.
      end: Number(span.end_time ?? rootEnd) - rootStart,
      children: children,
      inputs: span.inputs,
      outputs: span.outputs,
      attributes: span.attributes,
      events: span.events,
      parentId: span.parent_id ?? span.parent_span_id,
      assessments: [],
      traceId,
    };
  }

  return getSpanNodeFromData(rootSpanId);
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
) => {
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
}) => {
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

export const getSpanExceptionEvents = (span: ModelTraceSpanNode | ModelTraceSpan): ModelTraceEvent[] => {
  return (span.events ?? []).filter((event) => event.name === 'exception');
};

export const getSpanExceptionCount = (span: ModelTraceSpanNode | ModelTraceSpan): number => {
  return getSpanExceptionEvents(span).length;
};

export const langchainMessageToModelTraceMessage = (message: LangchainBaseMessage): ModelTraceChatMessage | null => {
  let role: ModelTraceChatMessage['role'];
  switch (message.type) {
    case 'user':
    case 'human':
      role = 'user';
      break;
    case 'assistant':
    case 'ai':
      role = 'assistant';
      break;
    case 'system':
      role = 'system';
      break;
    case 'tool':
      role = 'tool';
      break;
    case 'function':
      role = 'function';
      break;
    default:
      return null;
  }

  const normalizedMessage: ModelTraceChatMessage = {
    content: message.content,
    role,
  };

  const toolCalls = message.tool_calls;
  const toolCallsFromKwargs = message.additional_kwargs?.tool_calls;

  // attempt to parse tool calls from the top-level field,
  // otherwise fall back to the additional_kwargs field if it exists
  if (
    !isNil(toolCalls) &&
    Array.isArray(toolCalls) &&
    toolCalls.length > 0 &&
    toolCalls.every(isLangchainToolCallMessage)
  ) {
    // compact for typing. the coercion should not fail since we
    // check that the type is correct in the if condition above
    normalizedMessage.tool_calls = compact(toolCalls.map(normalizeLangchainToolCall));
  } else if (
    !isNil(toolCallsFromKwargs) &&
    Array.isArray(toolCallsFromKwargs) &&
    toolCallsFromKwargs.length > 0 &&
    toolCallsFromKwargs.every(isModelTraceToolCall)
  ) {
    normalizedMessage.tool_calls = toolCallsFromKwargs.map(prettyPrintToolCall);
  }

  if (!isNil(message.tool_call_id)) {
    normalizedMessage.tool_call_id = message.tool_call_id;
  }

  return normalizedMessage;
};

export const normalizeLangchainToolCall = (toolCall: LangchainToolCallMessage): ModelTraceToolCall | null => {
  return {
    id: toolCall.id,
    function: {
      arguments: JSON.stringify(toolCall.args, null, 2),
      name: toolCall.name,
    },
  };
};

export const isModelTraceChatToolParamProperty = (obj: any): obj is ModelTraceChatToolParamProperty => {
  if (isNil(obj)) {
    return false;
  }

  if (!isNil(obj.type) && !isString(obj.type)) {
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

export const isModelTraceToolCall = (obj: any): obj is ModelTraceToolCall => {
  return obj && isString(obj.id) && isString(obj.function?.arguments) && isString(obj.function?.name);
};

const isContentPart = (part: any) => {
  switch (part.type) {
    case 'text':
    case 'input_text':
    case 'output_text':
      return isString(part.text);
    case 'image_url':
      const { image_url } = part;
      if (isNil(image_url)) {
        return false;
      }
      return isString(image_url.url) && (isNil(image_url.detail) || ['auto', 'low', 'high'].includes(image_url.detail));
    case 'input_audio':
      const { input_audio } = part;
      if (isNil(input_audio)) {
        return false;
      }
      return isString(input_audio.data) && (isNil(input_audio.format) || ['wav', 'mp3'].includes(input_audio.format));
    default:
      return false;
  }
};

const isContentType = (content: any) => {
  if (isNil(content) || isString(content)) {
    return true;
  }

  if (isArray(content)) {
    return content.every((part) => isContentPart(part));
  }

  return false;
};

export const isModelTraceChatMessage = (message: any): message is ModelTraceChatMessage => {
  if (!isRawModelTraceChatMessage(message)) {
    return false;
  }

  return isNil(message.content) || isString(message.content);
};

export const isRawModelTraceChatMessage = (message: any): message is RawModelTraceChatMessage => {
  if (!message) {
    return false;
  }

  if (message.tool_calls) {
    if (!Array.isArray(message.tool_calls)) {
      return false;
    }

    if (!message.tool_calls.every(isModelTraceToolCall)) {
      return false;
    }
  }

  if (message.type === 'reasoning') {
    return true;
  }

  // verify if the message content is a valid content type or not
  if (!isContentType(message.content)) {
    return false;
  }

  return (
    message.role === 'user' || message.role === 'assistant' || message.role === 'system' || message.role === 'tool'
  );
};

export const isModelTraceChatInput = (obj: any): obj is ModelTraceChatInput => {
  return (
    obj && Array.isArray(obj.messages) && obj.messages.length > 0 && obj.messages.every(isRawModelTraceChatMessage)
  );
};

export const isModelTraceChoices = (obj: any): obj is ModelTraceChatResponse['choices'] => {
  return (
    Array.isArray(obj) &&
    obj.length > 0 &&
    obj.every((choice: any) => has(choice, 'message') && isModelTraceChatMessage(choice.message))
  );
};

export const isModelTraceChatResponse = (obj: any): obj is ModelTraceChatResponse => {
  return obj && isModelTraceChoices(obj.choices);
};

export const isLangchainBaseMessage = (obj: any): obj is LangchainBaseMessage => {
  if (!obj) {
    return false;
  }

  // it's okay if it's undefined / null, but if present it must be a string
  if (!isNil(obj.content) && !isString(obj.content)) {
    return false;
  }

  // tool call validation is handled by the normalization function
  return ['human', 'user', 'assistant', 'ai', 'system', 'tool', 'function'].includes(obj.type);
};

export const isLangchainToolCallMessage = (obj: any): obj is LangchainToolCallMessage => {
  return obj && isString(obj.name) && has(obj, 'args') && isString(obj.id);
};

export const isLangchainChatGeneration = (obj: any): obj is LangchainChatGeneration => {
  return obj && isLangchainBaseMessage(obj.message);
};

export const isLlamaIndexChatResponse = (obj: any): obj is LlamaIndexChatResponse => {
  return obj && isModelTraceChatMessage(obj.message);
};

/**
 * Attempt to normalize a conversation, return null in case the format is unrecognized
 * TODO: move all chat parsing logic to the chat-utils folder to avoid cluttering this
 * utils file.
 *
 * Supported formats:
 *   1. Langchain chat inputs
 *   2. Langchain chat results
 *   3. OpenAI ChatCompletions inputs
 *   4. OpenAI ChatCompletions responses
 *   5. OpenAI Responses inputs
 *   6. OpenAI Responses output
 *   7. LlamaIndex chat responses
 */
export const normalizeConversation = (input: any): ModelTraceChatMessage[] | null => {
  // wrap in try/catch to avoid crashing the UI. we're doing a lot of type coercion
  // and formatting, and it's possible that we miss some edge cases. in case of an error,
  // simply return null to signify that the input is not a chat input.
  try {
    // if the input is already in the correct format, return it
    if (Array.isArray(input) && input.length > 0 && input.every(isRawModelTraceChatMessage)) {
      return compact(input.map(prettyPrintChatMessage));
    }

    const langchainChatInput = normalizeLangchainChatInput(input);
    if (langchainChatInput) {
      return langchainChatInput;
    }

    const openAIChatInput = normalizeOpenAIChatInput(input);
    if (openAIChatInput) {
      return openAIChatInput;
    }

    const langchainChatResult = normalizeLangchainChatResult(input);
    if (langchainChatResult) {
      return langchainChatResult;
    }

    const openAIChatResponse = normalizeOpenAIChatResponse(input);
    if (openAIChatResponse) {
      return openAIChatResponse;
    }

    const openAIResponsesOutput = normalizeOpenAIResponsesOutput(input);
    if (openAIResponsesOutput) {
      return openAIResponsesOutput;
    }

    const llamaIndexChatResponse = normalizeLlamaIndexChatResponse(input);
    if (llamaIndexChatResponse) {
      return llamaIndexChatResponse;
    }

    return null;
  } catch (e) {
    return null;
  }
};

// normalize langchain chat input format
export const normalizeLangchainChatInput = (obj: any): ModelTraceChatMessage[] | null => {
  // it could be a list of list of messages
  if (
    Array.isArray(obj) &&
    obj.length === 1 &&
    Array.isArray(obj[0]) &&
    obj[0].length > 0 &&
    obj[0].every(isLangchainBaseMessage)
  ) {
    const messages = obj[0].map(langchainMessageToModelTraceMessage);
    // if we couldn't convert all the messages, then consider the input invalid
    if (messages.some((message) => message === null)) {
      return null;
    }

    return messages as ModelTraceChatMessage[];
  }

  // it could also be an object with the `messages` key
  if (Array.isArray(obj?.messages) && obj.messages.length > 0 && obj.messages.every(isLangchainBaseMessage)) {
    const messages = obj.messages.map(langchainMessageToModelTraceMessage);

    if (messages.some((message: ModelTraceChatMessage[] | null) => message === null)) {
      return null;
    }

    return messages as ModelTraceChatMessage[];
  }

  return null;
};

const isLangchainChatGenerations = (obj: any): obj is LangchainChatGeneration[][] => {
  if (!Array.isArray(obj) || obj.length < 1) {
    return false;
  }

  if (!Array.isArray(obj[0]) || obj[0].length < 1) {
    return false;
  }

  // langchain chat generations are a list of lists of messages
  return obj[0].every(isLangchainChatGeneration);
};

const getMessagesFromLangchainChatGenerations = (
  generations: LangchainChatGeneration[],
): ModelTraceChatMessage[] | null => {
  const messages = generations.map((generation: LangchainChatGeneration) =>
    langchainMessageToModelTraceMessage(generation.message),
  );

  if (messages.some((message) => message === null)) {
    return null;
  }

  return messages as ModelTraceChatMessage[];
};

// detect if an object is a langchain ChatResult, and normalize it to a list of messages
export const normalizeLangchainChatResult = (obj: any): ModelTraceChatMessage[] | null => {
  if (isLangchainChatGenerations(obj)) {
    return getMessagesFromLangchainChatGenerations(obj[0]);
  }

  if (
    !Array.isArray(obj?.generations) ||
    !(obj.generations.length > 0) ||
    !obj.generations[0].every(isLangchainChatGeneration)
  ) {
    return null;
  }

  return getMessagesFromLangchainChatGenerations(obj.generations[0]);
};

export const prettyPrintToolCall = (toolCall: ModelTraceToolCall): ModelTraceToolCall => {
  // add some spacing to the arguments for better readability
  let args = toolCall.function?.arguments;
  try {
    args = JSON.stringify(JSON.parse(args), null, 2);
  } catch (e) {
    // use original args
  }
  return {
    id: toolCall.id,
    function: {
      arguments: args,
      name: toolCall.function.name,
    },
  };
};

const formatChatContent = (content?: ModelTraceContentType | null): string | undefined | null => {
  if (isNil(content) || isString(content)) {
    return content;
  }

  return (
    content
      // eslint-disable-next-line array-callback-return
      .map((part) => {
        switch (part.type) {
          case 'text':
          case 'input_text':
          case 'output_text':
            return part.text;
          case 'image_url':
            // raw encoded image content is not displayed in the UI
            return '[image]';
          case 'input_audio':
            // raw encoded audio content is not displayed in the UI
            return '[audio]';
        }
      })
      .join('\n')
  );
};

export const prettyPrintChatMessage = (message: RawModelTraceChatMessage): ModelTraceChatMessage | null => {
  // TODO: support rich rendering of reasoning messages
  // for now, just return null and compact it away in the caller
  // this is because we want to still render the rest of the message
  // outputs.
  if (message.type === 'reasoning') {
    return null;
  }

  return {
    ...message,
    content: formatChatContent(message.content),
    tool_calls: message.tool_calls?.map(prettyPrintToolCall),
  };
};

// normalize the OpenAI chat input format (object with 'messages' or 'input' key)
export const normalizeOpenAIChatInput = (obj: any): ModelTraceChatMessage[] | null => {
  if (!obj) {
    return null;
  }

  const messages = obj.messages ?? obj.input;
  if (!Array.isArray(messages) || messages.length === 0 || !messages.every(isRawModelTraceChatMessage)) {
    return null;
  }

  return compact(messages.map(prettyPrintChatMessage));
};

// normalize the OpenAI chat response format (object with 'choices' key)
export const normalizeOpenAIChatResponse = (obj: any): ModelTraceChatMessage[] | null => {
  if (isModelTraceChoices(obj)) {
    return obj.map((choice) => ({
      ...choice.message,
      tool_calls: choice.message.tool_calls?.map(prettyPrintToolCall),
    }));
  }

  if (!isModelTraceChatResponse(obj)) {
    return null;
  }

  return obj.choices.map((choice) => ({
    ...choice.message,
    tool_calls: choice.message.tool_calls?.map(prettyPrintToolCall),
  }));
};

export const isOpenAIResponsesOutputItem = (obj: any): obj is OpenAIResponsesOutputItem => {
  if (!obj) {
    return false;
  }

  if (obj.type === 'message') {
    return isRawModelTraceChatMessage(obj);
  }

  if (obj.type === 'function_call') {
    return isString(obj.call_id) && isString(obj.name) && isString(obj.arguments);
  }

  if (obj.type === 'function_call_output') {
    return isString(obj.call_id) && isString(obj.output);
  }

  return false;
};

export const normalizeOpenAIResponsesOutputItem = (obj: OpenAIResponsesOutputItem): ModelTraceChatMessage | null => {
  if (obj.type === 'message') {
    return prettyPrintChatMessage(obj);
  }

  if (obj.type === 'function_call') {
    return {
      role: 'assistant',
      tool_calls: [
        prettyPrintToolCall({
          id: obj.call_id,
          function: {
            arguments: obj.arguments,
            name: obj.name,
          },
        }),
      ],
    };
  }

  if (obj.type === 'function_call_output') {
    return {
      role: 'tool',
      tool_call_id: obj.call_id,
      content: obj.output,
    };
  }

  return null;
};

export const normalizeOpenAIResponsesOutput = (obj: any): ModelTraceChatMessage[] | null => {
  if (!obj) {
    return null;
  }

  // list of output items
  if (Array.isArray(obj) && obj.length > 0 && obj.every(isOpenAIResponsesOutputItem)) {
    return compact(obj.map(normalizeOpenAIResponsesOutputItem));
  }

  // list of output chunks
  if (
    Array.isArray(obj) &&
    obj.length > 0 &&
    obj.every((chunk) => chunk.type === 'response.output_item.done' && isOpenAIResponsesOutputItem(chunk.item))
  ) {
    return compact(obj.map((chunk) => normalizeOpenAIResponsesOutputItem(chunk.item)));
  }

  return null;
};

export const normalizeLlamaIndexChatResponse = (obj: any): ModelTraceChatMessage[] | null => {
  if (!isLlamaIndexChatResponse(obj)) {
    return null;
  }

  return [obj.message];
};
