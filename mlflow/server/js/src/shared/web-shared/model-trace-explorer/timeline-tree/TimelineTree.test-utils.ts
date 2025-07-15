import type { SpanFilterState } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';

export const TEST_SPAN_FILTER_STATE: SpanFilterState = {
  showParents: true,
  showExceptions: true,
  spanTypeDisplayState: {
    [ModelSpanType.CHAIN]: true,
    [ModelSpanType.LLM]: true,
    [ModelSpanType.AGENT]: true,
    [ModelSpanType.TOOL]: true,
    [ModelSpanType.FUNCTION]: true,
    [ModelSpanType.CHAT_MODEL]: true,
    [ModelSpanType.RETRIEVER]: true,
    [ModelSpanType.PARSER]: true,
    [ModelSpanType.EMBEDDING]: true,
    [ModelSpanType.RERANKER]: true,
    [ModelSpanType.UNKNOWN]: true,
  },
};
