import { ModelSpanType, type ModelTrace, type ModelTraceSpan } from './ModelTrace.types';
import { ModelTraceExplorer } from './ModelTraceExplorer';

let counter = 0;

const lorem =
  'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?';

const loremParts = lorem
  .split(/[,.]/)
  .map((part) => part.trim())
  .filter(Boolean)
  .reduce((acc, part) => ({ ...acc, [part]: part }), {});

const getCommonSpanParts = () =>
  ({
    span_type: 'TEST',
    status: {
      description: 'OK',
      status_code: 1,
    },
    attributes: ++counter % 2 === 0 ? { function_name: 'predict' } : {},
    events: [] as any[],
    inputs:
      counter % 5 !== 0
        ? {
            firstInput: { query: `this-is-query-${counter}` },
            secondInput: loremParts,
          }
        : {},
    outputs:
      counter % 3 !== 0
        ? {
            firstOutput: { response: `this-is-response-${counter}` },
            secondOutput: { response: `this-is-another-response-${counter}` },
            thirdOutput: loremParts,
          }
        : {},
  } as const);

const mockSpans: ModelTraceSpan[] = [
  {
    context: { span_id: 'document-qa-chain', trace_id: '1' },
    parent_span_id: null,
    ...getCommonSpanParts(),
    name: 'document-qa-chain',
    start_time: 0 * 1e6,
    end_time: 42 * 1e6,
    inputs: ['this is an input', 'formatted as an array'],
    type: ModelSpanType.CHAIN,
  },
  {
    name: '_generate_response',
    context: { span_id: '_generate_response_1', trace_id: '1' },
    parent_span_id: 'document-qa-chain',
    ...getCommonSpanParts(),
    start_time: 0 * 1e6,
    end_time: 11 * 1e6,
    inputs: 'just a text input',
    outputs: 'just a text output',
    type: ModelSpanType.LLM,
  },
  {
    context: { span_id: 'rephrase_chat_to_queue_1', trace_id: '1' },
    parent_span_id: '_generate_response_1',
    ...getCommonSpanParts(),
    name: 'rephrase_chat_to_queue',
    start_time: 0 * 1e6,
    end_time: 3 * 1e6,
    attributes: {
      'this is a super long and even longer attribute name': 'this is a super long attribute value',
      'this is a super long attribute name': 'this is a super long and even longer and longer attribute value',
      this_is_a_super_long_and_even_longer_attribute_name: 'this_is_a_super_long_attribute_value',
      this_is_a_super_long_attribute_name: 'this_is_a_super_long_and_even_longer_and_longer_attribute_value',
    },
    type: ModelSpanType.CHAT_MODEL,
  },
  {
    context: { span_id: 'similarity_search_1', trace_id: '1' },
    parent_span_id: '_generate_response_1',
    ...getCommonSpanParts(),
    name: 'similarity_search',

    start_time: 3 * 1e6,
    end_time: 7 * 1e6,
    type: ModelSpanType.RETRIEVER,
  },
  {
    context: { span_id: '_get_query_messages_1', trace_id: '1' },
    parent_span_id: '_generate_response_1',
    ...getCommonSpanParts(),
    name: '_get_query_messages',
    start_time: 7 * 1e6,
    end_time: 8 * 1e6,
    type: ModelSpanType.FUNCTION,
  },
  {
    context: { span_id: '_get_token_count_1', trace_id: '1' },
    parent_span_id: '_generate_response_1',
    ...getCommonSpanParts(),
    name: '_get_token_count',
    start_time: 8 * 1e6,
    end_time: 9 * 1e6,
    type: ModelSpanType.FUNCTION,
  },
  {
    context: { span_id: 'query_llm_1', trace_id: '1' },
    parent_span_id: '_generate_response_1',
    ...getCommonSpanParts(),
    name: 'query_llm',
    start_time: 9 * 1e6,
    end_time: 11 * 1e6,
    type: ModelSpanType.LLM,
  },

  {
    name: '_generate_response',
    context: { span_id: '_generate_response_3', trace_id: '1' },
    parent_span_id: 'document-qa-chain',
    ...getCommonSpanParts(),

    start_time: 11 * 1e6,
    end_time: 24 * 1e6,
    type: ModelSpanType.FUNCTION,
  },
  {
    context: { span_id: 'rephrase_chat_to_queue_3', trace_id: '1' },
    parent_span_id: '_generate_response_3',
    ...getCommonSpanParts(),
    name: 'rephrase_chat_to_queue',

    start_time: 11 * 1e6,
    end_time: 12 * 1e6,
    type: ModelSpanType.PARSER,
  },
  {
    context: { span_id: 'similarity_search_3', trace_id: '1' },
    parent_span_id: '_generate_response_3',
    ...getCommonSpanParts(),
    name: 'similarity_search',

    start_time: 12 * 1e6,
    end_time: 18 * 1e6,
    type: ModelSpanType.RETRIEVER,
  },
  {
    context: { span_id: '_get_query_messages_3', trace_id: '1' },
    parent_span_id: '_generate_response_3',
    ...getCommonSpanParts(),
    name: '_get_query_xx_messages',

    start_time: 18 * 1e6,
    end_time: 19 * 1e6,
    type: ModelSpanType.FUNCTION,
  },
  {
    context: { span_id: '_get_token_count_3', trace_id: '1' },
    parent_span_id: '_generate_response_3',
    ...getCommonSpanParts(),
    name: '_get_token_count',

    start_time: 19 * 1e6,
    end_time: 20 * 1e6,
    type: ModelSpanType.FUNCTION,
  },
  {
    context: { span_id: 'query_llm_3', trace_id: '1' },
    parent_span_id: '_generate_response_3',
    ...getCommonSpanParts(),
    name: 'query_llm_xx',

    start_time: 20 * 1e6,
    end_time: 24 * 1e6,
    type: ModelSpanType.LLM,
  },

  {
    name: '_generate_response',
    context: { span_id: '_generate_response_5', trace_id: '1' },
    parent_span_id: 'document-qa-chain',
    ...getCommonSpanParts(),

    start_time: 24 * 1e6,
    end_time: 42 * 1e6,
    type: ModelSpanType.CHAIN,
  },
  {
    context: { span_id: 'rephrase_chat_to_queue_5', trace_id: '1' },
    parent_span_id: '_generate_response_5',
    ...getCommonSpanParts(),
    name: 'rephrase_chat_to_queue',

    start_time: 24 * 1e6,
    end_time: 26 * 1e6,
    type: ModelSpanType.CHAIN,
  },
  {
    context: { span_id: 'similarity_search_5', trace_id: '1' },
    parent_span_id: '_generate_response_5',
    ...getCommonSpanParts(),
    name: 'similarity_search',

    start_time: 26 * 1e6,
    end_time: 31 * 1e6,
    type: ModelSpanType.RETRIEVER,
  },
  {
    context: { span_id: '_get_query_messages_5', trace_id: '1' },
    parent_span_id: '_generate_response_5',
    ...getCommonSpanParts(),
    name: '_get_query_xx_messages',

    start_time: 31 * 1e6,
    end_time: 32 * 1e6,
    type: ModelSpanType.CHAIN,
  },
  {
    context: { span_id: '_get_token_count_5', trace_id: '1' },
    parent_span_id: '_generate_response_5',
    ...getCommonSpanParts(),
    name: '_get_token_count',
    start_time: 32 * 1e6,
    end_time: 40 * 1e6,
    type: ModelSpanType.CHAIN,
  },
  {
    context: { span_id: 'query_llm_5', trace_id: '1' },
    parent_span_id: '_generate_response_5',
    ...getCommonSpanParts(),
    name: 'query_llm_xx',
    start_time: 40 * 1e6,
    end_time: 42 * 1e6,
    type: ModelSpanType.LLM,
  },
];

const modelTrace: ModelTrace = {
  data: {
    spans: mockSpans,
  },
  info: {
    request_id: '1',
    experiment_id: '1',
    timestamp_ms: 1e9,
    execution_time_ms: 1e9,
    status: 'OK',
    tags: [],
    attributes: {},
  },
};

export const Simple = () => (
  <div css={{ marginTop: 48 }}>
    <ModelTraceExplorer modelTrace={modelTrace} />
  </div>
);

export const LimitedHeight = () => (
  <div css={{ marginTop: 48, height: 300, overflow: 'hidden' }}>
    {' '}
    <ModelTraceExplorer modelTrace={modelTrace} />
  </div>
);

export const Skeleton = () => <ModelTraceExplorer.Skeleton />;

const storyConfig = {
  title: 'Model trace explorer/Model trace explorer',
  component: null,
  argTypes: {},
  args: {},
};

export default storyConfig;
