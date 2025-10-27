import type {
  Assessment,
  ModelTrace,
  ModelTraceChatMessage,
  ModelTraceChatTool,
  ModelTraceInfo,
  ModelTraceInfoV3,
  ModelTraceSpanNode,
  ModelTraceSpanV2,
  ModelTraceSpanV3,
} from './ModelTrace.types';
import { ModelSpanType } from './ModelTrace.types';

const commonSpanParts: Pick<ModelTraceSpanV2, 'span_type' | 'status' | 'events'> = {
  span_type: 'TEST',
  status: {
    description: 'OK',
    status_code: 1,
  },
  events: [],
};

export const MOCK_RETRIEVER_SPAN: ModelTraceSpanNode = {
  key: 'Retriever span',
  type: ModelSpanType.RETRIEVER,
  start: 231205.888,
  end: 682486.272,
  inputs: 'tell me about python',
  outputs: [
    {
      page_content: 'Content with metadata',
      metadata: {
        chunk_id: '1',
        doc_uri: 'https://example.com',
        source: 'book-doc',
      },
    },
    {
      page_content: 'Content without metadata',
      metadata: {},
    },
  ],
  attributes: {},
  assessments: [],
  traceId: '',
};

export const MOCK_EVENTS_SPAN: ModelTraceSpanV2 = {
  ...commonSpanParts,
  attributes: {
    function_name: 'top-level-attribute',
    'mlflow.spanInputs': JSON.stringify({ query: 'events_span-input' }),
    'mlflow.spanOutputs': JSON.stringify({ response: 'events_span-output' }),
    'mlflow.spanType': JSON.stringify(ModelSpanType.FUNCTION),
  },
  context: { span_id: 'events_span', trace_id: '1' },
  parent_id: null,
  name: 'events_span',
  start_time: 3.1 * 1e6,
  end_time: 8.1 * 1e6,
  events: [
    {
      name: 'event1',
      attributes: {
        'event1-attr1': 'event-level-attribute',
        'event1-attr2': 'event1-attr2-value',
      },
    },
    {
      name: 'event2',
      attributes: {
        'event2-attr1': 'event2-attr1-value',
        'event2-attr2': 'event2-attr2-value',
      },
    },
  ],
};

export const mockSpans: ModelTraceSpanV2[] = [
  {
    ...commonSpanParts,
    attributes: {
      function_name: 'predict',
      'mlflow.spanInputs': JSON.stringify({ query: 'document-qa-chain-input' }),
      'mlflow.spanOutputs': JSON.stringify({ response: 'document-qa-chain-output' }),
      'mlflow.spanType': JSON.stringify(ModelSpanType.CHAIN),
    },
    context: { span_id: 'document-qa-chain', trace_id: '1' },
    parent_id: null,
    name: 'document-qa-chain',
    start_time: 0 * 1e9,
    end_time: 25 * 1e9,
  },
  {
    ...commonSpanParts,
    attributes: {
      function_name: 'predict',
      'mlflow.spanInputs': JSON.stringify({ query: '_generate_response-input' }),
      'mlflow.spanOutputs': JSON.stringify({ response: '_generate_response-output' }),
      'mlflow.spanType': JSON.stringify(ModelSpanType.CHAT_MODEL),
    },
    name: '_generate_response',
    context: { span_id: '_generate_response', trace_id: '1' },
    parent_id: 'document-qa-chain',
    start_time: 3 * 1e9,
    end_time: 8 * 1e9,
  },
  {
    ...commonSpanParts,
    attributes: {
      function_name: 'rephrase',
      'mlflow.spanInputs': JSON.stringify({ query: 'rephrase_chat_to_queue-input' }),
      'mlflow.spanOutputs': JSON.stringify({ response: 'rephrase_chat_to_queue-output' }),
      'mlflow.spanType': JSON.stringify(ModelSpanType.LLM),
    },
    context: { span_id: 'rephrase_chat_to_queue', trace_id: '1' },
    parent_id: '_generate_response',
    name: 'rephrase_chat_to_queue',
    start_time: 8 * 1e9,
    end_time: 8.5 * 1e9,
  },
];

export const MOCK_V3_SPANS: ModelTraceSpanV3[] = [
  {
    // 55b2f04aabe2a246b114ac6950118668 in hex
    trace_id: 'VbLwSqviokaxFKxpUBGGaA==',
    // a96bcf7b57a48b3d in hex
    span_id: 'qWvPe1ekiz0=',
    trace_state: '',
    parent_span_id: '',
    name: 'document-qa-chain',
    start_time_unix_nano: '0',
    end_time_unix_nano: String(2.5 * 1e10),
    attributes: {
      'mlflow.spanType': 'CHAT_MODEL',
      'mlflow.spanInputs': 'document-qa-chain-input',
      'mlflow.traceRequestId': '"tr-edb54b3d53a44732b8c61530d50b065a"',
      'mlflow.spanOutputs': 'document-qa-chain-output',
    },
    status: {
      message: '',
      code: 'STATUS_CODE_OK',
    },
  },
  {
    trace_id: 'VbLwSqviokaxFKxpUBGGaA==',
    // 31323334 in hex
    span_id: 'MTIzNA==',
    trace_state: '',
    parent_span_id: 'qWvPe1ekiz0=',
    name: 'document-qa-chain',
    start_time_unix_nano: '0',
    end_time_unix_nano: String(2.5 * 1e10),
    attributes: {
      'mlflow.spanType': 'CHAIN',
      'mlflow.spanInputs': 'rephrase_chat_to_queue-input',
      'mlflow.traceRequestId': '"tr-edb54b3d53a44732b8c61530d50b065a"',
      'mlflow.spanOutputs': 'rephrase_chat_to_queue-output',
    },
    status: {
      message: '',
      code: 'STATUS_CODE_OK',
    },
  },
  {
    trace_id: 'VbLwSqviokaxFKxpUBGGaA==',
    // 3132333435 in hex
    span_id: 'MTIzNDU=',
    trace_state: '',
    parent_span_id: 'MTIzNA==',
    name: 'rephrase_chat_to_queue',
    start_time_unix_nano: '0',
    end_time_unix_nano: String(2.5 * 1e10),
    attributes: {
      'mlflow.spanType': 'LLM',
      'mlflow.spanInputs': 'rephrase_chat_to_queue-input',
      'mlflow.traceRequestId': '"tr-edb54b3d53a44732b8c61530d50b065a"',
      'mlflow.spanOutputs': 'rephrase_chat_to_queue-output',
    },
    status: {
      message: '',
      code: 'STATUS_CODE_OK',
    },
  },
];

export const MOCK_ASSESSMENT: Assessment = {
  assessment_id: 'a-test-1',
  assessment_name: 'Relevance',
  trace_id: 'tr-test-v3',
  span_id: '',
  source: {
    source_type: 'LLM_JUDGE',
    source_id: '1',
  },
  create_time: '2025-04-19T09:04:07.875Z',
  last_update_time: '2025-04-19T09:04:07.875Z',
  feedback: {
    value: '5',
  },
  rationale: 'The thought process is sound and follows from the request',
};

export const MOCK_EXPECTATION: Assessment = {
  assessment_id: 'a-test-1',
  assessment_name: 'expected_facts',
  trace_id: 'tr-test-v3',
  span_id: '',
  source: {
    source_type: 'LLM_JUDGE',
    source_id: '1',
  },
  create_time: '2025-04-19T09:04:07.875Z',
  last_update_time: '2025-04-19T09:04:07.875Z',
  expectation: {
    serialized_value: {
      value: '["fact 1", "fact 2"]',
      serialization_format: 'json',
    },
  },
  rationale: 'The thought process is sound and follows from the request',
};

export const MOCK_TRACE_INFO_V3: ModelTraceInfoV3 = {
  trace_id: 'tr-test-v3',
  trace_location: {
    type: 'MLFLOW_EXPERIMENT',
    mlflow_experiment: {
      experiment_id: '3363486573189371',
    },
  },
  request_time: '2025-02-19T09:52:23.140Z',
  execution_duration: '32.583s',
  state: 'OK',
  trace_metadata: {
    'mlflow.sourceRun': '3129ab3cda944e88a995098fea73a808',
    'mlflow.traceInputs': '"test inputs"',
    'mlflow.traceOutputs': '"test outputs"',
    'mlflow.trace_schema.version': '3',
  },
  tags: {},
  assessments: [MOCK_ASSESSMENT],
};

export const MOCK_V3_TRACE: ModelTrace = {
  data: {
    spans: MOCK_V3_SPANS,
  },
  info: MOCK_TRACE_INFO_V3,
};

export const MOCK_TRACE: ModelTrace = {
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

export const MOCK_LANGCHAIN_CHAT_INPUT = [
  [
    {
      content: "What's the weather in Singapore and New York?",
      additional_kwargs: {},
      response_metadata: {},
      type: 'human',
      name: null,
      id: null,
      example: false,
    },
    // tool call specified in additional_kwargs
    {
      additional_kwargs: {
        tool_calls: [
          {
            id: '1',
            function: {
              arguments: '{"city": "Singapore"}',
              name: 'get_weather',
            },
            type: 'function',
          },
        ],
        refusal: null,
      },
      type: 'ai',
      name: null,
      id: null,
      example: false,
    },
    // tool call specified in tool_calls
    {
      content: '',
      additional_kwargs: {},
      tool_calls: [
        {
          name: 'get_weather',
          args: {
            city: 'New York',
          },
          id: '2',
          type: 'tool_call',
        },
      ],
      type: 'ai',
      name: null,
      id: null,
      example: false,
    },
    // tool response
    {
      content: "It's hot in Singapore",
      additional_kwargs: {},
      response_metadata: {},
      type: 'tool',
      name: null,
      id: null,
      tool_call_id: '1',
      artifact: null,
      status: 'success',
    },
  ],
];

export const MOCK_LANGCHAIN_CHAT_OUTPUT = {
  generations: [
    [
      {
        text: 'The weather in Singapore is hot, while in New York, it is cold.',
        generation_info: {
          finish_reason: 'stop',
          logprobs: null,
        },
        type: 'ChatGeneration',
        message: {
          content: 'The weather in Singapore is hot, while in New York, it is cold.',
          additional_kwargs: {
            refusal: null,
          },
          response_metadata: {
            token_usage: {
              completion_tokens: 17,
              prompt_tokens: 156,
              total_tokens: 173,
              completion_tokens_details: {
                audio_tokens: null,
                reasoning_tokens: 0,
              },
              prompt_tokens_details: {
                audio_tokens: null,
                cached_tokens: 0,
              },
            },
            model_name: 'gpt-4o-mini-2024-07-18',
            system_fingerprint: 'fp_f59a81427f',
            finish_reason: 'stop',
            logprobs: null,
          },
          type: 'ai',
          name: null,
          id: 'run-2e7d781c-b478-4a70-b8bf-d2c4ee04878e-0',
        },
      },
    ],
  ],
  llm_output: {
    token_usage: {
      completion_tokens: 17,
      prompt_tokens: 156,
      total_tokens: 173,
      completion_tokens_details: {
        audio_tokens: null,
        reasoning_tokens: 0,
      },
      prompt_tokens_details: {
        audio_tokens: null,
        cached_tokens: 0,
      },
    },
    model_name: 'gpt-4o-mini-2024-07-18',
    system_fingerprint: 'fp_f59a81427f',
  },
  run: null,
  type: 'LLMResult',
};

export const MOCK_OPENAI_CHAT_INPUT = {
  model: 'gpt-4o-mini',
  messages: [
    {
      role: 'user',
      content: 'tell me a joke in 50 words',
    },
    {
      role: 'assistant',
      tool_calls: [
        {
          id: '1',
          function: {
            arguments: '{"joke_length": 50}',
            name: 'tell_joke',
          },
        },
      ],
    },
    {
      role: 'tool',
      content: 'Why did the scarecrow win an award? Because he was outstanding in his field!',
      tool_call_id: '1',
    },
  ],
  tools: [
    {
      type: 'function',
      function: {
        name: 'tell_joke',
        description: 'Tells a joke',
        parameters: {
          properties: {
            joke_length: {
              type: 'integer',
              description: 'The length of the joke in words',
            },
          },
          required: ['joke_length'],
        },
      },
    },
  ],
  temperature: 0,
};

export const MOCK_OPENAI_CHAT_OUTPUT = {
  id: 'chatcmpl-A8HdoWt2DsJgtZoxjjAcPdx01jkul',
  choices: [
    {
      finish_reason: 'stop',
      index: 0,
      logprobs: null,
      message: {
        content: 'Why did the scarecrow win an award? Because he was outstanding in his field!',
        refusal: null,
        role: 'assistant',
        function_call: null,
        tool_calls: null,
      },
    },
  ],
  created: 1726537800,
  model: 'gpt-4o-mini-2024-07-18',
  object: 'chat.completion',
  service_tier: null,
  system_fingerprint: 'fp_483d39d857',
  usage: {
    completion_tokens: 68,
    prompt_tokens: 15,
    total_tokens: 83,
    completion_tokens_details: {
      reasoning_tokens: 0,
    },
  },
};

export const MOCK_LLAMA_INDEX_CHAT_OUTPUT = {
  message: {
    role: 'assistant',
    content: 'Test',
    additional_kwargs: {},
  },
  delta: null,
  logprobs: null,
  additional_kwargs: {
    prompt_tokens: 404,
    completion_tokens: 94,
    total_tokens: 498,
  },
};

export const MOCK_CHAT_MESSAGES = MOCK_OPENAI_CHAT_INPUT.messages as ModelTraceChatMessage[];
export const MOCK_CHAT_TOOLS = MOCK_OPENAI_CHAT_INPUT.tools as ModelTraceChatTool[];

export const MOCK_CHAT_SPAN: ModelTraceSpanNode = {
  ...commonSpanParts,
  attributes: {},
  parentId: null,
  key: 'chat_span',
  start: 3.1 * 1e6,
  end: 8.1 * 1e6,
  inputs: MOCK_LANGCHAIN_CHAT_INPUT,
  outputs: MOCK_LANGCHAIN_CHAT_OUTPUT,
  chatMessages: MOCK_CHAT_MESSAGES,
  chatTools: MOCK_CHAT_TOOLS,
  type: ModelSpanType.CHAT_MODEL,
  assessments: [],
  traceId: '',
};

export const MOCK_CHAT_TOOL_CALL_SPAN: ModelTraceSpanV2 = {
  ...commonSpanParts,
  attributes: {
    'mlflow.spanType': 'CHAT_MODEL',
    'mlflow.spanInputs': MOCK_OPENAI_CHAT_INPUT,
    'mlflow.spanOutputs': MOCK_OPENAI_CHAT_OUTPUT,
    'mlflow.chat.messages': MOCK_OPENAI_CHAT_INPUT.messages,
    'mlflow.chat.tools': MOCK_OPENAI_CHAT_INPUT.tools,
  },
  context: { span_id: 'chat_span', trace_id: '1' },
  parent_id: null,
  name: 'chat_span',
  start_time: 3.1 * 1e6,
  end_time: 8.1 * 1e6,
};

export const MOCK_ROOT_ASSESSMENT: Assessment = {
  assessment_id: 'a-test-1',
  assessment_name: 'Thumbs',
  trace_id: 'tr-test-v3',
  source: {
    source_type: 'HUMAN',
    source_id: 'daniel.lok@databricks.com',
  },
  create_time: '2025-04-28T01:35:53.621Z',
  last_update_time: '2025-04-28T06:28:27.686Z',
  feedback: {
    value: 'up',
  },
  rationale: 'good job',
  valid: false,
};

export const MOCK_OVERRIDDING_ASSESSMENT: Assessment = {
  ...MOCK_ROOT_ASSESSMENT,
  assessment_id: 'a-test-2',
  overrides: MOCK_ROOT_ASSESSMENT.assessment_id,
  valid: true,
};

export const MOCK_SPAN_ASSESSMENT: Assessment = {
  assessment_id: 'a-test-3',
  assessment_name: 'Thumbs',
  trace_id: 'tr-test-v3',
  span_id: 'span-test-1',
  source: {
    source_type: 'HUMAN',
    source_id: 'daniel.lok@databricks.com',
  },
  create_time: '2025-04-28T01:35:53.621Z',
  last_update_time: '2025-04-28T06:28:27.686Z',
  feedback: {
    value: 'down',
  },
  rationale: 'bad job',
};

export const MOCK_TRACE_INFO_V2: ModelTraceInfo = {
  request_id: 'tr-5dfafe5bde684ea5809f0f9524540ce4',
  experiment_id: '3363486573189371',
  timestamp_ms: 1740026648570,
  execution_time_ms: 1546,
  status: 'OK',
  request_metadata: [
    {
      key: 'mlflow.trace_schema.version',
      value: '2',
    },
    {
      key: 'mlflow.traceInputs',
      value: '"test inputs"',
    },
    {
      key: 'mlflow.traceOutputs',
      value: '"test outputs"',
    },
  ],
  tags: [],
};

export const MOCK_OTEL_TRACE: ModelTrace = {
  data: {
    spans: [
      {
        trace_id: 'tLdfEcjATdf92uVsKMbM/Q==',
        span_id: 'i8XoH5ibpGc=',
        trace_state: '',
        parent_span_id: '',
        name: 'conversation_turn',
        start_time_unix_nano: '1757912311704325000',
        end_time_unix_nano: '1757912313258527000',
        attributes: {
          'input.value': 'Now multiply that result by 3',
          'output.value': 'Multiplying 25 by 3 gives you 75.',
          'mlflow.traceRequestId': 'tr-b4b75f11c8c04dd7fddae56c28c6ccfd',
        },
        status: {
          code: 'STATUS_CODE_UNSET',
        },
      },
      {
        trace_id: 'tLdfEcjATdf92uVsKMbM/Q==',
        span_id: 'yuhZxKf3p8w=',
        trace_state: '',
        parent_span_id: 'i8XoH5ibpGc=',
        name: 'tool_execution',
        start_time_unix_nano: '1757912312585773000',
        end_time_unix_nano: '1757912312586234000',
        attributes: {
          'tool_call.function.name': 'calculate',
          'output.value': 'Result: 75',
          'mlflow.traceRequestId': 'tr-b4b75f11c8c04dd7fddae56c28c6ccfd',
          'openinference.span.kind': 'TOOL',
          'tool_call.function.arguments': '{"expression": "25 * 3"}',
        },
        status: {
          code: 'STATUS_CODE_UNSET',
        },
      },
    ],
  },
  info: {
    trace_id: 'tr-b4b75f11c8c04dd7fddae56c28c6ccfd',
    trace_location: {
      type: 'MLFLOW_EXPERIMENT',
      mlflow_experiment: {
        experiment_id: '1',
      },
    },
    request_time: '2025-09-15T04:58:31.704Z',
    execution_duration: '1.554s',
    state: 'OK',
    trace_metadata: {
      'mlflow.trace_schema.version': '3',
    },
    tags: {
      'mlflow.artifactLocation': 'mlflow-artifacts:/1/traces/tr-b4b75f11c8c04dd7fddae56c28c6ccfd/artifacts',
      'mlflow.trace.spansLocation': 'tracking_store',
    },
  },
};
