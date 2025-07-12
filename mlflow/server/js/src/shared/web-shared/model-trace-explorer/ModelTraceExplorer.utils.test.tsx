import { describe, expect, it } from '@jest/globals';

import { ModelSpanType } from './ModelTrace.types';
import type { ModelTraceChatMessage, ModelTraceSpanNode } from './ModelTrace.types';
import {
  MOCK_CHAT_TOOL_CALL_SPAN,
  MOCK_LANGCHAIN_CHAT_INPUT,
  MOCK_LANGCHAIN_CHAT_OUTPUT,
  MOCK_LLAMA_INDEX_CHAT_OUTPUT,
  MOCK_OPENAI_CHAT_INPUT,
  MOCK_OPENAI_CHAT_OUTPUT,
  MOCK_OVERRIDDING_ASSESSMENT,
  MOCK_ROOT_ASSESSMENT,
  MOCK_SPAN_ASSESSMENT,
  MOCK_TRACE,
  MOCK_TRACE_INFO_V2,
  MOCK_TRACE_INFO_V3,
  MOCK_V3_SPANS,
} from './ModelTraceExplorer.test-utils';
import {
  parseModelTraceToTree,
  searchTree,
  searchTreeBySpanId,
  getMatchesFromSpan,
  normalizeConversation,
  isModelTraceChatTool,
  normalizeNewSpanData,
  prettyPrintChatMessage,
  isRawModelTraceChatMessage,
  isModelTrace,
  isModelTraceChatMessage,
  getAssessmentMap,
  decodeSpanId,
  prettyPrintToolCall,
} from './ModelTraceExplorer.utils';
import {
  MOCK_OPENAI_RESPONSES_INPUT,
  MOCK_OPENAI_RESPONSES_OUTPUT,
  MOCK_OPENAI_RESPONSES_STREAMING_OUTPUT,
} from './chat-utils/openai.test-utils';
import { TEST_SPAN_FILTER_STATE } from './timeline-tree/TimelineTree.test-utils';

describe('parseTraceToTree', () => {
  it('should parse a trace into an MLflowSpanNode', () => {
    const rootNode = parseModelTraceToTree(MOCK_TRACE);

    expect(rootNode).toBeDefined();
    expect(rootNode).toEqual(
      expect.objectContaining({
        key: 'document-qa-chain',
        title: 'document-qa-chain',
        children: [
          expect.objectContaining({
            key: '_generate_response',
            title: '_generate_response',
            children: [
              expect.objectContaining({
                key: 'rephrase_chat_to_queue',
                title: 'rephrase_chat_to_queue',
              }),
            ],
          }),
        ],
      }),
    );
  });

  it('should return null if the trace has no spans', () => {
    const rootNode = parseModelTraceToTree({
      ...MOCK_TRACE,
      trace_data: {
        spans: [],
      },
    });

    expect(rootNode).toBeNull();
  });
});

describe('searchTree', () => {
  it('should filter a tree based on a search string', () => {
    const rootNode = parseModelTraceToTree(MOCK_TRACE) as ModelTraceSpanNode;

    const filterState = {
      ...TEST_SPAN_FILTER_STATE,
      showParents: false,
    };

    const { filteredTreeNodes: rephraseNodes } = searchTree(rootNode, 'rephrase', filterState);

    expect(rephraseNodes).toEqual([
      expect.objectContaining({
        key: 'rephrase_chat_to_queue',
        title: 'rephrase_chat_to_queue',
      }),
    ]);

    const { filteredTreeNodes: predictNodes } = searchTree(rootNode, 'predict', filterState);
    expect(predictNodes).toEqual([
      expect.objectContaining({
        key: 'document-qa-chain',
        title: 'document-qa-chain',
        children: [
          expect.objectContaining({
            key: '_generate_response',
            title: '_generate_response',
            // the child of `_generate_response` should be
            // cut out as it does not contain `predict`.
            children: [],
          }),
        ],
      }),
    ]);
  });

  it('should return a list of matches from each node', () => {
    const rootNode = parseModelTraceToTree(MOCK_TRACE) as ModelTraceSpanNode;

    // should match the "response" key in the output of all spans
    // and also the "_generate_response-*" values in the second span
    const { matches: resMatches } = searchTree(rootNode, 'res', TEST_SPAN_FILTER_STATE);

    expect(resMatches).toEqual([
      // first span
      expect.objectContaining({
        section: 'outputs',
        key: 'response',
        isKeyMatch: true,
        matchIndex: 0,
      }),
      // second span
      expect.objectContaining({
        section: 'inputs',
        key: 'query',
        isKeyMatch: false,
        matchIndex: 0,
      }),
      expect.objectContaining({
        section: 'outputs',
        key: 'response',
        isKeyMatch: true,
        matchIndex: 0,
      }),
      expect.objectContaining({
        section: 'outputs',
        key: 'response',
        isKeyMatch: false,
        matchIndex: 0,
      }),
      // last span
      expect.objectContaining({
        section: 'outputs',
        key: 'response',
        isKeyMatch: true,
        matchIndex: 0,
      }),
    ]);

    // should work on attributes as well
    const { matches: predictMatches } = searchTree(rootNode, 'predict', TEST_SPAN_FILTER_STATE);
    expect(predictMatches).toEqual([
      // first span
      expect.objectContaining({
        section: 'attributes',
        key: 'function_name',
        isKeyMatch: false,
        matchIndex: 0,
      }),
      // second span
      expect.objectContaining({
        section: 'attributes',
        key: 'function_name',
        isKeyMatch: false,
        matchIndex: 0,
      }),
    ]);
  });
});

describe('searchTreeBySpanId', () => {
  it('should return a matched node if exists', () => {
    const rootNode = parseModelTraceToTree(MOCK_TRACE) as ModelTraceSpanNode;
    const node = searchTreeBySpanId(rootNode, 'rephrase_chat_to_queue');

    expect(node).toEqual(
      expect.objectContaining({
        key: 'rephrase_chat_to_queue',
      }),
    );
  });

  it('should return undefined when selectedSpanId is undefined', () => {
    const rootNode = parseModelTraceToTree(MOCK_TRACE) as ModelTraceSpanNode;
    const node = searchTreeBySpanId(rootNode, undefined);

    expect(node).toBeUndefined();
  });

  it('should return undefined when no node matches selectedSpanId', () => {
    const rootNode = parseModelTraceToTree(MOCK_TRACE) as ModelTraceSpanNode;
    const node = searchTreeBySpanId(rootNode, 'unknown');

    expect(node).toBeUndefined();
  });
});

describe('getMatchesFromSpan', () => {
  it('should not crash if a span has any undefined fields', () => {
    const spanNode: ModelTraceSpanNode = {
      key: 'test',
      title: 'test',
      children: [],
      inputs: undefined,
      outputs: undefined,
      attributes: undefined,
      start: 0,
      end: 1,
      type: ModelSpanType.UNKNOWN,
      assessments: [],
      traceId: 'test',
    };

    expect(getMatchesFromSpan(spanNode, 'no-match')).toHaveLength(0);
  });
});

describe('normalizeConversation', () => {
  it('handles a langchain chat input', () => {
    expect(normalizeConversation(MOCK_LANGCHAIN_CHAT_INPUT)).toEqual([
      expect.objectContaining({
        role: 'user',
        content: "What's the weather in Singapore and New York?",
      }),
      expect.objectContaining({
        role: 'assistant',
        content: undefined,
        tool_calls: [
          {
            id: '1',
            function: {
              // assert that it pretty prints
              arguments: '{\n  "city": "Singapore"\n}',
              name: 'get_weather',
            },
          },
        ],
      }),
      expect.objectContaining({
        role: 'assistant',
        tool_calls: [
          {
            id: '2',
            function: {
              arguments: '{\n  "city": "New York"\n}',
              name: 'get_weather',
            },
          },
        ],
      }),
      expect.objectContaining({
        role: 'tool',
        content: "It's hot in Singapore",
        tool_call_id: '1',
      }),
    ]);
  });

  it('handles a langchain chat output', () => {
    expect(normalizeConversation(MOCK_LANGCHAIN_CHAT_OUTPUT)).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: 'The weather in Singapore is hot, while in New York, it is cold.',
      }),
    ]);
  });

  it('handles an OpenAI chat input', () => {
    expect(normalizeConversation(MOCK_OPENAI_CHAT_INPUT)).toEqual([
      expect.objectContaining({
        role: 'user',
        content: 'tell me a joke in 50 words',
      }),
      expect.objectContaining({
        role: 'assistant',
        tool_calls: [
          {
            id: '1',
            function: {
              arguments: '{\n  "joke_length": 50\n}',
              name: 'tell_joke',
            },
          },
        ],
      }),
      expect.objectContaining({
        role: 'tool',
        content: 'Why did the scarecrow win an award? Because he was outstanding in his field!',
      }),
    ]);
  });

  it('handles an OpenAI chat output', () => {
    expect(normalizeConversation(MOCK_OPENAI_CHAT_OUTPUT)).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: 'Why did the scarecrow win an award? Because he was outstanding in his field!',
      }),
    ]);
  });

  it('handles a LlamaIndex chat output', () => {
    expect(normalizeConversation(MOCK_LLAMA_INDEX_CHAT_OUTPUT)).toEqual([
      expect.objectContaining({
        role: 'assistant',
        content: 'Test',
      }),
    ]);
  });

  it('handles a properly formatted input', () => {
    const input = [{ role: 'user', content: 'Hello' }];
    // should be unchanged
    expect(normalizeConversation(input)).toEqual(input);
  });

  it('handles an empty input', () => {
    expect(normalizeConversation(undefined)).toEqual(null);
  });

  it('returns null on unknown roles', () => {
    // openai format
    expect(normalizeConversation({ messages: [{ role: 'unknown', content: 'Hello' }] })).toBeNull();

    // langchain format
    expect(normalizeConversation({ messages: [{ type: 'unknown', content: 'Hello' }] })).toBeNull();
  });

  it('returns null if tools have no function name', () => {
    expect(
      normalizeConversation({ messages: [{ role: 'assistant', tool_calls: [{ id: 'hello', type: 'yay' }] }] }),
    ).toBeNull();
  });

  it('handles an OpenAI responses formats', () => {
    expect(normalizeConversation(MOCK_OPENAI_RESPONSES_INPUT)).toEqual([
      expect.objectContaining({
        role: 'user',
        content: 'Test content',
      }),
    ]);

    expect(normalizeConversation(MOCK_OPENAI_RESPONSES_OUTPUT)).toEqual([
      expect.objectContaining({
        role: 'assistant',
        tool_calls: [
          prettyPrintToolCall({
            id: MOCK_OPENAI_RESPONSES_OUTPUT[0].call_id ?? '',
            function: {
              arguments: MOCK_OPENAI_RESPONSES_OUTPUT[0].arguments ?? '',
              name: MOCK_OPENAI_RESPONSES_OUTPUT[0].name ?? '',
            },
          }),
        ],
      }),
      expect.objectContaining({
        role: 'tool',
        tool_call_id: MOCK_OPENAI_RESPONSES_OUTPUT[1].call_id,
        content: MOCK_OPENAI_RESPONSES_OUTPUT[1].output,
      }),
      expect.objectContaining({
        role: 'assistant',
        content: MOCK_OPENAI_RESPONSES_OUTPUT[2]?.content?.[0]?.text,
      }),
      expect.objectContaining({
        role: 'assistant',
        content: MOCK_OPENAI_RESPONSES_OUTPUT[3]?.content?.[0]?.text,
      }),
    ]);

    expect(normalizeConversation(MOCK_OPENAI_RESPONSES_STREAMING_OUTPUT)).toEqual([
      expect.objectContaining({
        role: 'assistant',
        tool_calls: [
          prettyPrintToolCall({
            id: MOCK_OPENAI_RESPONSES_STREAMING_OUTPUT[0].item.call_id ?? '',
            function: {
              arguments: MOCK_OPENAI_RESPONSES_STREAMING_OUTPUT[0].item.arguments ?? '',
              name: MOCK_OPENAI_RESPONSES_STREAMING_OUTPUT[0].item.name ?? '',
            },
          }),
        ],
      }),
      expect.objectContaining({
        role: 'tool',
        tool_call_id: MOCK_OPENAI_RESPONSES_STREAMING_OUTPUT[1].item.call_id,
        content: MOCK_OPENAI_RESPONSES_STREAMING_OUTPUT[1].item.output,
      }),
      expect.objectContaining({
        role: 'assistant',
        content: MOCK_OPENAI_RESPONSES_STREAMING_OUTPUT[2]?.item?.content?.[0]?.text,
      }),
    ]);
  });
});

describe('isModelTraceChatTool', () => {
  it('should return true if the object conforms to the tool spec', () => {
    // minimally, all the object needs is a type and a function name
    const tool1 = {
      type: 'function',
      function: {
        name: 'test',
      },
    };

    const tool2 = {
      type: 'function',
      function: {
        name: 'test',
        description: 'test',
        parameters: {
          properties: {
            test: {},
          },
        },
      },
    };

    const tool3 = {
      type: 'function',
      function: {
        name: 'test',
        description: 'test',
        parameters: {
          properties: {
            test: {
              type: 'test',
              description: 'test',
              enum: ['test'],
            },
          },
          required: ['test'],
        },
      },
    };

    expect([tool1, tool2, tool3].every(isModelTraceChatTool)).toBe(true);
  });

  it('should return false if the object does not conform to the tool spec', () => {
    // no `function` key at top level
    const tool1 = {
      type: 'invalid',
      invalid: {
        name: 'test',
      },
    };

    // no name within `function`
    const tool2 = {
      type: 'function',
      function: {
        notName: 'test',
      },
    };

    // enum is not an array
    const tool3 = {
      type: 'function',
      function: {
        name: 'test',
        parameters: {
          properties: {
            test: {
              enum: 'test',
            },
          },
          required: ['test'],
        },
      },
    };

    // required is not an array
    const tool4 = {
      type: 'function',
      function: {
        name: 'test',
        parameters: {
          properties: {
            test: {},
          },
          required: 'test',
        },
      },
    };

    expect([tool1, tool2, tool3, tool4].every((tool) => !isModelTraceChatTool(tool))).toBe(true);
  });
});

describe('normalizeNewSpanData', () => {
  it('should process messages and tools if not contained in attributes', () => {
    const modifiedChatInput = {
      ...MOCK_CHAT_TOOL_CALL_SPAN,
      attributes: {
        ...MOCK_CHAT_TOOL_CALL_SPAN.attributes,
        'mlflow.chat.messages': undefined,
        'mlflow.chat.tools': undefined,
      },
    };

    const normalized = normalizeNewSpanData(modifiedChatInput, 0, 0, [], {}, '');

    const inputMessages = MOCK_OPENAI_CHAT_INPUT.messages;
    const outputMessage = MOCK_OPENAI_CHAT_OUTPUT.choices[0].message;
    const messages = ([...inputMessages, outputMessage] as ModelTraceChatMessage[]).map(prettyPrintChatMessage);
    expect(normalized.chatMessages).toEqual(messages);
    expect(normalized.chatTools).toEqual(MOCK_OPENAI_CHAT_INPUT.tools);
  });

  it('should defer to the attribute value if present', () => {
    const modifiedMessages = [{ role: 'user', content: 'different content' }];
    const modifiedTools = [{ type: 'function', function: { name: 'different tool' } }];

    const modifiedChatInput = {
      ...MOCK_CHAT_TOOL_CALL_SPAN,
      attributes: {
        ...MOCK_CHAT_TOOL_CALL_SPAN.attributes,
        'mlflow.chat.messages': JSON.stringify(modifiedMessages),
        'mlflow.chat.tools': JSON.stringify(modifiedTools),
      },
    };

    const normalized = normalizeNewSpanData(modifiedChatInput, 0, 0, [], {}, '');

    expect(normalized.chatMessages).toEqual(modifiedMessages);
    expect(normalized.chatTools).toEqual(modifiedTools);
  });

  it('return undefined chat messages when either input or output is not chat', () => {
    const inputs = {
      messages: [
        {
          content: 'input',
          role: 'user',
        },
      ],
    };

    const modifiedChatInput = {
      ...MOCK_CHAT_TOOL_CALL_SPAN,
      attributes: {
        ...MOCK_CHAT_TOOL_CALL_SPAN.attributes,
        'mlflow.spanInputs': inputs,
        'mlflow.spanOutputs': {},
        'mlflow.chat.messages': undefined,
        'mlflow.chat.tools': undefined,
      },
    };

    const normalized = normalizeNewSpanData(modifiedChatInput, 0, 0, [], {}, '');

    expect(normalized.chatMessages).toBeUndefined();
  });

  it('should process assessments', () => {
    const rootSpan = MOCK_V3_SPANS[0];
    const childSpan = MOCK_V3_SPANS[1];
    const traceInfo = {
      ...MOCK_TRACE_INFO_V3,
      assessments: [
        MOCK_ROOT_ASSESSMENT,
        MOCK_OVERRIDDING_ASSESSMENT,
        { ...MOCK_SPAN_ASSESSMENT, span_id: decodeSpanId(childSpan.span_id, true) },
      ],
    };
    const assessmentMap = getAssessmentMap(traceInfo);

    const rootNormalized = normalizeNewSpanData(rootSpan, 0, 0, [], assessmentMap, traceInfo.trace_id);
    expect(rootNormalized.assessments).toHaveLength(2);
    expect(rootNormalized.assessments[0]).toEqual(MOCK_ROOT_ASSESSMENT);
    expect(rootNormalized.assessments[1].assessment_id).toEqual(MOCK_OVERRIDDING_ASSESSMENT.assessment_id);
    expect(rootNormalized.assessments[1].overriddenAssessment).toEqual(MOCK_ROOT_ASSESSMENT);

    const childNormalized = normalizeNewSpanData(childSpan, 0, 0, [], assessmentMap, traceInfo.trace_id);
    expect(childNormalized.assessments).toEqual([traceInfo.assessments[2]]);
  });
});

describe('isRawModelTraceChatMessage', () => {
  it('should allow string content and content parts', () => {
    const messages = [
      {
        content: null,
        role: 'user',
      },
      {
        content: 'string',
        role: 'user',
      },
      {
        content: [
          {
            type: 'text',
            text: 'string',
          },
        ],
        role: 'user',
      },
      {
        content: [
          {
            type: 'text',
            text: 'string',
          },
        ],
        role: 'user',
      },
      {
        content: [
          {
            type: 'text',
            text: 'string',
          },
          {
            type: 'image_url',
            image_url: { url: 'data:image/jpeg;base64,aaa' },
          },
        ],
        role: 'user',
      },
      {
        content: [
          {
            type: 'input_audio',
            input_audio: { format: 'wav', data: 'aaa' },
          },
        ],
        role: 'user',
      },
    ];

    expect(messages.every((message) => isRawModelTraceChatMessage(message))).toBe(true);
  });

  it('should not allow invalid objects', () => {
    const messages = [
      {
        content: null,
        role: 'error',
      },
      {
        content: 5,
        role: 'user',
      },
      {
        content: [
          {
            type: 'text',
            text: 6,
          },
        ],
        role: 'user',
      },
      {
        content: [
          {
            type: 'image_url',
            image_url: { url: null },
          },
        ],
        role: 'user',
      },
      {
        content: [
          {
            type: 'input_audio',
            input_audio: { format: 'error', data: 'aaa' },
          },
        ],
        role: 'user',
      },
    ];

    expect(messages.every((message) => !isRawModelTraceChatMessage(message))).toBe(true);
  });
});

describe('isModelTraceChatMessage', () => {
  it('should allow string content', () => {
    const messages = [
      {
        content: null,
        role: 'user',
      },
      {
        content: 'string',
        role: 'user',
      },
    ];

    expect(messages.every((message) => isModelTraceChatMessage(message))).toBe(true);
  });

  it('should not allow invalid objects and content parts', () => {
    const messages = [
      {
        content: null,
        role: 'error',
      },
      {
        content: 5,
        role: 'user',
      },
      {
        content: [
          {
            type: 'text',
            text: 'string',
          },
        ],
        role: 'user',
      },
      {
        content: [
          {
            type: 'text',
            text: 'string',
          },
          {
            type: 'image_url',
            image_url: { url: 'data:image/jpeg;base64,aaa' },
          },
        ],
        role: 'user',
      },
      {
        content: [
          {
            type: 'input_audio',
            input_audio: { format: 'wav', data: 'aaa' },
          },
        ],
        role: 'user',
      },
    ];

    expect(messages.every((message) => !isModelTraceChatMessage(message))).toBe(true);
  });
});

describe('isModelTrace', () => {
  it.each([MOCK_TRACE_INFO_V2, MOCK_TRACE_INFO_V3])(
    'should return true if the object is a model trace',
    (traceInfo) => {
      expect(
        isModelTrace({
          ...MOCK_TRACE,
          info: traceInfo,
        }),
      ).toBe(true);
    },
  );

  it.each([
    { data: { spans: [] }, info: {} },
    { data: { spans: [] }, info: { request_metadata: [] } },
    { data: { spans: [] }, info: { trace_metadata: [] } },
  ])("should return false if the object doesn't have the required fields", (trace) => {
    expect(isModelTrace(trace)).toBe(false);
  });
});
