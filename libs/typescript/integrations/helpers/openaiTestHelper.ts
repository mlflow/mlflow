/**
 * MSW-based OpenAI API mock server for testing
 * Provides realistic OpenAI API responses for comprehensive testing
 */

import { http, HttpResponse } from 'msw';
import {
  ChatCompletion,
  ChatCompletionCreateParams,
  CreateEmbeddingResponse,
  EmbeddingCreateParams,
} from 'openai/resources/index';
import { ResponseCreateParams, Response } from 'openai/resources/responses/responses';
import { setupServer } from 'msw/node';

/**
 * Create a realistic chat completion response
 */
function createChatCompletionResponse(request: ChatCompletionCreateParams): ChatCompletion {
  const timestamp = Math.floor(Date.now() / 1000);
  const requestId = `chatcmpl-${Math.random().toString(36).substring(2, 15)}`;

  return {
    id: requestId,
    object: 'chat.completion',
    created: timestamp,
    model: request.model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: 'Test response content',
          refusal: null,
        },
        finish_reason: 'stop',
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: 100,
      completion_tokens: 200,
      total_tokens: 300,
    },
  };
}

/**
 * Create a mock response for Responses API
 */
function createResponsesResponse(request: ResponseCreateParams): Response {
  return {
    id: 'responses-123',
    object: 'response',
    model: request.model || '',
    output: [
      {
        id: 'response-123',
        content: [
          {
            type: 'output_text',
            text: 'Dummy output',
            annotations: [],
          },
        ],
        role: 'assistant',
        status: 'completed',
        type: 'message',
      },
    ],
    usage: {
      input_tokens: 36,
      output_tokens: 87,
      total_tokens: 123,
      input_tokens_details: {
        cached_tokens: 0,
      },
      output_tokens_details: {
        reasoning_tokens: 0,
      },
    },
    created_at: 123,
    output_text: 'Dummy output',
    error: null,
    incomplete_details: null,
    instructions: null,
    metadata: null,
    parallel_tool_calls: false,
    temperature: 0.5,
    tools: [],
    top_p: 1,
    tool_choice: 'auto',
  };
}

/**
 * Create a mock response for Embeddings API
 */
function createEmbeddingResponse(request: EmbeddingCreateParams): CreateEmbeddingResponse {
  const inputs = Array.isArray(request.input) ? request.input : [request.input];

  return {
    object: 'list',
    data: inputs.map((_, index) => ({
      object: 'embedding',
      index,
      embedding: Array(1536)
        .fill(0)
        .map(() => Math.random() * 0.1 - 0.05),
    })),
    model: request.model,
    usage: {
      prompt_tokens: inputs.length * 10,
      total_tokens: inputs.length * 10,
    },
  };
}

/**
 * Main MSW handlers for OpenAI API endpoints
 */
export const openAIMockHandlers = [
  http.post('https://api.openai.com/v1/chat/completions', async ({ request }) => {
    const body = (await request.json()) as ChatCompletionCreateParams;
    if (body.stream) {
      return new HttpResponse(
        [
          'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","created":123,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Test "},"finish_reason":null}]}\n\n',
          'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","created":123,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"response","tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather","arguments":"{\\"city\\":"}}]},"finish_reason":null}]}\n\n',
          'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","created":123,"model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"Paris\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n',
          'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","created":123,"model":"gpt-4","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}\n\n',
          'data: [DONE]\n\n',
        ].join(''),
        { headers: { 'Content-Type': 'text/event-stream' } },
      );
    }
    return HttpResponse.json(createChatCompletionResponse(body));
  }),
  http.post('https://api.openai.com/v1/responses', async ({ request }) => {
    const body = (await request.json()) as ResponseCreateParams;
    if ('stream' in body && body.stream) {
      return new HttpResponse(
        [
          'data: {"type":"response.created","sequence_number":0,"response":{"id":"responses-stream","object":"response","created_at":123,"status":"in_progress","model":"gpt-4o","output":[]}}\n\n',
          'data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"msg_123","type":"message","status":"in_progress","role":"assistant","content":[]}}\n\n',
          'data: {"type":"response.output_text.delta","sequence_number":2,"item_id":"msg_123","output_index":0,"content_index":0,"delta":"Test "}\n\n',
          'data: {"type":"response.output_text.delta","sequence_number":3,"item_id":"msg_123","output_index":0,"content_index":0,"delta":"response"}\n\n',
          'data: {"type":"response.output_item.done","sequence_number":4,"output_index":0,"item":{"id":"msg_123","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Test response","annotations":[]}]}}\n\n',
          'data: {"type":"response.output_item.added","sequence_number":5,"output_index":1,"item":{"id":"fc_123","type":"function_call","status":"in_progress","call_id":"call_123","name":"get_weather","arguments":""}}\n\n',
          'data: {"type":"response.function_call_arguments.delta","sequence_number":6,"item_id":"fc_123","output_index":1,"delta":"{\\"city\\":"}\n\n',
          'data: {"type":"response.function_call_arguments.delta","sequence_number":7,"item_id":"fc_123","output_index":1,"delta":"\\"Paris\\"}"}\n\n',
          'data: {"type":"response.output_item.done","sequence_number":8,"output_index":1,"item":{"id":"fc_123","type":"function_call","status":"completed","call_id":"call_123","name":"get_weather","arguments":"{\\"city\\":\\"Paris\\"}"}}\n\n',
          'data: {"type":"response.completed","sequence_number":9,"response":{"id":"responses-stream","object":"response","created_at":123,"status":"completed","model":"gpt-4o","output":[{"id":"msg_123","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Test response","annotations":[]}]},{"id":"fc_123","type":"function_call","status":"completed","call_id":"call_123","name":"get_weather","arguments":"{\\"city\\":\\"Paris\\"}"}],"usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":0}},"output_text":"Test response"}}\n\n',
          'data: [DONE]\n\n',
        ].join(''),
        { headers: { 'Content-Type': 'text/event-stream' } },
      );
    }
    return HttpResponse.json(createResponsesResponse(body));
  }),
  http.post('https://api.openai.com/v1/embeddings', async ({ request }) => {
    const body = (await request.json()) as EmbeddingCreateParams;
    return HttpResponse.json(createEmbeddingResponse(body));
  }),
];

export const openAIMswServer = setupServer(...openAIMockHandlers);

export function useMockOpenAIServer(): void {
  beforeAll(() => openAIMswServer.listen());
  afterEach(() => openAIMswServer.resetHandlers());
  afterAll(() => openAIMswServer.close());
}
