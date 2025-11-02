/**
 * MSW-based OpenAI API mock server for testing
 * Provides realistic OpenAI API responses for comprehensive testing
 */

import { http, HttpResponse } from 'msw';
import {
  ChatCompletion,
  ChatCompletionCreateParams,
  CreateEmbeddingResponse,
  EmbeddingCreateParams
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
          refusal: null
        },
        finish_reason: 'stop',
        logprobs: null
      }
    ],
    usage: {
      prompt_tokens: 100,
      completion_tokens: 200,
      total_tokens: 300
    }
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
            annotations: []
          }
        ],
        role: 'assistant',
        status: 'completed',
        type: 'message'
      }
    ],
    usage: {
      input_tokens: 36,
      output_tokens: 87,
      total_tokens: 123,
      input_tokens_details: {
        cached_tokens: 0
      },
      output_tokens_details: {
        reasoning_tokens: 0
      }
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
    tool_choice: 'auto'
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
        .map(() => Math.random() * 0.1 - 0.05)
    })),
    model: request.model,
    usage: {
      prompt_tokens: inputs.length * 10,
      total_tokens: inputs.length * 10
    }
  };
}

/**
 * Main MSW handlers for OpenAI API endpoints
 */
export const openAIMockHandlers = [
  http.post('https://api.openai.com/v1/chat/completions', async ({ request }) => {
    const body = (await request.json()) as ChatCompletionCreateParams;
    return HttpResponse.json(createChatCompletionResponse(body));
  }),
  http.post('https://api.openai.com/v1/responses', async ({ request }) => {
    const body = (await request.json()) as ResponseCreateParams;
    return HttpResponse.json(createResponsesResponse(body));
  }),
  http.post('https://api.openai.com/v1/embeddings', async ({ request }) => {
    const body = (await request.json()) as EmbeddingCreateParams;
    return HttpResponse.json(createEmbeddingResponse(body));
  })
];

export const openAIMswServer = setupServer(...openAIMockHandlers);

export function useMockOpenAIServer(): void {
  beforeAll(() => openAIMswServer.listen());
  afterEach(() => openAIMswServer.resetHandlers());
  afterAll(() => openAIMswServer.close());
}
