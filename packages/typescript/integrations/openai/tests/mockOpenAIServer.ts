/**
 * MSW-based OpenAI API mock server for testing
 * Provides realistic OpenAI API responses for comprehensive testing
 */

import { http, HttpResponse } from 'msw';

// Types for OpenAI API responses
interface ChatCompletionRequest {
  model: string;
  messages: Array<{
    role: 'system' | 'user' | 'assistant' | 'tool';
    content: string;
    tool_call_id?: string;
  }>;
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
  tools?: any[];
  response_format?: any;
}

interface ChatCompletionResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: 'assistant';
      content: string | null;
      tool_calls?: any[];
      refusal?: null;
    };
    logprobs?: null;
    finish_reason: 'stop' | 'length' | 'tool_calls' | null;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  service_tier?: string;
  system_fingerprint?: string;
}

// Types for Responses API
interface ResponsesRequest {
  model: string;
  input: string | Array<{ role: string; content: string }>;
  temperature?: number;
}

interface ResponsesResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: 'assistant';
      content: Array<{
        type: 'text';
        text: string;
      }>;
    };
    finish_reason: string;
  }>;
  usage: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  };
}

/**
 * Create a realistic chat completion response
 */
function createChatCompletionResponse(request: ChatCompletionRequest): ChatCompletionResponse {
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
          content: 'Test response content'
        },
        finish_reason: 'stop'
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
function createResponsesResponse(request: ResponsesRequest): ResponsesResponse {
  const timestamp = Math.floor(Date.now() / 1000);

  return {
    id: 'responses-123',
    object: 'chat.completion',
    created: timestamp,
    model: request.model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: [
            {
              type: 'text',
              text: 'Dummy output'
            }
          ]
        },
        finish_reason: 'stop'
      }
    ],
    usage: {
      input_tokens: 36,
      output_tokens: 87,
      total_tokens: 123
    }
  };
}

/**
 * Main MSW handlers for OpenAI API endpoints
 */
export const openAIMockHandlers = [
  // Chat completions (non-streaming)
  http.post('https://api.openai.com/v1/chat/completions', async ({ request }) => {
    const body = (await request.json()) as ChatCompletionRequest;
    return HttpResponse.json(createChatCompletionResponse(body));
  }),

  // Responses API
  http.post('https://api.openai.com/v1/responses', async ({ request }) => {
    const body = (await request.json()) as ResponsesRequest;
    return HttpResponse.json(createResponsesResponse(body));
  })
];
