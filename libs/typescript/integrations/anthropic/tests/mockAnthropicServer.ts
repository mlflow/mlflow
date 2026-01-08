/**
 * MSW-based Anthropic API mock server for testing
 */

import { http, HttpResponse } from 'msw';

interface MessagesCreateRequest {
  model: string;
  messages: Array<{
    role: string;
    content: string | Array<{ type: string; text: string }>;
  }>;
  max_tokens?: number;
  system?: string | Array<{ type: string; text: string }>;
}

function createMessageResponse(request: MessagesCreateRequest) {
  return {
    id: `msg_${Math.random().toString(36).slice(2)}`,
    type: 'message',
    role: 'assistant',
    model: request.model,
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: {
      input_tokens: 128,
      output_tokens: 256,
      total_tokens: 384
    },
    content: [
      {
        type: 'text',
        text: 'This is a mocked Anthropic response.'
      }
    ]
  };
}

export const anthropicMockHandlers = [
  http.post('https://api.anthropic.com/v1/messages', async ({ request }) => {
    const body = (await request.json()) as MessagesCreateRequest;
    return HttpResponse.json(createMessageResponse(body));
  })
];
