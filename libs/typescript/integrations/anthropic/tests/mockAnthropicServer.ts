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
  stream?: boolean;
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
      total_tokens: 384,
    },
    content: [
      {
        type: 'text',
        text: 'This is a mocked Anthropic response.',
      },
    ],
  };
}

function createStreamingEvents(request: MessagesCreateRequest): string[] {
  const messageId = `msg_${Math.random().toString(36).slice(2)}`;
  const responseText = 'This is a mocked streaming response.';

  return [
    `event: message_start\ndata: ${JSON.stringify({
      type: 'message_start',
      message: {
        id: messageId,
        type: 'message',
        role: 'assistant',
        model: request.model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: 128,
          output_tokens: 0
        }
      }
    })}\n\n`,
    `event: content_block_start\ndata: ${JSON.stringify({
      type: 'content_block_start',
      index: 0,
      content_block: {
        type: 'text',
        text: ''
      }
    })}\n\n`,
    `event: content_block_delta\ndata: ${JSON.stringify({
      type: 'content_block_delta',
      index: 0,
      delta: {
        type: 'text_delta',
        text: responseText
      }
    })}\n\n`,
    `event: content_block_stop\ndata: ${JSON.stringify({
      type: 'content_block_stop',
      index: 0
    })}\n\n`,
    `event: message_delta\ndata: ${JSON.stringify({
      type: 'message_delta',
      delta: {
        stop_reason: 'end_turn',
        stop_sequence: null
      },
      usage: {
        output_tokens: 256
      }
    })}\n\n`,
    `event: message_stop\ndata: ${JSON.stringify({
      type: 'message_stop'
    })}\n\n`
  ];
}

export const anthropicMockHandlers = [
  http.post('https://api.anthropic.com/v1/messages', async ({ request }) => {
    const body = (await request.json()) as MessagesCreateRequest;

    // Check if streaming is requested
    if (body.stream) {
      const events = createStreamingEvents(body);
      const encoder = new TextEncoder();

      const stream = new ReadableStream({
        async start(controller) {
          for (const event of events) {
            controller.enqueue(encoder.encode(event));
            // Small delay to simulate streaming
            await new Promise((resolve) => setTimeout(resolve, 10));
          }
          controller.close();
        }
      });

      return new HttpResponse(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive'
        }
      });
    }

    return HttpResponse.json(createMessageResponse(body));
  }),
];
