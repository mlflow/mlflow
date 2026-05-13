/**
 * MSW-based Gemini API mock server for testing
 * Provides realistic Gemini API responses for comprehensive testing
 */

import { http, HttpResponse } from 'msw';

interface GeminiGenerateContentRequest {
  model?: string;
  contents: any;
  config?: any;
}

interface GeminiGenerateContentResponse {
  candidates?: Array<{
    content: {
      parts: Array<{
        text: string;
      }>;
      role: string;
    };
    finishReason: string;
    index: number;
  }>;
  usageMetadata: {
    promptTokenCount: number;
    candidatesTokenCount: number;
    totalTokenCount: number;
  };
  text?: () => string;
}

/**
 * Create a realistic Gemini generateContent response
 */
function createGenerateContentResponse(
  _request: GeminiGenerateContentRequest,
): GeminiGenerateContentResponse {
  const responseText = 'Test response from Gemini';

  return {
    candidates: [
      {
        content: {
          parts: [
            {
              text: responseText,
            },
          ],
          role: 'model',
        },
        finishReason: 'STOP',
        index: 0,
      },
    ],
    usageMetadata: {
      promptTokenCount: 10,
      candidatesTokenCount: 5,
      totalTokenCount: 15,
    },
    text: () => responseText,
  };
}

/**
 * Main MSW handlers for Gemini API endpoints
 */
export const geminiMockHandlers = [
  http.post(
    'https://generativelanguage.googleapis.com/v1beta/models/*\\:generateContent',
    async ({ request }) => {
      const body = (await request.json()) as GeminiGenerateContentRequest;
      return HttpResponse.json(createGenerateContentResponse(body));
    },
  ),

  http.post(
    'https://generativelanguage.googleapis.com/v1/models/*\\:generateContent',
    async ({ request }) => {
      const body = (await request.json()) as GeminiGenerateContentRequest;
      return HttpResponse.json(createGenerateContentResponse(body));
    },
  ),
];
