import { normalizeConversation } from '../ModelTraceExplorer.utils';

const MOCK_GEMINI_INPUT = {
  model: 'gemini-2.5-flash',
  contents: 'Explain how AI works in a few words',
  config: null,
};

const MOCK_GEMINI_OUTPUT = {
  sdk_http_response: {
    headers: {
      'content-type': 'application/json; charset=UTF-8',
      vary: 'Origin, X-Origin, Referer',
      'content-encoding': 'gzip',
      date: 'Fri, 18 Jul 2025 12:28:55 GMT',
      server: 'scaffolding on HTTPServer2',
      'x-xss-protection': '0',
      'x-frame-options': 'SAMEORIGIN',
      'x-content-type-options': 'nosniff',
      'server-timing': 'gfet4t7; dur=5179',
      'alt-svc': 'h3=":443"; ma=2592000,h3-29=":443"; ma=2592000',
      'transfer-encoding': 'chunked',
    },
    body: null,
  },
  candidates: [
    {
      content: {
        parts: [
          {
            video_metadata: null,
            thought: null,
            inline_data: null,
            file_data: null,
            thought_signature: null,
            code_execution_result: null,
            executable_code: null,
            function_call: null,
            function_response: null,
            text: 'AI learns patterns from data to make decisions.',
          },
        ],
        role: 'model',
      },
      citation_metadata: null,
      finish_message: null,
      token_count: null,
      finish_reason: 'STOP',
      url_context_metadata: null,
      avg_logprobs: null,
      grounding_metadata: null,
      index: 0,
      logprobs_result: null,
      safety_ratings: null,
    },
  ],
  create_time: null,
  response_id: null,
  model_version: 'gemini-2.5-flash',
  prompt_feedback: null,
  usage_metadata: {
    cache_tokens_details: null,
    cached_content_token_count: null,
    candidates_token_count: 9,
    candidates_tokens_details: null,
    prompt_token_count: 9,
    prompt_tokens_details: [
      {
        modality: 'TEXT',
        token_count: 9,
      },
    ],
    thoughts_token_count: 881,
    tool_use_prompt_token_count: null,
    tool_use_prompt_tokens_details: null,
    total_token_count: 899,
    traffic_type: null,
  },
  automatic_function_calling_history: null,
  parsed: null,
};

describe('normalizeConversation', () => {
  it('should handle gemini input', () => {
    expect(normalizeConversation(MOCK_GEMINI_INPUT, 'gemini')).toEqual([
      expect.objectContaining({
        role: 'user',
        content: expect.stringMatching(/explain how ai works/i),
      }),
    ]);
  });

  it('should handle gemini output', () => {
    expect(normalizeConversation(MOCK_GEMINI_OUTPUT, 'gemini')).toEqual([
      expect.objectContaining({
        content: expect.stringMatching(/ai learns patterns from data to make decisions/i),
        role: 'assistant',
      }),
    ]);
  });
});
