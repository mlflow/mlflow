import { describe, it, expect } from '@jest/globals';

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

// ADK input with function_call and function_response parts
const MOCK_ADK_INPUT = {
  model: 'gemini-3-flash-preview',
  config: { system_instruction: 'You are a helpful assistant.' },
  contents: [
    { role: 'user', parts: [{ text: 'Hi, I want to learn about Google ADK' }] },
    {
      role: 'model',
      parts: [{ function_call: { name: 'doc_search', args: { query: 'Google ADK overview' } } }],
    },
    {
      role: 'user',
      parts: [
        {
          function_response: {
            name: 'doc_search',
            response: { result: 'Google ADK is a framework for building AI agents.' },
          },
        },
      ],
    },
    { role: 'model', parts: [{ text: 'The Google ADK is a framework for building AI agents.' }] },
  ],
};

// ADK output with direct content (no candidates wrapper)
const MOCK_ADK_OUTPUT = {
  model_version: 'gemini-3-flash-preview',
  content: {
    role: 'model',
    parts: [{ text: 'The Google ADK is a framework for building AI agents.' }],
  },
  finish_reason: 'STOP',
  usage_metadata: {
    prompt_token_count: 100,
    candidates_token_count: 20,
    total_token_count: 120,
  },
};

// ADK output with function_call in direct content
const MOCK_ADK_OUTPUT_WITH_FUNCTION_CALL = {
  model_version: 'gemini-3-flash-preview',
  content: {
    role: 'model',
    parts: [{ function_call: { name: 'doc_search', args: { query: 'Google ADK overview' } } }],
  },
  finish_reason: 'STOP',
};

// Gemini input with inline_data image part
const MOCK_GEMINI_INPUT_WITH_IMAGE = {
  model: 'gemini-2.5-flash',
  contents: [
    {
      role: 'user',
      parts: [{ inline_data: { data: 'iVBORw0KGgo=', mime_type: 'image/jpeg' } }, { text: 'What is in this image?' }],
    },
  ],
};

// Gemini input with inline_data audio part
const MOCK_GEMINI_INPUT_WITH_AUDIO = {
  model: 'gemini-2.5-flash',
  contents: [
    {
      role: 'user',
      parts: [{ inline_data: { data: 'UklGR...', mime_type: 'audio/wav' } }, { text: 'Transcribe this audio' }],
    },
  ],
};

// Gemini input with file_data image part
const MOCK_GEMINI_INPUT_WITH_FILE_DATA = {
  model: 'gemini-2.5-flash',
  contents: [
    {
      role: 'user',
      parts: [
        { file_data: { file_uri: 'gs://bucket/image.png', mime_type: 'image/png' } },
        { text: 'Describe this image' },
      ],
    },
  ],
};

// Flat parts list (e.g., from Part.from_bytes() usage)
const MOCK_GEMINI_FLAT_INPUT_WITH_IMAGE = {
  model: 'gemini-2.5-flash',
  contents: [{ inline_data: { data: 'iVBORw0KGgo=', mime_type: 'image/png' } }, 'Caption this image'],
};

// Inline data with Python b'...' bytes literal format (base64 wrapped in b'...')
const MOCK_GEMINI_INPUT_WITH_BYTES_FORMAT = {
  model: 'gemini-2.5-flash',
  contents: [
    {
      role: 'user',
      parts: [{ inline_data: { data: "b'iVBORw0KGgo='", mime_type: 'image/jpeg' } }, { text: 'Describe this' }],
    },
  ],
};

// Inline data with Python bytes repr containing hex escapes (real Gemini SDK format)
const MOCK_GEMINI_INPUT_WITH_RAW_BYTES = {
  model: 'gemini-2.5-flash',
  contents: [
    {
      inline_data: { data: "b'\\x89PNG\\r\\n\\x1a\\n'", mime_type: 'image/png' },
      text: null,
      file_data: null,
      function_call: null,
      function_response: null,
    },
    'What is in this image?',
  ],
};

const MOCK_GEMINI_OUTPUT_WITH_THINKING = {
  candidates: [
    {
      content: {
        parts: [
          {
            thought: true,
            text: "Let me think about how many r's are in strawberry...",
          },
          {
            thought: null,
            text: 'There are 3 r\'s in the word "strawberry".',
          },
        ],
        role: 'model',
      },
      finish_reason: 'STOP',
    },
  ],
  model_version: 'gemini-2.5-flash',
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

  it('should handle gemini output with thinking content', () => {
    const result = normalizeConversation(MOCK_GEMINI_OUTPUT_WITH_THINKING, 'gemini');
    expect(result).toEqual([
      expect.objectContaining({
        content: expect.stringMatching(/there are 3 r's in the word/i),
        role: 'assistant',
        reasoning: expect.stringMatching(/let me think about how many r's/i),
      }),
    ]);
  });

  it('should handle ADK input with function_call and function_response parts', () => {
    const result = normalizeConversation(MOCK_ADK_INPUT, 'gemini');
    expect(result).not.toBeNull();
    expect(result).toHaveLength(4);

    // First message: user text
    expect(result![0]).toMatchObject({
      role: 'user',
      content: expect.stringMatching(/I want to learn about Google ADK/i),
    });

    // Second message: model function_call → assistant with tool_calls
    expect(result![1]).toMatchObject({
      role: 'assistant',
      tool_calls: [
        {
          id: 'doc_search',
          function: {
            name: 'doc_search',
            arguments: JSON.stringify({ query: 'Google ADK overview' }, null, 2),
          },
        },
      ],
    });

    // Third message: user function_response → tool message
    expect(result![2]).toMatchObject({
      role: 'tool',
      name: 'doc_search',
      content: expect.stringMatching(/Google ADK is a framework/i),
    });

    // Fourth message: model text response
    expect(result![3]).toMatchObject({
      role: 'assistant',
      content: expect.stringMatching(/Google ADK is a framework/i),
    });
  });

  it('should handle ADK output with direct content (no candidates wrapper)', () => {
    const result = normalizeConversation(MOCK_ADK_OUTPUT, 'gemini');
    expect(result).not.toBeNull();
    expect(result).toHaveLength(1);
    expect(result![0]).toMatchObject({
      role: 'assistant',
      content: expect.stringMatching(/Google ADK is a framework/i),
    });
  });

  it('should handle ADK output with function_call in direct content', () => {
    const result = normalizeConversation(MOCK_ADK_OUTPUT_WITH_FUNCTION_CALL, 'gemini');
    expect(result).not.toBeNull();
    expect(result).toHaveLength(1);
    expect(result![0]).toMatchObject({
      role: 'assistant',
      tool_calls: [
        {
          id: 'doc_search',
          function: {
            name: 'doc_search',
            arguments: JSON.stringify({ query: 'Google ADK overview' }, null, 2),
          },
        },
      ],
    });
  });

  it('should handle ADK data without messageFormat via default fallback', () => {
    // When no messageFormat is set (as in ADK traces), the default case should
    // fall back to Gemini normalizers after OpenAI fails
    const result = normalizeConversation(MOCK_ADK_INPUT, undefined);
    expect(result).not.toBeNull();
    expect(result).toHaveLength(4);
    expect(result![0]).toMatchObject({ role: 'user' });
    expect(result![1]).toMatchObject({ role: 'assistant', tool_calls: expect.any(Array) });
    expect(result![2]).toMatchObject({ role: 'tool' });
    expect(result![3]).toMatchObject({ role: 'assistant' });
  });

  it('should handle ADK output without messageFormat via default fallback', () => {
    const result = normalizeConversation(MOCK_ADK_OUTPUT, undefined);
    expect(result).not.toBeNull();
    expect(result).toHaveLength(1);
    expect(result![0]).toMatchObject({
      role: 'assistant',
      content: expect.stringMatching(/Google ADK is a framework/i),
    });
  });

  it('should handle gemini input with inline_data image', () => {
    const result = normalizeConversation(MOCK_GEMINI_INPUT_WITH_IMAGE, 'gemini');
    expect(result).not.toBeNull();
    expect(result).toHaveLength(1);
    expect(result![0]).toMatchObject({
      role: 'user',
      content: expect.stringContaining('![](data:image/jpeg;base64,iVBORw0KGgo=)'),
    });
    expect(result![0].content).toContain('What is in this image?');
  });

  it('should handle gemini input with inline_data audio', () => {
    const result = normalizeConversation(MOCK_GEMINI_INPUT_WITH_AUDIO, 'gemini');
    expect(result).not.toBeNull();
    expect(result).toHaveLength(1);
    expect(result![0]).toMatchObject({
      role: 'user',
      audioParts: [{ data: 'UklGR...', format: 'wav' }],
    });
    expect(result![0].content).toContain('Transcribe this audio');
  });

  it('should handle gemini input with file_data image', () => {
    const result = normalizeConversation(MOCK_GEMINI_INPUT_WITH_FILE_DATA, 'gemini');
    expect(result).not.toBeNull();
    expect(result).toHaveLength(1);
    expect(result![0]).toMatchObject({
      role: 'user',
      content: expect.stringContaining('![](gs://bucket/image.png)'),
    });
    expect(result![0].content).toContain('Describe this image');
  });

  it('should handle flat parts list with inline_data image', () => {
    const result = normalizeConversation(MOCK_GEMINI_FLAT_INPUT_WITH_IMAGE, 'gemini');
    expect(result).not.toBeNull();
    expect(result).toHaveLength(1);
    expect(result![0]).toMatchObject({
      role: 'user',
      content: expect.stringContaining('![](data:image/png;base64,iVBORw0KGgo=)'),
    });
    expect(result![0].content).toContain('Caption this image');
  });

  it('should clean Python bytes literal format from inline_data', () => {
    const result = normalizeConversation(MOCK_GEMINI_INPUT_WITH_BYTES_FORMAT, 'gemini');
    expect(result).not.toBeNull();
    expect(result).toHaveLength(1);
    expect(result![0]).toMatchObject({
      role: 'user',
      content: expect.stringContaining('![](data:image/jpeg;base64,iVBORw0KGgo=)'),
    });
    // Should NOT contain the b'...' wrapper
    expect(result![0].content).not.toContain("b'");
  });

  it('should decode Python bytes repr with hex escapes to base64', () => {
    const result = normalizeConversation(MOCK_GEMINI_INPUT_WITH_RAW_BYTES, 'gemini');
    expect(result).not.toBeNull();
    expect(result).toHaveLength(1);
    expect(result![0]).toMatchObject({
      role: 'user',
      content: expect.stringMatching(/!\[\]\(data:image\/png;base64,[A-Za-z0-9+/]+=*\)/),
    });
    expect(result![0].content).toContain('What is in this image?');
    // Should NOT contain raw Python bytes
    expect(result![0].content).not.toContain('\\x89');
  });
});
