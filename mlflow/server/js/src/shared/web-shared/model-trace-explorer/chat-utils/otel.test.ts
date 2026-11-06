import { describe, expect, it } from '@jest/globals';

import { normalizeConversation } from '../ModelTraceExplorer.utils';
import { isOtelGenAIChatMessage } from './otel';

describe('normalizeConversation (OTEL GenAI)', () => {
  it('normalizes simple text messages', () => {
    const input = [
      {
        role: 'user',
        parts: [{ type: 'text', content: 'Hello there' }],
      },
      {
        role: 'assistant',
        parts: [{ type: 'text', content: 'Hi! How can I help?' }],
      },
    ];

    const result = normalizeConversation(input);

    expect(result).toEqual([
      expect.objectContaining({ role: 'user', content: 'Hello there' }),
      expect.objectContaining({ role: 'assistant', content: 'Hi! How can I help?' }),
    ]);
  });

  it('normalizes messages with non-standard role casing', () => {
    const input = [
      {
        role: 'System',
        parts: [{ type: 'text', content: 'You are a helpful assistant.' }],
      },
      {
        role: 'User',
        parts: [{ type: 'text', content: 'What is MLflow?' }],
      },
      {
        role: 'Assistant',
        parts: [{ type: 'text', content: 'MLflow is an open-source platform.' }],
      },
    ];

    const result = normalizeConversation(input);

    expect(result).toEqual([
      expect.objectContaining({ role: 'system', content: 'You are a helpful assistant.' }),
      expect.objectContaining({ role: 'user', content: 'What is MLflow?' }),
      expect.objectContaining({ role: 'assistant', content: 'MLflow is an open-source platform.' }),
    ]);
  });

  it('normalizes tool call request and response', () => {
    const input = [
      {
        role: 'assistant',
        parts: [
          { type: 'text', content: 'Let me check the weather.' },
          { type: 'tool_call', id: 'call_weather_1', name: 'get_weather', arguments: { city: 'NYC' } },
        ],
      },
      {
        role: 'assistant',
        parts: [{ type: 'tool_call_response', id: 'call_weather_1', response: { tempC: 22 } }],
      },
    ];

    const result = normalizeConversation(input);
    expect(result).toHaveLength(2);

    // First message: assistant with a tool_call
    expect(result?.[0]).toEqual(
      expect.objectContaining({
        role: 'assistant',
        content: expect.stringContaining('Let me check the weather.'),
        tool_calls: [
          expect.objectContaining({
            id: 'call_weather_1',
            function: expect.objectContaining({ name: 'get_weather', arguments: expect.any(String) }),
          }),
        ],
      }),
    );

    // Second message: tool response mapped to role 'tool'
    expect(result?.[1]).toEqual(
      expect.objectContaining({
        role: 'tool',
        tool_call_id: 'call_weather_1',
        content: expect.stringContaining('tempC'),
      }),
    );
  });

  it('preserves text and collects reasoning from mixed messages', () => {
    const result = normalizeConversation([
      {
        role: 'assistant',
        parts: [
          { type: 'reasoning', content: 'Consider the question.' },
          { type: 'text', content: 'The answer is 42.' },
          { type: 'reasoning', content: 'Check the answer.' },
        ],
      },
    ]);

    expect(result).toEqual([
      expect.objectContaining({
        content: 'The answer is 42.',
        reasoning: 'Consider the question.\n\nCheck the answer.',
      }),
    ]);
  });

  it('normalizes reasoning-only messages', () => {
    expect(
      normalizeConversation([{ role: 'assistant', parts: [{ type: 'reasoning', content: 'Thinking...' }] }]),
    ).toEqual([expect.objectContaining({ role: 'assistant', reasoning: 'Thinking...' })]);
  });

  it.each(['https://example.com/image.png', 'http://example.com/image.png'])(
    'normalizes image URI %s without dropping accompanying text',
    (uri) => {
      const result = normalizeConversation([
        {
          role: 'user',
          parts: [
            { type: 'text', content: 'Describe this.' },
            { type: 'uri', modality: 'image', uri },
          ],
        },
      ]);

      expect(result?.[0].content).toBe(`Describe this.\n\n![](${uri})`);
    },
  );

  it('escapes markdown destination delimiters in image URIs', () => {
    const result = normalizeConversation([
      { role: 'user', parts: [{ type: 'uri', modality: 'image', uri: 'https://example.com/image(1).png' }] },
    ]);

    expect(result?.[0].content).toBe('![](https://example.com/image%281%29.png)');
  });

  it('normalizes inline images and audio using the existing media types', () => {
    const result = normalizeConversation([
      {
        role: 'user',
        parts: [
          { type: 'text', content: 'Compare the media.' },
          { type: 'blob', modality: 'image', mime_type: 'image/png', content: 'aW1hZ2U=' },
          { type: 'blob', modality: 'audio', mime_type: 'audio/wav', content: 'YXVkaW8=' },
          { type: 'blob', modality: 'audio', mime_type: 'audio/mpeg', content: 'YXVkaW8=' },
        ],
      },
    ]);

    expect(result?.[0]).toMatchObject({
      content: 'Compare the media.\n\n![](data:image/png;base64,aW1hZ2U=)',
      audioParts: [
        { data: 'YXVkaW8=', format: 'wav' },
        { data: 'YXVkaW8=', format: 'mp3' },
      ],
    });
  });

  it.each([
    { type: 'uri', modality: 'audio', uri: 'mlflow-attachment://audio-1?content_type=audio%2Fwav&trace_id=trace-1' },
    {
      type: 'blob',
      modality: 'image',
      content: 'mlflow-attachment://image-1?content_type=image%2Fpng&trace_id=trace-1',
    },
  ])('preserves extracted media attachment references', (part) => {
    const result = normalizeConversation([{ role: 'user', parts: [part] }]);

    expect(result?.[0].content).toBe(`![](${'uri' in part ? part.uri : part.content})`);
  });

  it.each([
    { type: 'file', modality: 'image', file_id: 'file-123' },
    { type: 'uri', modality: 'image', uri: 'gs://bucket/image.png' },
    { type: 'uri', modality: 'audio', uri: 'https://example.com/audio.wav' },
    { type: 'blob', modality: 'video', mime_type: 'video/mp4', content: 'dmlkZW8=' },
    { type: 'blob', modality: 'audio', mime_type: 'audio/ogg', content: 'YXVkaW8=' },
    { type: 'blob', modality: 'image', content: 'aW1hZ2U=' },
    { type: 'custom_part', content: { nested: 'preserved' } },
    { type: 'server_tool_call', name: 'web_search', server_tool_call: { type: 'web_search', query: 'MLflow' } },
    { type: 'server_tool_call_response', server_tool_call_response: { type: 'web_search', results: ['MLflow'] } },
  ])('keeps unsupported or provider-specific content inspectable alongside text: $type', (part) => {
    const result = normalizeConversation([
      { role: 'user', parts: [{ type: 'text', content: 'Keep this text.' }, part] },
    ]);

    expect(result).toHaveLength(1);
    expect(result?.[0].content).toBe(`Keep this text.\n\n\`\`\`json\n${JSON.stringify(part, null, 2)}\n\`\`\``);
  });

  it.each([
    // eslint-disable-next-line no-script-url
    { type: 'uri', modality: 'image', uri: 'javascript:alert(1)' },
    { type: 'uri', modality: 'image', uri: 'data:text/html;base64,PGgxPkhlbGxvPC9oMT4=' },
    { type: 'blob', modality: 'image', mime_type: 'image/svg+xml', content: 'PHN2Zy8+' },
    { type: 'blob', modality: 'image', mime_type: 'image/png', content: 'not base64!' },
  ])('renders unsupported media as data rather than an embedded image', (part) => {
    const result = normalizeConversation([{ role: 'user', parts: [part] }]);

    expect(result?.[0].content).toContain(JSON.stringify(part, null, 2));
    expect(result?.[0].content).not.toContain('![](');
  });

  it.each([undefined, null])('keeps tool calls compatible when the ID is %s', (id) => {
    const result = normalizeConversation([
      { role: 'assistant', parts: [{ type: 'tool_call', id, name: 'get_weather', arguments: { city: 'Paris' } }] },
      { role: 'tool', parts: [{ type: 'tool_call_response', id, response: { tempC: 22 } }] },
    ]);

    expect(result?.[0].tool_calls?.[0]).toMatchObject({ id: 'get_weather', function: { name: 'get_weather' } });
    expect(result?.[1]).toMatchObject({ role: 'tool', tool_call_id: '', content: '{"tempC":22}' });
  });

  it('preserves mixed text and multiple tool responses instead of retaining only the last response', () => {
    const result = normalizeConversation([
      {
        role: 'tool',
        parts: [
          { type: 'text', content: 'Results from both tools.' },
          { type: 'tool_call_response', id: 'first', response: { first: 'result' } },
          { type: 'tool_call_response', id: 'second', response: { second: 'result' } },
        ],
      },
    ]);

    expect(result?.[0].content).toContain('Results from both tools.');
    expect(result?.[0].content).toContain('"first": "result"');
    expect(result?.[0].content).toContain('"second": "result"');
  });

  it('accepts custom roles and empty part arrays allowed by the message schema', () => {
    expect(isOtelGenAIChatMessage({ role: 'critic', parts: [] })).toBe(true);
    expect(normalizeConversation([{ role: 'critic', parts: [{ type: 'text', content: 'Review complete.' }] }])).toEqual(
      [expect.objectContaining({ role: 'critic', content: 'Review complete.' })],
    );
  });

  it.each([null, 'text', {}, { type: 123 }])('rejects parts without a string type', (part) => {
    expect(isOtelGenAIChatMessage({ role: 'user', parts: [part] })).toBe(false);
  });
});
