import { describe, it, expect } from '@jest/globals';

import { deriveSection } from './ReviewTraceInputOutput';

describe('deriveSection', () => {
  it('renders a parsed conversation as chat messages', () => {
    const section = deriveSection(JSON.stringify({ messages: [{ role: 'user', content: 'hi' }] }));
    expect(section.kind).toBe('chat');
    expect(section.kind === 'chat' && section.messages.length).toBeGreaterThan(0);
  });

  it('recognizes an Anthropic-shaped conversation (not just the default formats)', () => {
    const section = deriveSection(
      JSON.stringify({ messages: [{ role: 'user', content: [{ type: 'text', text: 'hi' }] }] }),
    );
    expect(section.kind).toBe('chat');
  });

  it('renders a plain-text payload as markdown', () => {
    expect(deriveSection('## Heading\n\nsome **bold** text').kind).toBe('markdown');
  });

  it('renders a structured, non-conversation payload as JSON', () => {
    expect(deriveSection('{"question": "what is mlflow?"}').kind).toBe('json');
  });

  it('renders a truncated/JSON-looking string as JSON, not markdown', () => {
    // A payload truncated past the server preview cap: invalid JSON but clearly
    // structured — must not be mangled by the markdown renderer.
    expect(deriveSection('{"messages": [{"role": "user", "content": "a very long prompt that got cut o').kind).toBe(
      'json',
    );
  });
});
