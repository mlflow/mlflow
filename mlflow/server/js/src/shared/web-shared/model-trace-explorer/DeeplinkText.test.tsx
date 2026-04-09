import { parseDescription, parseDeeplinkPath } from './DeeplinkText';
import type { Segment } from './DeeplinkText';

describe('parseDeeplinkPath', () => {
  it('parses fully-qualified path', () => {
    expect(parseDeeplinkPath('/experiments/123/traces/tr-abc/spans/span-def')).toEqual({
      experimentId: '123',
      traceId: 'tr-abc',
      spanId: 'span-def',
    });
  });

  it('parses relative path', () => {
    expect(parseDeeplinkPath('spans/span-abc')).toEqual({
      experimentId: '',
      traceId: '',
      spanId: 'span-abc',
    });
  });

  it('returns null for non-deeplink paths', () => {
    expect(parseDeeplinkPath('https://example.com')).toBeNull();
    expect(parseDeeplinkPath('/some/other/path')).toBeNull();
    expect(parseDeeplinkPath('')).toBeNull();
  });
});

describe('parseDescription', () => {
  it('returns plain text for descriptions without links', () => {
    const result = parseDescription('Just some plain text');
    expect(result).toEqual([{ type: 'text', content: 'Just some plain text' }]);
  });

  it('parses a single deeplink', () => {
    const result = parseDescription('Click [here](spans/span-abc) to see');
    expect(result).toEqual<Segment[]>([
      { type: 'text', content: 'Click ' },
      { type: 'deeplink', text: 'here', spanId: 'span-abc', experimentId: '', traceId: '' },
      { type: 'text', content: ' to see' },
    ]);
  });

  it('parses fully-qualified deeplinks', () => {
    const result = parseDescription('See [this span](/experiments/123/traces/tr-abc/spans/span-def)');
    expect(result).toEqual<Segment[]>([
      { type: 'text', content: 'See ' },
      { type: 'deeplink', text: 'this span', spanId: 'span-def', experimentId: '123', traceId: 'tr-abc' },
    ]);
  });

  it('parses multiple deeplinks', () => {
    const result = parseDescription('First [here](spans/s1) then [there](spans/s2)');
    expect(result).toEqual<Segment[]>([
      { type: 'text', content: 'First ' },
      { type: 'deeplink', text: 'here', spanId: 's1', experimentId: '', traceId: '' },
      { type: 'text', content: ' then ' },
      { type: 'deeplink', text: 'there', spanId: 's2', experimentId: '', traceId: '' },
    ]);
  });

  it('renders non-deeplink markdown links as plain text', () => {
    const result = parseDescription('See [docs](https://example.com) for more');
    expect(result).toEqual<Segment[]>([
      { type: 'text', content: 'See ' },
      { type: 'text', content: '[docs](https://example.com)' },
      { type: 'text', content: ' for more' },
    ]);
  });

  it('handles description with only a deeplink', () => {
    const result = parseDescription('[click](spans/span-1)');
    expect(result).toEqual<Segment[]>([
      { type: 'deeplink', text: 'click', spanId: 'span-1', experimentId: '', traceId: '' },
    ]);
  });

  it('handles empty string', () => {
    expect(parseDescription('')).toEqual([]);
  });
});
