import { describe, expect, it } from '@jest/globals';

import { mayBeJsonText } from './attributeJson';

describe('mayBeJsonText', () => {
  it.each(['', 'CHAIN', 'falsehood', 'nullish', 'true false', 'provider:model', '\u00a0null', '\ufeff{}'])(
    'rejects an impossible JSON prefix: %j',
    (text) => expect(mayBeJsonText(text)).toBe(false),
  );

  it.each(['null', 'true', 'false', ' \t\r\nnull \n', '"hello"', '{', '[', '-0', '1e-5'])(
    'leaves JSON-looking text to the parser: %j',
    (text) => expect(mayBeJsonText(text)).toBe(true),
  );

  it('rejects native values that are already decoded', () => {
    expect(mayBeJsonText(undefined)).toBe(false);
    for (const value of [null, false, 42, [], [5], [false], {}, { toString: () => 'true' }]) {
      expect(mayBeJsonText(value)).toBe(false);
    }
  });

  it('never rejects valid JSON with arbitrary content and JSON whitespace', () => {
    let seed = 42;
    const random = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed;
    };
    const alphabet = ['a', '"', '\\', '\n', '\t', 'é', '🙂', '\ud800', '\u00a0', '\ufeff'];
    for (let index = 0; index < 5000; index++) {
      const text = Array.from({ length: random() % 30 }, () => alphabet[random() % alphabet.length]).join('');
      const value = [text, { [text]: index }, [text, null, true], index - 2500][index % 4];
      const json = [' ', '\t', '\n', '\r'][index % 4] + JSON.stringify(value) + '\n';
      expect(mayBeJsonText(json)).toBe(true);
      expect(JSON.parse(json)).toEqual(value);
    }
  });
});
