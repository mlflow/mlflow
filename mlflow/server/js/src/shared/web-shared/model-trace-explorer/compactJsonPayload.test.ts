import { afterAll, beforeAll, describe, expect, it } from '@jest/globals';
import { TextDecoder, TextEncoder } from 'util';

import {
  compactJsonPayloadInSteps,
  compactTraceSpanPayloads,
  compactTraceSpanPayloadsInSteps,
} from './compactJsonPayload';

const compact = (text: string) => {
  const steps = compactJsonPayloadInSteps(text);
  let step = steps.next();
  while (!step.done) step = steps.next();
  return step.value;
};

describe('compact JSON payload storage', () => {
  const globals = Object.getOwnPropertyDescriptors(globalThis);
  beforeAll(() => {
    Object.defineProperty(globalThis, 'TextEncoder', { configurable: true, value: TextEncoder });
    Object.defineProperty(globalThis, 'TextDecoder', { configurable: true, value: TextDecoder });
  });
  afterAll(() => {
    for (const key of ['TextEncoder', 'TextDecoder']) {
      if (globals[key]) Object.defineProperty(globalThis, key, globals[key]);
      else Reflect.deleteProperty(globalThis, key);
    }
  });

  it('preserves numeric spelling, whitespace, escapes and exact Unicode code units', () => {
    const text =
      '{ "n": 9007199254740993, "zero": -0, "exp": 1.2300e+09, "text": "' +
      'a'.repeat(1000) +
      ' café 日本語 🙂 e\u0301 \ud800 \udfff' +
      String.raw` \" \\ \u1234" }`;
    const result = compact(text);
    expect(result).not.toBe(text);
    expect(result).not.toMatch(/[\u0080-\uffff]/);
    expect(result).toContain('9007199254740993');
    expect(result).toContain('"zero": -0, "exp": 1.2300e+09');
    expect(JSON.parse(result)).toEqual(JSON.parse(text));
    expect(JSON.parse(result).text).toBe(JSON.parse(text).text);
  });

  it.each(['plain 日本語', '{"broken": "日本語}', '', 'null', 'true', '42', '"café"', '"日本語🙂"', '"\\u65e5"'])(
    'retains unhelpful or malformed input %j',
    (text) => {
      expect(compact(text)).toBe(text);
    },
  );

  it('preserves each half when a surrogate pair crosses a work slice', () => {
    const text = '"' + 'a'.repeat(256 * 1024 - 2) + '🙂' + 'a'.repeat(1000) + '"';
    const steps = compactJsonPayloadInSteps(text);
    expect(steps.next().done).toBe(false);
    expect(steps.next().done).toBe(false);
    const result = steps.next();
    expect(result.done).toBe(true);
    expect(JSON.parse(result.value!)).toBe(JSON.parse(text));
    expect(result.value).toContain('\\ud83d\\ude42');
  });

  it.each(['object', 'OTLP'])('rewrites only parsed payload attributes in %s spans without mutation', (shape) => {
    const value = JSON.stringify({ text: 'a'.repeat(1000) + ' café 日本語 🙂' });
    const attributes: any = {
      'mlflow.spanInputs': value,
      'mlflow.spanOutputs': value,
      'mlflow.chat.messages': value,
      'mlflow.chat.tools': value,
      'custom.raw': value,
      'mlflow.spanType': '日本語',
      malformed: '日本語{',
      native: { value: '🙂' },
    };
    const span: any = {
      name: '日本語 🙂',
      attributes:
        shape === 'object'
          ? attributes
          : Object.entries(attributes).map(([key, value]) => ({ key, value: { string_value: value } })),
    };
    const original = JSON.stringify(span);
    const result: any = compactTraceSpanPayloads(span);
    expect(JSON.stringify(span)).toBe(original);
    expect(result).not.toBe(span);
    const read = (key: string) =>
      shape === 'object'
        ? result.attributes[key]
        : result.attributes.find((a: any) => a.key === key).value.string_value;
    for (const key of ['mlflow.spanInputs', 'mlflow.spanOutputs', 'mlflow.chat.messages', 'mlflow.chat.tools']) {
      expect(read(key)).not.toBe(value);
      expect(JSON.parse(read(key))).toEqual(JSON.parse(value));
    }
    for (const key of ['custom.raw', 'mlflow.spanType', 'malformed', 'native']) {
      expect(read(key)).toEqual(attributes[key]);
    }
    expect(result.name).toBe(span.name);
  });

  it.each([
    null,
    42,
    '日本語',
    {},
    { attributes: null },
    { attributes: [] },
    { attributes: { 'mlflow.spanInputs': { text: '🙂' } } },
  ])('keeps unchanged span identity for %j', (span) => {
    expect(compactTraceSpanPayloads(span)).toBe(span);
  });

  it('can stop stepped work without mutating its source', () => {
    const span = { attributes: { 'mlflow.spanInputs': JSON.stringify({ text: 'a'.repeat(600000) + '🙂' }) } };
    const original = JSON.stringify(span);
    const steps = compactTraceSpanPayloadsInSteps(span);
    expect(steps.next().done).toBe(false);
    steps.return(undefined);
    expect(JSON.stringify(span)).toBe(original);
  });
});
