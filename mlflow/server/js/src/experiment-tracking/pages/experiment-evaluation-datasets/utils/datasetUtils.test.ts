import { describe, it, expect } from '@jest/globals';

import { extractDatasetInfoFromTraces } from './datasetUtils';

const SEMCONV_MESSAGES = [{ role: 'user', parts: [{ type: 'text', content: 'hi' }] }];

const wrapTrace = (rootSpan: Record<string, any>) =>
  ({
    info: { request_id: 'tr-1' },
    data: { spans: [rootSpan] },
  }) as any; // Cast to any to avoid bringing full ModelTrace typings into the test

const makeV2Trace = (rootSpanOverrides: Record<string, any>) =>
  wrapTrace({ parent_id: null, attributes: {}, ...rootSpanOverrides });

const makeV3Trace = (attributes: Record<string, string>) =>
  wrapTrace({
    parent_span_id: null,
    start_time_unix_nano: '1',
    end_time_unix_nano: '2',
    attributes,
  });

const makeV4Trace = (attributes: { key: string; value: Record<string, any> }[]) =>
  wrapTrace({
    parent_span_id: null,
    start_time_unix_nano: '1',
    end_time_unix_nano: '2',
    attributes,
  });

describe('extractDatasetInfoFromTraces', () => {
  it('wraps V3 inputs that equal the decoded gen_ai.input.messages attribute', () => {
    const trace = makeV3Trace({
      'mlflow.spanInputs': JSON.stringify(SEMCONV_MESSAGES),
      'gen_ai.input.messages': JSON.stringify(SEMCONV_MESSAGES),
    });

    const [row] = extractDatasetInfoFromTraces([trace]);

    expect(row.inputs).toEqual({ messages: SEMCONV_MESSAGES });
  });

  it('wraps V4 inputs when gen_ai.input.messages is a string value', () => {
    const serialized = JSON.stringify(SEMCONV_MESSAGES);
    const trace = makeV4Trace([
      { key: 'mlflow.spanInputs', value: { string_value: serialized } },
      { key: 'gen_ai.input.messages', value: { string_value: serialized } },
    ]);

    const [row] = extractDatasetInfoFromTraces([trace]);

    expect(row.inputs).toEqual({ messages: SEMCONV_MESSAGES });
  });

  it('wraps V2 inputs that equal the gen_ai.input.messages attribute', () => {
    const trace = makeV2Trace({
      inputs: SEMCONV_MESSAGES,
      attributes: { 'gen_ai.input.messages': JSON.stringify(SEMCONV_MESSAGES) },
    });

    const [row] = extractDatasetInfoFromTraces([trace]);

    expect(row.inputs).toEqual({ messages: SEMCONV_MESSAGES });
  });

  it.each([
    ['the attribute is missing', SEMCONV_MESSAGES, {}],
    [
      'the inputs came from another source (e.g. OpenInference input.value)',
      [{ role: 'user', content: 'hi' }],
      { 'gen_ai.input.messages': JSON.stringify(SEMCONV_MESSAGES) },
    ],
    ['an item is not an object', ['hi'], { 'gen_ai.input.messages': JSON.stringify(['hi']) }],
    [
      'a message role is not a string',
      [{ role: 1, content: 'hi' }],
      { 'gen_ai.input.messages': JSON.stringify([{ role: 1, content: 'hi' }]) },
    ],
    ['the list is empty', [], { 'gen_ai.input.messages': '[]' }],
    ['the inputs are a bare string', 'hello', { 'gen_ai.input.messages': 'hello' }],
  ])('does not wrap inputs when %s', (_description, inputs, attributes) => {
    const trace = makeV2Trace({ inputs, attributes });

    const [row] = extractDatasetInfoFromTraces([trace]);

    expect(row.inputs).toEqual(inputs);
  });

  it('leaves dict-shaped inputs unaffected', () => {
    const trace = makeV2Trace({
      inputs: { question: 'hi' },
      attributes: { 'gen_ai.input.messages': 'ignored' },
    });

    const [row] = extractDatasetInfoFromTraces([trace]);

    expect(row.inputs).toEqual({ question: 'hi' });
  });
});
