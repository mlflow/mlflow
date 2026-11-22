import { afterAll, beforeAll, describe, expect, it } from '@jest/globals';
import { TextDecoder, TextEncoder } from 'util';

import { compactTraceSpanPayloads } from './compactJsonPayload';
import { MOCK_V3_SPANS } from './ModelTraceExplorer.test-utils';
import { normalizeNewSpanData as normalizeV1 } from './ModelTraceExplorer.utils';
import { normalizeNewSpanData as normalizeV2 } from './v2/ModelTraceExplorer.utils';

describe('compacted payload consumers', () => {
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

  it.each([normalizeV1, normalizeV2])(
    'preserves normalized inputs, outputs, chat and custom attributes',
    (normalize) => {
      const content = 'a'.repeat(1000) + ' café 日本語 🙂 e\u0301 \ud800';
      const span = {
        ...MOCK_V3_SPANS[0],
        type: undefined,
        attributes: {
          'mlflow.spanType': 'CHAT_MODEL',
          'mlflow.spanInputs': JSON.stringify({ messages: [{ role: 'user', content }] }),
          'mlflow.spanOutputs': JSON.stringify({ choices: [{ message: { role: 'assistant', content } }] }),
          'mlflow.chat.messages': JSON.stringify([
            { role: 'user', content },
            { role: 'assistant', content },
          ]),
          'mlflow.chat.tools': JSON.stringify([
            { type: 'function', function: { name: 'tool', description: content, parameters: {} } },
          ]),
          'custom.raw': content,
        },
      };
      const compacted = compactTraceSpanPayloads(span) as typeof span;
      expect(compacted).not.toBe(span);
      expect(normalize(compacted, 0, 1000, [], {}, 'owned')).toEqual(normalize(span, 0, 1000, [], {}, 'owned'));
    },
  );
});
