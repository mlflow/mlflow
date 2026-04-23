import { resolve } from 'path';

import {
  buildToolResultMap,
  formatResultDisplay,
  getFunctionCalls,
  getLastTurnRecords,
  getMessageText,
  getTokenUsage,
  getToolOutput,
  isFunctionCallPart,
  isTextPart,
  parseTimestampToNs,
  readTranscript,
} from '../src/transcript';
import type { ChatRecord, GeminiPart } from '../src/types';

const FIXTURES_DIR = resolve(__dirname, 'fixtures');

describe('parseTimestampToNs', () => {
  it('parses ISO timestamp', () => {
    const result = parseTimestampToNs('2026-04-05T10:00:00Z');
    expect(result).toBeGreaterThan(0);
    expect(typeof result).toBe('number');
  });

  it('returns null for missing/empty input', () => {
    expect(parseTimestampToNs(null)).toBeNull();
    expect(parseTimestampToNs(undefined)).toBeNull();
    expect(parseTimestampToNs('')).toBeNull();
  });
});

describe('getMessageText', () => {
  function rec(message: ChatRecord['message']): ChatRecord {
    return {
      uuid: 'u',
      parentUuid: null,
      sessionId: 's',
      timestamp: '',
      type: 'assistant',
      message,
    };
  }

  it('joins non-thought text parts', () => {
    expect(
      getMessageText(rec({ role: 'model', parts: [{ text: 'hello' }, { text: 'world' }] })),
    ).toBe('hello\nworld');
  });

  it('excludes thought parts by default', () => {
    const msg = {
      role: 'model',
      parts: [{ text: 'internal', thought: true }, { text: 'visible' }],
    };
    expect(getMessageText(rec(msg))).toBe('visible');
  });

  it('includes thought parts when includeThoughts is true', () => {
    const msg = {
      role: 'model',
      parts: [{ text: 'internal', thought: true }, { text: 'visible' }],
    };
    expect(getMessageText(rec(msg), true)).toBe('internal\nvisible');
  });

  it('accepts plain string messages', () => {
    expect(getMessageText(rec('plain text'))).toBe('plain text');
  });

  it('skips functionCall and functionResponse parts (they contribute no text)', () => {
    const msg = {
      role: 'model',
      parts: [{ functionCall: { id: 'c1', name: 'ls' } }, { text: 'hello' }] as GeminiPart[],
    };
    expect(getMessageText(rec(msg))).toBe('hello');
  });
});

describe('getFunctionCalls', () => {
  it('extracts all functionCall parts from an assistant message', () => {
    const record: ChatRecord = {
      uuid: 'u',
      parentUuid: null,
      sessionId: 's',
      timestamp: '',
      type: 'assistant',
      message: {
        role: 'model',
        parts: [
          { text: 'thinking', thought: true },
          { functionCall: { id: 'c1', name: 'ls', args: { path: '.' } } },
          { functionCall: { id: 'c2', name: 'stat', args: { path: 'a.txt' } } },
        ] as GeminiPart[],
      },
    };
    const calls = getFunctionCalls(record);
    expect(calls.map((c) => c.name)).toEqual(['ls', 'stat']);
    expect(calls[0].args).toEqual({ path: '.' });
  });

  it('returns [] for records with no message or no function calls', () => {
    expect(
      getFunctionCalls({
        uuid: 'u',
        parentUuid: null,
        sessionId: 's',
        timestamp: '',
        type: 'system',
      }),
    ).toEqual([]);
  });
});

describe('part type guards', () => {
  it('isTextPart recognizes text parts', () => {
    expect(isTextPart({ text: 'x' })).toBe(true);
    expect(isTextPart({ functionCall: { id: 'c', name: 'n' } } as GeminiPart)).toBe(false);
  });
  it('isFunctionCallPart recognizes functionCall parts', () => {
    expect(isFunctionCallPart({ functionCall: { id: 'c', name: 'n' } })).toBe(true);
    expect(isFunctionCallPart({ text: 'x' } as GeminiPart)).toBe(false);
  });
});

describe('buildToolResultMap', () => {
  it('indexes tool_result records by callId', () => {
    const records: ChatRecord[] = [
      {
        uuid: 't',
        parentUuid: null,
        sessionId: 's',
        timestamp: '',
        type: 'tool_result',
        toolCallResult: { callId: 'c1', status: 'success', resultDisplay: 'ok' },
      },
      {
        uuid: 'u',
        parentUuid: null,
        sessionId: 's',
        timestamp: '',
        type: 'user',
        message: { role: 'user', parts: [{ text: 'q' }] },
      },
    ];
    const map = buildToolResultMap(records);
    expect(map.size).toBe(1);
    expect(map.get('c1')?.toolCallResult?.status).toBe('success');
  });
});

describe('formatResultDisplay', () => {
  it('passes through strings unchanged', () => {
    expect(formatResultDisplay('hello')).toBe('hello');
  });
  it('JSON-stringifies objects', () => {
    expect(formatResultDisplay({ a: 1 })).toBe('{"a":1}');
  });
  it('returns empty string for null/undefined', () => {
    expect(formatResultDisplay(null)).toBe('');
    expect(formatResultDisplay(undefined)).toBe('');
  });
});

describe('getToolOutput', () => {
  function toolResult(overrides: Partial<ChatRecord>): ChatRecord {
    return {
      uuid: 't',
      parentUuid: null,
      sessionId: 's',
      timestamp: '',
      type: 'tool_result',
      ...overrides,
    } as ChatRecord;
  }

  it('prefers toolCallResult.resultDisplay', () => {
    const record = toolResult({
      toolCallResult: { callId: 'c1', status: 'success', resultDisplay: 'shown' },
      message: {
        role: 'user',
        parts: [{ functionResponse: { id: 'c1', name: 'ls', response: { output: 'raw' } } }],
      },
    });
    expect(getToolOutput(record)).toBe('shown');
  });

  it('falls back to message.parts[].functionResponse.response when resultDisplay is missing', () => {
    const record = toolResult({
      toolCallResult: { callId: 'c1', status: 'success' },
      message: {
        role: 'user',
        parts: [{ functionResponse: { id: 'c1', name: 'ls', response: { output: 'raw' } } }],
      },
    });
    expect(getToolOutput(record)).toBe('{"output":"raw"}');
  });

  it('returns empty string when neither resultDisplay nor functionResponse is available', () => {
    expect(getToolOutput(toolResult({ toolCallResult: { callId: 'c1', status: 'success' } }))).toBe(
      '',
    );
  });
});

describe('readTranscript + turn slicing', () => {
  const basicRecords = readTranscript(resolve(FIXTURES_DIR, 'basic.jsonl'));
  const toolRecords = readTranscript(resolve(FIXTURES_DIR, 'with-tool-call.jsonl'));

  it('reads basic transcript (user → system → assistant)', () => {
    expect(basicRecords.length).toBe(3);
  });

  it('reads tool-call transcript', () => {
    expect(toolRecords.length).toBeGreaterThan(3);
  });

  it('getLastTurnRecords starts at the last user record', () => {
    const turn = getLastTurnRecords(toolRecords);
    expect(turn[0].type).toBe('user');
  });

  it('gets token usage from assistant record', () => {
    const assistant = basicRecords.find((r) => r.type === 'assistant');
    const usage = getTokenUsage(assistant?.usageMetadata);
    expect(usage).not.toBeNull();
    expect(usage!.input).toBe(100);
    expect(usage!.output).toBe(10);
    expect(usage!.total).toBe(110);
  });

  it('returns null for missing usage', () => {
    expect(getTokenUsage(undefined)).toBeNull();
  });
});
