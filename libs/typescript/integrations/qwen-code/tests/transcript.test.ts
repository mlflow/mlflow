import { resolve } from 'path';

import {
  readTranscript,
  parseTimestampToNs,
  getMessageText,
  findLastUserRecord,
  buildRecordTree,
  getTokenUsage,
} from '../src/transcript';

const FIXTURES_DIR = resolve(__dirname, 'fixtures');

describe('parseTimestampToNs', () => {
  it('parses ISO timestamp', () => {
    const result = parseTimestampToNs('2026-04-05T10:00:00Z');
    expect(result).toBeGreaterThan(0);
    expect(typeof result).toBe('number');
  });

  it('returns null for empty input', () => {
    expect(parseTimestampToNs(null)).toBeNull();
    expect(parseTimestampToNs(undefined)).toBeNull();
    expect(parseTimestampToNs('')).toBeNull();
  });
});

describe('getMessageText', () => {
  it('extracts text from Gemini-style parts', () => {
    const record = {
      uuid: '1',
      parentUuid: null,
      sessionId: 's',
      timestamp: '',
      type: 'user' as const,
      message: { role: 'user', parts: [{ text: 'hello' }, { text: 'world' }] },
    };
    expect(getMessageText(record)).toBe('hello\nworld');
  });

  it('returns plain string message', () => {
    const record = {
      uuid: '1',
      parentUuid: null,
      sessionId: 's',
      timestamp: '',
      type: 'user' as const,
      message: 'plain text',
    };
    expect(getMessageText(record)).toBe('plain text');
  });
});

describe('readTranscript + parsing', () => {
  const basicRecords = readTranscript(resolve(FIXTURES_DIR, 'basic.jsonl'));
  const toolRecords = readTranscript(resolve(FIXTURES_DIR, 'with-tool-call.jsonl'));

  it('reads basic transcript', () => {
    expect(basicRecords.length).toBe(2);
  });

  it('reads tool call transcript', () => {
    expect(toolRecords.length).toBe(3);
  });

  it('finds last user record', () => {
    const record = findLastUserRecord(basicRecords);
    expect(record).not.toBeNull();
    expect(getMessageText(record!)).toBe('what is 2+2');
  });

  it('builds record tree', () => {
    const { byUuid, children } = buildRecordTree(basicRecords);
    expect(byUuid.size).toBe(2);
    expect(children.get('user-001')).toContain('assistant-001');
  });

  it('gets token usage', () => {
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

  it('finds tool call result in tree', () => {
    const { children } = buildRecordTree(toolRecords);
    // user-001 should have tool-001 and assistant-001 as children
    const userChildren = children.get('user-001');
    expect(userChildren).toContain('tool-001');
    expect(userChildren).toContain('assistant-001');
  });
});
