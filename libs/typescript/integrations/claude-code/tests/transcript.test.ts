import { resolve } from 'node:path';

import {
  readTranscript,
  parseTimestampToNs,
  extractTextContent,
  findLastUserMessageIndex,
  findFinalAssistantResponse,
} from '../src/transcript';

import type { TranscriptEntry } from '../src/types';

const FIXTURES_DIR = resolve(__dirname, 'fixtures');

// ============================================================================
// readTranscript
// ============================================================================

describe('readTranscript', () => {
  it('parses a basic JSONL file', () => {
    const entries = readTranscript(resolve(FIXTURES_DIR, 'basic.jsonl'));
    expect(entries.length).toBe(5);
    expect(entries[0].type).toBe('user');
    expect(entries[1].type).toBe('assistant');
  });
});

// ============================================================================
// parseTimestampToNs
// ============================================================================

describe('parseTimestampToNs', () => {
  it('parses ISO string', () => {
    const result = parseTimestampToNs('2024-01-15T10:30:45.123Z');
    expect(typeof result).toBe('number');
    expect(result).toBeGreaterThan(0);
  });

  it('converts Unix seconds to nanoseconds', () => {
    const unixTs = 1705312245.123456;
    const result = parseTimestampToNs(unixTs);
    const expected = Math.floor(unixTs * 1e9);
    expect(result).toBe(expected);
  });

  it('converts milliseconds to nanoseconds', () => {
    const msTs = 1705312245123;
    const result = parseTimestampToNs(msTs);
    expect(result).toBe(Math.floor(msTs * 1e6));
  });

  it('returns nanoseconds as-is for large numbers', () => {
    // Use a value that's >= 1e13 (routes through the ns branch) and
    // below Number.MAX_SAFE_INTEGER (~9e15) so no precision is lost.
    const nsTs = 1705312245123456;
    const result = parseTimestampToNs(nsTs);
    expect(result).toBe(Math.floor(nsTs));
  });

  it('returns null for empty/null input', () => {
    expect(parseTimestampToNs(null)).toBeNull();
    expect(parseTimestampToNs(undefined)).toBeNull();
    expect(parseTimestampToNs('')).toBeNull();
  });

  it('returns null for invalid string', () => {
    expect(parseTimestampToNs('not-a-date')).toBeNull();
  });
});

// ============================================================================
// extractTextContent
// ============================================================================

describe('extractTextContent', () => {
  it('extracts text from content block array', () => {
    const content = [
      { type: 'text' as const, text: 'Hello' },
      { type: 'tool_use' as const, id: 'x', name: 'Bash', input: {} },
      { type: 'text' as const, text: 'World' },
    ];
    expect(extractTextContent(content)).toBe('Hello\nWorld');
  });

  it('returns string content directly', () => {
    expect(extractTextContent('plain text')).toBe('plain text');
  });

  it('handles empty array', () => {
    expect(extractTextContent([])).toBe('');
  });
});

// ============================================================================
// findLastUserMessageIndex
// ============================================================================

describe('findLastUserMessageIndex', () => {
  it('finds the last user message in basic transcript', () => {
    const transcript: TranscriptEntry[] = [
      {
        type: 'user',
        message: { role: 'user', content: 'First question' },
        timestamp: '2025-01-01T00:00:00Z',
      },
      {
        type: 'assistant',
        message: { role: 'assistant', content: [{ type: 'text', text: 'First answer' }] },
        timestamp: '2025-01-01T00:00:01Z',
      },
      {
        type: 'user',
        message: { role: 'user', content: 'Second question' },
        timestamp: '2025-01-01T00:00:02Z',
      },
      {
        type: 'assistant',
        message: { role: 'assistant', content: [{ type: 'text', text: 'Second answer' }] },
        timestamp: '2025-01-01T00:00:03Z',
      },
    ];

    const idx = findLastUserMessageIndex(transcript);
    expect(idx).toBe(2);
  });

  it('skips tool result messages', () => {
    const entries = readTranscript(resolve(FIXTURES_DIR, 'basic.jsonl'));
    const idx = findLastUserMessageIndex(entries);
    // Entry 0 is the real user message; entry 3 is a tool_result → skipped
    expect(idx).toBe(0);
  });

  it('skips skill injection messages', () => {
    const transcript: TranscriptEntry[] = [
      {
        type: 'user',
        message: { role: 'user', content: 'Enable tracing on the agent.' },
        timestamp: '2025-01-01T00:00:00Z',
      },
      {
        type: 'assistant',
        message: {
          role: 'assistant',
          content: [
            { type: 'tool_use', id: 'toolu_abc', name: 'Skill', input: { skill: 'my-skill' } },
          ],
        },
        timestamp: '2025-01-01T00:00:01Z',
      },
      {
        type: 'user',
        toolUseResult: { success: true, commandName: 'my-skill' },
        message: {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'toolu_abc',
              content: 'Launching skill: my-skill',
            },
          ],
        },
        timestamp: '2025-01-01T00:00:02Z',
      },
      // Skill content injection — should be skipped
      {
        type: 'user',
        message: {
          role: 'user',
          content: [{ type: 'text', text: 'Base directory: /skill\n# Guide' }],
        },
        timestamp: '2025-01-01T00:00:03Z',
      },
      {
        type: 'assistant',
        message: { role: 'assistant', content: [{ type: 'text', text: 'Done.' }] },
        timestamp: '2025-01-01T00:00:04Z',
      },
    ];

    const idx = findLastUserMessageIndex(transcript);
    expect(idx).toBe(0);
    expect(transcript[idx!].message!.content as string).toBe('Enable tracing on the agent.');
  });

  it('skips compaction summary messages', () => {
    const transcript: TranscriptEntry[] = [
      {
        type: 'user',
        message: { role: 'user', content: 'Real question after context reset' },
        timestamp: '2025-01-01T00:00:00Z',
      },
      {
        type: 'assistant',
        message: { role: 'assistant', content: [{ type: 'text', text: 'Answer' }] },
        timestamp: '2025-01-01T00:00:01Z',
      },
      {
        type: 'user',
        isCompactSummary: true,
        message: { role: 'user', content: 'Summary of prior conversation...' },
        timestamp: '2025-01-01T00:00:02Z',
      },
    ];

    const idx = findLastUserMessageIndex(transcript);
    expect(idx).toBe(0);
  });

  it('returns null for empty transcript', () => {
    expect(findLastUserMessageIndex([])).toBeNull();
  });
});

// ============================================================================
// findFinalAssistantResponse
// ============================================================================

describe('findFinalAssistantResponse', () => {
  it('finds the last text response', () => {
    const entries = readTranscript(resolve(FIXTURES_DIR, 'basic.jsonl'));
    const response = findFinalAssistantResponse(entries, 1);
    expect(response).toBe('The answer is 4.');
  });

  it('returns null when no text response found', () => {
    const transcript: TranscriptEntry[] = [
      {
        type: 'assistant',
        message: {
          role: 'assistant',
          content: [{ type: 'tool_use', id: 'x', name: 'Bash', input: {} }],
        },
      },
    ];
    expect(findFinalAssistantResponse(transcript, 0)).toBeNull();
  });
});
