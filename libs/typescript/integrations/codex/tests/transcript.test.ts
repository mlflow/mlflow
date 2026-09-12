import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

jest.mock('node:os', () => ({
  ...jest.requireActual<typeof import('node:os')>('node:os'),
  homedir: jest.fn(),
}));

import {
  readTranscript,
  parseTimestampToNs,
  extractTextFromContent,
  findLastUserPrompt,
  getLastTurnRecords,
  getTokenUsage,
  getModel,
  getSessionId,
  buildToolResultMap,
  findTranscriptForThread,
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

describe('extractTextFromContent', () => {
  it('extracts text from content blocks', () => {
    const content = [
      { type: 'output_text' as const, text: 'hello' },
      { type: 'output_text' as const, text: 'world' },
    ];
    expect(extractTextFromContent(content)).toBe('hello\nworld');
  });

  it('returns string content as-is', () => {
    expect(extractTextFromContent('plain text')).toBe('plain text');
  });

  it('returns empty string for undefined', () => {
    expect(extractTextFromContent(undefined)).toBe('');
  });
});

describe('readTranscript + parsing', () => {
  const basicRecords = readTranscript(resolve(FIXTURES_DIR, 'basic.jsonl'));
  const toolRecords = readTranscript(resolve(FIXTURES_DIR, 'with-tool-call.jsonl'));

  it('reads basic transcript', () => {
    expect(basicRecords.length).toBe(6);
  });

  it('reads tool call transcript', () => {
    expect(toolRecords.length).toBe(9);
  });

  it('finds last user prompt', () => {
    const result = findLastUserPrompt(basicRecords);
    expect(result).not.toBeNull();
    expect(result!.text).toBe('what is 2+2');
  });

  it('gets last turn records', () => {
    const turn = getLastTurnRecords(basicRecords);
    expect(turn.length).toBeGreaterThan(0);
    // Should include task_started through task_complete
    expect(turn[0].type).toBe('event_msg');
  });

  it('gets token usage', () => {
    const usage = getTokenUsage(basicRecords);
    expect(usage).not.toBeNull();
    expect(usage!.input_tokens).toBe(100);
    expect(usage!.output_tokens).toBe(10);
    expect(usage!.total_tokens).toBe(110);
  });

  it('gets model from session meta', () => {
    // No model in our fixture, should return unknown
    expect(getModel(basicRecords)).toBe('unknown');
  });

  it('gets session ID', () => {
    expect(getSessionId(basicRecords)).toBe('test-session-001');
  });

  it('builds tool result map', () => {
    const results = buildToolResultMap(toolRecords);
    expect(results['call_abc123']).toBe('file1.txt\nfile2.txt\nfile3.txt');
  });

  it('returns empty tool result map for basic transcript', () => {
    const results = buildToolResultMap(basicRecords);
    expect(Object.keys(results).length).toBe(0);
  });
});

describe('findTranscriptForThread', () => {
  let originalCodexHome: string | undefined;
  let testRoot: string;

  beforeEach(() => {
    originalCodexHome = process.env.CODEX_HOME;
    testRoot = mkdtempSync(join(tmpdir(), 'mlflow-codex-transcript-'));
    jest.useFakeTimers().setSystemTime(new Date(2026, 3, 5, 10));
  });

  afterEach(() => {
    if (originalCodexHome === undefined) {
      delete process.env.CODEX_HOME;
    } else {
      process.env.CODEX_HOME = originalCodexHome;
    }
    jest.mocked(homedir).mockReset();
    jest.useRealTimers();
    rmSync(testRoot, { recursive: true, force: true });
  });

  function writeTranscript(codexHome: string, threadId: string): string {
    const transcriptDir = join(codexHome, 'sessions', '2026', '04', '05');
    mkdirSync(transcriptDir, { recursive: true });
    const transcriptPath = join(transcriptDir, `rollout-2026-04-05T10-00-00-${threadId}.jsonl`);
    writeFileSync(transcriptPath, '{}\n', 'utf-8');
    return transcriptPath;
  }

  it('uses CODEX_HOME instead of the legacy home directory', () => {
    const codexHome = join(testRoot, 'custom-codex-home');
    const legacyHome = join(testRoot, 'legacy-home');
    process.env.CODEX_HOME = codexHome;
    jest.mocked(homedir).mockReturnValue(legacyHome);

    const expectedPath = writeTranscript(codexHome, 'shared-thread-id');
    writeTranscript(join(legacyHome, '.codex'), 'shared-thread-id');

    expect(findTranscriptForThread('shared-thread-id')).toBe(expectedPath);
  });

  it('uses ~/.codex when CODEX_HOME is unset', () => {
    const legacyHome = join(testRoot, 'legacy-home');
    delete process.env.CODEX_HOME;
    jest.mocked(homedir).mockReturnValue(legacyHome);

    const expectedPath = writeTranscript(join(legacyHome, '.codex'), 'legacy-thread-id');

    expect(findTranscriptForThread('legacy-thread-id')).toBe(expectedPath);
  });

  it('uses ~/.codex when CODEX_HOME is empty', () => {
    const legacyHome = join(testRoot, 'legacy-home');
    process.env.CODEX_HOME = '';
    jest.mocked(homedir).mockReturnValue(legacyHome);

    const expectedPath = writeTranscript(join(legacyHome, '.codex'), 'empty-home-thread-id');

    expect(findTranscriptForThread('empty-home-thread-id')).toBe(expectedPath);
  });
});
