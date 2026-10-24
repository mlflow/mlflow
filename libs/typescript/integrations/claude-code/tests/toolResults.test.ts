import { isBackgroundLaunch, toolResultsInEntry } from '../src/toolResults';
import type { TranscriptEntry } from '../src/types';

function userEntry(
  parts: Array<{ id: string; toolUseResult?: { agentId?: string; status?: string } }>,
  entryToolUseResult?: TranscriptEntry['toolUseResult'],
): TranscriptEntry {
  const content = parts.map((p) => ({
    type: 'tool_result' as const,
    tool_use_id: p.id,
    content: 'ok',
    ...(p.toolUseResult ? { toolUseResult: p.toolUseResult } : {}),
  }));
  return {
    type: 'user',
    message: { role: 'user', content },
    ...(entryToolUseResult ? { toolUseResult: entryToolUseResult } : {}),
  };
}

const BACKGROUND_RECEIPT = { agentId: 'bg1', status: 'async_launched' };

describe('toolResultsInEntry', () => {
  it('applies the entry-level toolUseResult to a single tool_result part', () => {
    const results = toolResultsInEntry(userEntry([{ id: 'a' }], BACKGROUND_RECEIPT));

    expect(results.a.agentId).toBe('bg1');
    expect(isBackgroundLaunch(results.a)).toBe(true);
  });

  it('uses only part-level fields when an entry has several tool_result parts', () => {
    const results = toolResultsInEntry(
      userEntry(
        [{ id: 'a' }, { id: 'b', toolUseResult: { agentId: 'bg2', status: 'async_launched' } }],
        BACKGROUND_RECEIPT,
      ),
    );

    expect(results.a.agentId).toBeUndefined();
    expect(isBackgroundLaunch(results.a)).toBe(false);
    expect(results.b.agentId).toBe('bg2');
    expect(isBackgroundLaunch(results.b)).toBe(true);
  });
});
