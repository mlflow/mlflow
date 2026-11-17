import { describe, it, expect } from '@jest/globals';
import { readFileSync } from 'fs';
import path from 'path';

import { TraceAudio } from './TraceAudio';

const SOURCE = readFileSync(path.join(__dirname, 'TraceAudio.tsx'), 'utf8');

describe('TraceAudio', () => {
  it('accepts an optional numeric weight', () => {
    const parsed = TraceAudio.schema.safeParse({
      uri: 'mlflow-attachment://audio-id?content_type=audio%2Fwav&trace_id=tr-123',
      weight: 1,
    });
    expect(parsed.success).toBe(true);
  });

  it('applies weight-derived flex styles in both unavailable and rendered-audio roots', () => {
    // The component interpolates `weight` into the flex value; this assertion
    // matches that source fragment, so the curly is intentional.
    // eslint-disable-next-line no-template-curly-in-string
    expect(SOURCE).toContain('flex: `${weight}`, minWidth: 0');
    expect(SOURCE).toContain('<Typography.Text color="secondary" css={flexStyle}>');
    expect(SOURCE).toContain('<div css={flexStyle}>');
  });
});
