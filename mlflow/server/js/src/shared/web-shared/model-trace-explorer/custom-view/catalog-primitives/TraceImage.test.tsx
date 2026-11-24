import { describe, it, expect } from '@jest/globals';
import { readFileSync } from 'fs';
import path from 'path';

import { TraceImage } from './TraceImage';

const SOURCE = readFileSync(path.join(__dirname, 'TraceImage.tsx'), 'utf8');

describe('TraceImage', () => {
  it('accepts an optional numeric weight', () => {
    const parsed = TraceImage.schema.safeParse({
      uri: 'mlflow-attachment://image-id?content_type=image%2Fjpeg&trace_id=tr-123',
      weight: 1,
    });
    expect(parsed.success).toBe(true);
  });

  it('applies weight-derived flex styles in both unavailable and rendered-image roots', () => {
    // The component interpolates `weight` into the flex value; this assertion
    // matches that source fragment, so the curly is intentional.
    // eslint-disable-next-line no-template-curly-in-string
    expect(SOURCE).toContain('flex: `${weight}`, minWidth: 0');
    expect(SOURCE).toContain('<Typography.Text color="secondary" css={flexStyle}>');
    expect(SOURCE).toContain('<div css={flexStyle}>');
  });
});
