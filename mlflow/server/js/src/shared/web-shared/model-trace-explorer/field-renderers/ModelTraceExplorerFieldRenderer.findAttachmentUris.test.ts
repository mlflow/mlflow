import { describe, it, expect } from '@jest/globals';

import { findAttachmentUris } from './ModelTraceExplorerFieldRenderer';

const VALID_URI = 'mlflow-attachment://abc-123?content_type=image%2Fpng&trace_id=tr-456';

describe('findAttachmentUris', () => {
  it('extracts a top-level attachment URI string', () => {
    const result = findAttachmentUris(VALID_URI);
    expect(result).toEqual([{ attachmentId: 'abc-123', contentType: 'image/png', traceId: 'tr-456' }]);
  });

  it('extracts attachment URIs from a nested object', () => {
    const result = findAttachmentUris({
      data: [{ b64_json: VALID_URI, revised_prompt: 'a logo' }],
    });
    expect(result).toHaveLength(1);
    expect(result[0].attachmentId).toBe('abc-123');
  });

  it('extracts multiple attachment URIs from arrays', () => {
    const uri2 = 'mlflow-attachment://def-456?content_type=audio%2Fwav&trace_id=tr-789';
    const result = findAttachmentUris([VALID_URI, { nested: uri2 }]);
    expect(result).toHaveLength(2);
  });

  it('returns empty for non-attachment strings', () => {
    expect(findAttachmentUris('hello world')).toEqual([]);
    expect(findAttachmentUris('https://example.com')).toEqual([]);
  });

  it('returns empty for primitives', () => {
    expect(findAttachmentUris(42)).toEqual([]);
    expect(findAttachmentUris(null)).toEqual([]);
    expect(findAttachmentUris(undefined)).toEqual([]);
    expect(findAttachmentUris(true)).toEqual([]);
  });

  it('respects depth limit', () => {
    // Build a structure deeper than MAX_ATTACHMENT_SEARCH_DEPTH (10)
    let deep: unknown = VALID_URI;
    for (let i = 0; i < 15; i++) {
      deep = { nested: deep };
    }
    const result = findAttachmentUris(deep);
    expect(result).toEqual([]);
  });
});
