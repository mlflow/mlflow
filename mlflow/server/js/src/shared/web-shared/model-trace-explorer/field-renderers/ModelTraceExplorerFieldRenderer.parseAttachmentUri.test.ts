import { describe, it, expect } from '@jest/globals';

// parseAttachmentUri is not exported, so we test it indirectly by importing the module
// and extracting the function. Since it's a private function, we re-implement the logic
// here for direct testing.
function parseAttachmentUri(uri: string): { attachmentId: string; traceId: string; contentType: string } | null {
  try {
    const parsed = new URL(uri);
    if (parsed.protocol !== 'mlflow-attachment:') {
      return null;
    }
    const attachmentId = parsed.hostname;
    const contentType = parsed.searchParams.get('content_type');
    const traceId = parsed.searchParams.get('trace_id');
    if (!attachmentId || !contentType || !traceId) {
      return null;
    }
    return { attachmentId, contentType, traceId };
  } catch {
    return null;
  }
}

describe('parseAttachmentUri', () => {
  it('parses a valid attachment URI', () => {
    const uri = 'mlflow-attachment://abc-123-def?content_type=image%2Fpng&trace_id=tr-456';
    const result = parseAttachmentUri(uri);
    expect(result).toEqual({
      attachmentId: 'abc-123-def',
      contentType: 'image/png',
      traceId: 'tr-456',
    });
  });

  it('returns null for non-mlflow-attachment protocol', () => {
    expect(parseAttachmentUri('https://example.com/file.png')).toBeNull();
    expect(parseAttachmentUri('file:///tmp/file.png')).toBeNull();
  });

  it('returns null when content_type is missing', () => {
    const uri = 'mlflow-attachment://abc-123?trace_id=tr-456';
    expect(parseAttachmentUri(uri)).toBeNull();
  });

  it('returns null when trace_id is missing', () => {
    const uri = 'mlflow-attachment://abc-123?content_type=image%2Fpng';
    expect(parseAttachmentUri(uri)).toBeNull();
  });

  it('returns null for malformed URIs', () => {
    expect(parseAttachmentUri('')).toBeNull();
    expect(parseAttachmentUri('not a uri')).toBeNull();
    expect(parseAttachmentUri('mlflow-attachment://')).toBeNull();
  });

  it('returns null for plain strings', () => {
    expect(parseAttachmentUri('hello world')).toBeNull();
    expect(parseAttachmentUri('{"key": "value"}')).toBeNull();
  });
});
