import { describe, it, expect } from '@jest/globals';

import { parseAttachmentUri } from './attachment-utils';

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

  it('parses a valid size parameter', () => {
    const uri = 'mlflow-attachment://abc-123?content_type=image%2Fpng&trace_id=tr-456&size=1048576';
    const result = parseAttachmentUri(uri);
    expect(result).toEqual({
      attachmentId: 'abc-123',
      contentType: 'image/png',
      traceId: 'tr-456',
      size: 1048576,
    });
  });

  it('omits size when not present in URI', () => {
    const uri = 'mlflow-attachment://abc-123?content_type=image%2Fpng&trace_id=tr-456';
    const result = parseAttachmentUri(uri);
    expect(result).toEqual({
      attachmentId: 'abc-123',
      contentType: 'image/png',
      traceId: 'tr-456',
    });
    expect(result?.size).toBeUndefined();
  });

  it.each(['abc', '1.5', '-1', '0', 'Infinity', 'NaN', ''])('omits invalid size value: %s', (bad) => {
    const uri = `mlflow-attachment://abc-123?content_type=image%2Fpng&trace_id=tr-456&size=${bad}`;
    const result = parseAttachmentUri(uri);
    expect(result?.size).toBeUndefined();
  });
});
