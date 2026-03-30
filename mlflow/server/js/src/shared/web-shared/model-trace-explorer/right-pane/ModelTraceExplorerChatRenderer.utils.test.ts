import {
  CONTENT_TRUNCATION_LIMIT,
  getDisplayLength,
  truncatePreservingImages,
} from './ModelTraceExplorerChatRenderer.utils';
import { describe, it, expect } from '@jest/globals';

describe('getDisplayLength', () => {
  it('returns the length of plain text', () => {
    expect(getDisplayLength('hello world')).toBe(11);
  });

  it('excludes markdown images from the length', () => {
    const content = 'hello ![](http://example.com/img.png) world';
    expect(getDisplayLength(content)).toBe('hello  world'.length);
  });

  it('excludes data URI images from the length', () => {
    const dataUri = 'data:image/png;base64,' + 'A'.repeat(1000);
    const content = `hello ![](${dataUri}) world`;
    expect(getDisplayLength(content)).toBe('hello  world'.length);
  });

  it('handles multiple images', () => {
    const content = 'text ![](url1) middle ![](url2) end';
    expect(getDisplayLength(content)).toBe('text  middle  end'.length);
  });

  it('handles content with no images', () => {
    const content = 'a'.repeat(500);
    expect(getDisplayLength(content)).toBe(500);
  });
});

describe('truncatePreservingImages', () => {
  it('does not truncate content shorter than the limit', () => {
    const content = 'short text';
    expect(truncatePreservingImages(content, CONTENT_TRUNCATION_LIMIT)).toBe('short text');
  });

  it('truncates plain text at the limit', () => {
    const content = 'a'.repeat(500);
    const result = truncatePreservingImages(content, CONTENT_TRUNCATION_LIMIT);
    expect(result).toBe('a'.repeat(300) + '...');
  });

  it('preserves a data URI image when text content is short', () => {
    const dataUri = 'data:image/png;base64,' + 'A'.repeat(1000);
    const content = `Here is my image:\n\n![](${dataUri})`;
    const result = truncatePreservingImages(content, CONTENT_TRUNCATION_LIMIT);
    // Text portion is short, so the image should be fully preserved
    expect(result).toBe(content);
    expect(result).toContain(dataUri);
  });

  it('preserves images and truncates surrounding long text', () => {
    const dataUri = 'data:image/png;base64,' + 'A'.repeat(500);
    const longText = 'b'.repeat(400);
    const content = `![](${dataUri})\n\n${longText}`;
    const result = truncatePreservingImages(content, CONTENT_TRUNCATION_LIMIT);
    // Image should be preserved, text truncated at limit
    expect(result).toContain(dataUri);
    expect(result.endsWith('...')).toBe(true);
  });

  it('truncates text before an image if text alone exceeds the limit', () => {
    const longText = 'c'.repeat(400);
    const content = `${longText} ![](http://example.com/img.png)`;
    const result = truncatePreservingImages(content, CONTENT_TRUNCATION_LIMIT);
    expect(result).toBe('c'.repeat(300) + '...');
    expect(result).not.toContain('![](');
  });

  it('handles multiple images interspersed with text', () => {
    const img1 = '![](data:image/png;base64,AAA)';
    const img2 = '![](data:image/png;base64,BBB)';
    const content = `hello ${img1} world ${img2} end`;
    const result = truncatePreservingImages(content, CONTENT_TRUNCATION_LIMIT);
    // Total text is short, so everything should be preserved
    expect(result).toBe(content);
  });

  it('handles content that is only an image', () => {
    const dataUri = 'data:image/png;base64,' + 'A'.repeat(2000);
    const content = `![](${dataUri})`;
    const result = truncatePreservingImages(content, CONTENT_TRUNCATION_LIMIT);
    expect(result).toBe(content);
  });
});
