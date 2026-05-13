/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { useCallback } from 'react';
import sanitizeHtml from 'sanitize-html';
// @ts-expect-error TS(7016): Could not find a declaration file for module 'show... Remove this comment to see the full error message
import { Converter } from 'showdown';

// Use Github-like Markdown (i.e. support for tasklists, strikethrough,
// simple line breaks, code blocks, emojis)
const DEFAULT_MARKDOWN_FLAVOR = 'github';

let _converter: Converter | null = null;

export const getMarkdownConverter = () => {
  // Reuse the same converter instance if available
  if (_converter) {
    return _converter;
  }
  _converter = new Converter();
  _converter.setFlavor(DEFAULT_MARKDOWN_FLAVOR);
  return _converter;
};

// Options for HTML sanitizer.
// See https://www.npmjs.com/package/sanitize-html#what-are-the-default-options for usage.
// These options were chosen to be similar to Github's allowlist but simpler (i.e. we don't
// do any transforms of the contained HTML and we disallow script entirely instead of
// removing contents).
const sanitizerOptions = {
  allowedTags: [
    'h1',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
    'h7',
    'h8',
    'blockquote',
    'p',
    'a',
    'ul',
    'ol',
    'nl',
    'li',
    'ins',
    'b',
    'i',
    'strong',
    'em',
    'strike',
    'code',
    'hr',
    'br',
    'div',
    'table',
    'thead',
    'tbody',
    'tr',
    'th',
    'td',
    'pre',
    'del',
    'sup',
    'sub',
    'dl',
    'dt',
    'dd',
    'kbd',
    'q',
    'samp',
    'samp',
    'var',
    'hr',
    'rt',
    'rp',
    'summary',
    'iframe',
    'img',
    'caption',
    'figure',
  ],
  allowedAttributes: {
    a: ['href', 'name', 'target'],
    img: ['src', 'longdesc'],
    div: ['itemscope', 'itemtype'],
  },
};

export const sanitizeConvertedHtml = (dirtyHtml: any) => {
  return sanitizeHtml(dirtyHtml, sanitizerOptions);
};

export const forceAnchorTagNewTab = (html: any) => {
  return html.replace(new RegExp('<a', 'g'), '<a target="_blank"');
};

export const useMarkdownConverter = () =>
  useCallback((markdown?: string) => {
    const converter = getMarkdownConverter();
    const html = converter.makeHtml(markdown);
    return sanitizeConvertedHtml(html);
  }, []);
