import sanitizeHtml from 'sanitize-html';
import { Converter } from 'showdown';

// Use Github-like Markdown (i.e. support for tasklists, strikethrough,
// simple line breaks, code blocks, emojis)
const DEFAULT_MARKDOWN_FLAVOR = 'github';

export const getConverter = () => {
  const converter = new Converter();
  converter.setFlavor(DEFAULT_MARKDOWN_FLAVOR);
  return converter;
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

export const sanitizeConvertedHtml = (dirtyHtml) => {
  return sanitizeHtml(dirtyHtml, sanitizerOptions);
};
