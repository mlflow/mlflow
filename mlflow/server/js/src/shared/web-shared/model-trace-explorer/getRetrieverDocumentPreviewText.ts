const MAX_PREVIEW_CHARS = 1000;

// CSS ellipsis still lays out the complete text node, so bound the collapsed prefix.
export const getRetrieverDocumentPreviewText = (text: string): string =>
  text.length > MAX_PREVIEW_CHARS ? `${text.slice(0, MAX_PREVIEW_CHARS - 1).replace(/[\uD800-\uDBFF]$/, '')}…` : text;
