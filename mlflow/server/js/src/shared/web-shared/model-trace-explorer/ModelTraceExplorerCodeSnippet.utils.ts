/** Same line count check as split, without allocating every line of a large payload. */
export const hasMoreLinesThan = (text: string, maxLines: number): boolean => {
  let searchFrom = 0;
  for (let newlines = 0; newlines < maxLines; newlines++) {
    const newline = text.indexOf('\n', searchFrom);
    if (newline === -1) {
      return false;
    }
    searchFrom = newline + 1;
  }
  return true;
};

/** Reads only the displayed prefix instead of splitting the complete text. */
export const getTextPreview = (text: string, maxLines: number, maxChars: number): string => {
  const preview = text.slice(0, maxChars);
  let searchFrom = 0;
  for (let line = 1; line <= maxLines; line++) {
    const newline = preview.indexOf('\n', searchFrom);
    if (newline === -1) {
      return preview;
    }
    if (line === maxLines) {
      return preview.slice(0, newline);
    }
    searchFrom = newline + 1;
  }
  return '';
};
