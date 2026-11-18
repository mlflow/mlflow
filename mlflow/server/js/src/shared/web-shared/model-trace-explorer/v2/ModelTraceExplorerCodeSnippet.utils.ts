/** Stop counting once the collapsed preview's line budget is exceeded. */
export const hasMoreLinesThan = (text: string, maxLines: number): boolean => {
  let start = 0;
  for (let line = 0; line < maxLines; line++) {
    const newline = text.indexOf('\n', start);
    if (newline === -1) return false;
    start = newline + 1;
  }
  return true;
};

/** Match split/slice/join without allocating one string for every payload line. */
export const getTextPreview = (text: string, maxLines: number, maxChars: number): string => {
  const preview = text.slice(0, maxChars);
  let start = 0;
  for (let line = 1; line <= maxLines; line++) {
    const newline = preview.indexOf('\n', start);
    if (newline === -1) return preview;
    if (line === maxLines) return preview.slice(0, newline);
    start = newline + 1;
  }
  return '';
};
