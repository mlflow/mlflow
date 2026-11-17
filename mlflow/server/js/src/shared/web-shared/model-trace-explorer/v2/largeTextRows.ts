export interface TextRow {
  start: number;
  end: number;
}

// Keep a single long line bounded too; never split a UTF-16 surrogate pair across rows.
export const getLargeTextRows = (text: string): TextRow[] => {
  const rows: TextRow[] = [];
  let start = 0;
  for (;;) {
    const newline = text.indexOf('\n', start);
    const lineEnd = newline < 0 ? text.length : newline;
    let position = start;
    do {
      let end = Math.min(position + 2000, lineEnd);
      if (end < lineEnd && /[\uD800-\uDBFF]/.test(text[end - 1]) && /[\uDC00-\uDFFF]/.test(text[end])) {
        end--;
      }
      rows.push({ start: position, end });
      position = end;
    } while (position < lineEnd);
    if (newline < 0) break;
    start = newline + 1;
  }
  return rows;
};
