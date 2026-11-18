import { escapeRegExp } from 'lodash';

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

export const getLargeTextSearchRows = (text: string, rows: TextRow[], searchFilter: string) => {
  const pattern = searchFilter.trim() ? escapeRegExp(searchFilter.trim()) : undefined;
  const matches: ({ offset: number; index: number; lastIndex: number } | undefined)[] = [];
  if (pattern) {
    const regex = new RegExp(pattern, 'gi');
    let rowIndex = 0;
    let matchIndex = 0;
    for (let match = regex.exec(text); match; match = regex.exec(text)) {
      while (rowIndex < rows.length && rows[rowIndex].end <= match.index) rowIndex++;
      for (let index = rowIndex; index < rows.length && rows[index].start < regex.lastIndex; index++) {
        if (rows[index].end <= match.index) continue;
        const row = matches[index] ?? { offset: match.index, index: matchIndex, lastIndex: matchIndex };
        row.lastIndex = matchIndex;
        matches[index] = row;
      }
      matchIndex++;
    }
  }
  return { pattern, matches, queryLength: searchFilter.trim().length };
};
