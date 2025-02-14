const duBoisClassPrefix = 'du-bois-light';

/**
 * Used with the `selectEvent` utils for both Enzyme and RTL.
 */
export const selectClasses = {
  clear: `${duBoisClassPrefix}-select-clear`,
  item: `${duBoisClassPrefix}-select-selection-item`,
  list: 'rc-virtual-list',
  open: `${duBoisClassPrefix}-select-open`,
  option: `${duBoisClassPrefix}-select-item-option-content`,
  removeItem: `${duBoisClassPrefix}-select-selection-item-remove`,
  selector: `${duBoisClassPrefix}-select-selector`,
};

export interface GetTableRowByCellTextOptions {
  columnHeaderName?: string | RegExp;
}

export interface RowIdentifier {
  /** Text in a cell that uniquely identifies a row. */
  cellText: string;
  /** Name of the column to use when searching for `cellText`.  */
  columnHeaderName?: string | RegExp;
}

export interface TableRows<T> {
  bodyRows: T[];
  headerRow: T;
}

/**
 * @param columns List of column names
 * @param rows List of rows, where each row is a list of cell texts
 * @returns Markdown formatted string representing the data
 *
 * @example
 * // returns the string:
 * //   | Name | Fruit |
 * //   | --- | --- |
 * //   | Alice | Apple |
 * //   | Brady | Banana |
 * createMarkdownTable(['Name', 'Age'], [['Alice', 'Apple'], ['Brady', 'Banana']])
 */
export function createMarkdownTable(columns: string[], rows: string[][]) {
  const headerRow = `| ${columns.join(' | ')} |`;
  const separatorRow = `| ${columns.fill('---').join(' | ')} |`;
  const dataRows = `${rows.map((row) => `| ${row.join(' | ')} |`).join('\n')}`;
  const markdownTable = `${headerRow}\n${separatorRow}\n${dataRows}`;
  return markdownTable;
}
