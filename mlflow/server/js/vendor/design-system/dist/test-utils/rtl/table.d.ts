import type { GetTableRowByCellTextOptions, TableRows } from '../common';
/**
 * Returns the table row that contains the specified `cellText`. The `cellText`
 * must be in the column with name `columnHeaderName` if it is specified. Otherwise,
 * the `cellText` must be in the first column. Throws an error if either multiple
 * rows or no rows can be found that match the given options. Also throws an error
 * if the column with name `columnHeaderName` cannot be found.
 *
 * @param tableElement The HTMLElement representing the table to query in. This is likely
 * a `<div role="table">` element, so it can be queried by `screen.getByRole('table')`.
 * @param cellText The cell text that uniquely identifies the row.
 * @param columnHeaderName The name of the column to search the text for. If not provided,
 * the first column will be used.
 */
export declare function getTableRowByCellText(tableElement: HTMLElement, cellText: string, { columnHeaderName }?: GetTableRowByCellTextOptions): HTMLElement;
/**
 * Converts a Du Bois table to a markdown table string. This means that each cell
 * is separated by a pipe (including the edges), the header row is on its own line
 * at the top, each data row is on its own line below, and the header row is separated
 * by a row of dashes from the data rows. This is useful for checking table contents
 * in tests.
 *
 * @param tableElement The HTMLElement representing the table to query in. This is likely
 * a `<div role="table">` element, so it can be queried by `screen.getByRole('table')`.
 *
 * @example
 * The HTML table:
 * ```jsx
 *   <Table>
 *     <TableRow isHeader>
 *       <TableHeader>Name</TableHeader>
 *       <TableHeader>Fruit</TableHeader>
 *     </TableRow>
 *     <TableRow>
 *       <TableCell>Alice</TableCell>
 *       <TableCell>Apple</TableCell>
 *     </TableRow>
 *     <TableRow>
 *       <TableCell>Brady</TableCell>
 *       <TableCell>Banana</TableCell>
 *     </TableRow>
 *   </Table>
 * ```
 *
 * The Markdown table:
 * ```md
 *   | Name | Fruit |
 *   | --- | --- |
 *   | Alice | Apple |
 *   | Brady | Banana |
 * ```
 */
export declare function toMarkdownTable(tableElement: HTMLElement): string;
/**
 * Returns the header row and all body rows (non-header rows) in order. Assumes that the
 * `tableElement` has a single header row (as the first row) and the rest of the rows are
 * body rows.
 *
 * @param tableElement The HTMLElement representing the table to query in. This is likely
 * a `<div role="table">` element, so it can be queried by `screen.getByRole('table')`.
 */
export declare function getTableRows(tableElement: HTMLElement): TableRows<HTMLElement>;
//# sourceMappingURL=table.d.ts.map