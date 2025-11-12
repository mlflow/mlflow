import { queryHelpers, within } from '@testing-library/react';
import { createMarkdownTable } from '../common';
function getColumnHeaderIndex(tableElement, columnHeaderName) {
    const columnHeader = within(tableElement).getByRole('columnheader', { name: columnHeaderName });
    const columnHeaderIndex = Array.from(columnHeader.parentElement?.children ?? []).indexOf(columnHeader);
    return columnHeaderIndex;
}
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
export function getTableRowByCellText(tableElement, cellText, { columnHeaderName } = {}) {
    const columnHeaderIndex = columnHeaderName === undefined ? 0 : getColumnHeaderIndex(tableElement, columnHeaderName);
    const matchingRows = within(tableElement)
        .getAllByRole('row')
        // Skip first row (table header)
        .slice(1)
        .filter((row) => {
        const cells = within(row).getAllByRole('cell');
        const cell = cells[columnHeaderIndex];
        const cellContainsText = within(cell).queryByText(cellText) !== null;
        return cellContainsText;
    });
    if (matchingRows.length === 0) {
        throw queryHelpers.getElementError(`Unable to find a table row with text "${cellText}" in the column "${columnHeaderName}"`, tableElement);
    }
    if (matchingRows.length > 1) {
        throw queryHelpers.getElementError(`Found multiple table rows with text "${cellText}" in the column "${columnHeaderName}"`, tableElement);
    }
    return matchingRows[0];
}
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
export function toMarkdownTable(tableElement) {
    const { bodyRows, headerRow } = getTableRows(tableElement);
    const columns = within(headerRow)
        .getAllByRole('columnheader')
        .map((column) => column.textContent ?? '');
    const rows = bodyRows.map((row) => within(row)
        .getAllByRole('cell')
        .map((cell) => cell.textContent ?? ''));
    return createMarkdownTable(columns, rows);
}
/**
 * Returns the header row and all body rows (non-header rows) in order. Assumes that the
 * `tableElement` has a single header row (as the first row) and the rest of the rows are
 * body rows.
 *
 * @param tableElement The HTMLElement representing the table to query in. This is likely
 * a `<div role="table">` element, so it can be queried by `screen.getByRole('table')`.
 */
export function getTableRows(tableElement) {
    const [firstRow, ...restRows] = within(tableElement).getAllByRole('row');
    return {
        bodyRows: restRows,
        headerRow: firstRow,
    };
}
/**
 * Returns the table cell in the specified table row corresponding to the given
 * `columnHeaderName`. This is useful for checking that a row has a particular value
 * for a given column, especially when there are duplicate values in the column.
 *
 * @example
 * The HTML table:
 * ```jsx
 *   <Table>
 *     <TableRow isHeader>
 *       <TableHeader>Name</TableHeader>
 *       <TableHeader>Age</TableHeader>
 *     </TableRow>
 *     <TableRow>
 *       <TableCell>Alex</TableCell>
 *       <TableCell>25</TableCell>
 *     </TableRow>
 *     <TableRow>
 *       <TableCell>Brenda</TableCell>
 *       <TableCell>39</TableCell>
 *     </TableRow>
 *     <TableRow>
 *       <TableCell>Carlos</TableCell>
 *       <TableCell>39</TableCell>
 *     </TableRow>
 *   </Table>
 * ```
 *
 * ```js
 * const table = screen.getByRole('table');
 * const result = getTableCellInRow(table, { cellText: 'Carlos' }, 'Age');
 * expect(result.textContent).toEqual('39');
 * ```
 */
export function getTableCellInRow(tableElement, row, columnHeaderName) {
    const tableRowElement = getTableRowByCellText(tableElement, row.cellText, { columnHeaderName: row.columnHeaderName });
    const columnHeaderIndex = getColumnHeaderIndex(tableElement, columnHeaderName);
    const cells = within(tableRowElement).getAllByRole('cell');
    const cell = cells[columnHeaderIndex];
    return cell;
}
//# sourceMappingURL=table.js.map