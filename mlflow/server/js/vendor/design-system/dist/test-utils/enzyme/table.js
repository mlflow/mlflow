import { findAllByRole, findByText } from './utils';
import { createMarkdownTable } from '../common';
function getColumnHeaderIndex(tableWrapper, columnHeaderName) {
    const columnHeaders = findAllByRole(tableWrapper, 'columnheader');
    const columnHeaderIndex = columnHeaders.findIndex((n) => {
        try {
            findByText(n, columnHeaderName);
            return true;
        }
        catch {
            return false;
        }
    });
    if (columnHeaderIndex === -1) {
        throw new Error(`Unable to find a column with name "${columnHeaderName}"\n${tableWrapper.debug()}`);
    }
    return columnHeaderIndex;
}
/**
 * Returns the table row that contains the specified `cellText`. The `cellText`
 * must be in the column with name `columnHeaderName` if it is specified. Otherwise,
 * the `cellText` must be in the first column. Throws an error if either multiple
 * rows or no rows can be found that match the given options. Also throws an error
 * if the column with name `columnHeaderName` cannot be found.
 *
 * @param tableWrapper The ReactWrapper containing the table to query in.
 * @param cellText The cell text that uniquely identifies the row.
 * @param columnHeaderName The name of the column to search the text for. If not provided,
 * the first column will be used.
 */
export function getTableRowByCellText(tableWrapper, cellText, { columnHeaderName } = {}) {
    const columnHeaderIndex = columnHeaderName === undefined ? 0 : getColumnHeaderIndex(tableWrapper, columnHeaderName);
    const matchingRows = findAllByRole(tableWrapper, 'row')
        // Skip first row (table header)
        .slice(1)
        .filter((row) => {
        const cells = findAllByRole(row, 'cell');
        const cell = cells[columnHeaderIndex];
        try {
            findByText(cell, cellText);
            return true;
        }
        catch {
            return false;
        }
    });
    if (matchingRows.length === 0) {
        throw new Error(`Unable to find a table row with text "${cellText}" in the column "${columnHeaderName}"\n${tableWrapper.debug()}`);
    }
    if (matchingRows.length > 1) {
        throw new Error(`Found multiple table rows with text "${cellText}" in the column "${columnHeaderName}"\n${tableWrapper.debug()}`);
    }
    return matchingRows[0].hostNodes();
}
/**
 * Converts a Du Bois table to a markdown table string. This means that each cell
 * is separated by a pipe (including the edges), the header row is on its own line
 * at the top, each data row is on its own line below, and the header row is separated
 * by a row of dashes from the data rows. This is useful for checking table contents
 * in tests.
 *
 * @param tableWrapper The ReactWrapper containing the table to query in.
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
export function toMarkdownTable(tableWrapper) {
    const { bodyRows, headerRow } = getTableRows(tableWrapper);
    const columns = findAllByRole(headerRow, 'columnheader').map((column) => column.text());
    const rows = bodyRows.map((row) => findAllByRole(row, 'cell').map((cell) => cell.text()));
    return createMarkdownTable(columns, rows);
}
/**
 * Returns the header row and all body rows (non-header rows) in order. Assumes that the
 * `tableWrapper` has a single header row (as the first row) and the rest of the rows are
 * body rows.
 *
 * @param tableWrapper The ReactWrapper containing the table to query in.
 */
export function getTableRows(tableWrapper) {
    const [firstRow, ...restRows] = findAllByRole(tableWrapper, 'row');
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
 * const result = getTableCellInRow(wrapper, { cellText: 'Carlos' }, 'Age');
 * expect(result.textContent).toEqual('39');
 * ```
 */
export function getTableCellInRow(tableWrapper, row, columnHeaderName) {
    const tableRowWrapper = getTableRowByCellText(tableWrapper, row.cellText, { columnHeaderName: row.columnHeaderName });
    const columnHeaderIndex = getColumnHeaderIndex(tableWrapper, columnHeaderName);
    const cells = findAllByRole(tableRowWrapper, 'cell');
    const cell = cells[columnHeaderIndex];
    return cell;
}
//# sourceMappingURL=table.js.map