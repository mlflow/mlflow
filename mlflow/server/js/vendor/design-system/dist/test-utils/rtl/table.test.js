import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { expect, describe, it } from '@jest/globals';
import { render, screen, within } from '@testing-library/react';
import { getTableCellInRow, getTableRowByCellText, getTableRows, toMarkdownTable } from './index';
import { DesignSystemProvider, Table, TableCell, TableHeader, TableRow } from '../../design-system';
function renderTable() {
    const data = Array.from({ length: 5 }).map((_, i) => ({
        age: '0',
        id: i,
        name: `Name ${i}`,
    }));
    return render(_jsx(DesignSystemProvider, { children: _jsxs(Table, { children: [_jsxs(TableRow, { isHeader: true, children: [_jsx(TableHeader, { componentId: "codegen_design-system_src_test-utils_rtl_table.test.tsx_18", children: "Name" }), _jsx(TableHeader, { componentId: "codegen_design-system_src_test-utils_rtl_table.test.tsx_19", children: "Age" })] }), data.map((row) => (_jsxs(TableRow, { children: [_jsx(TableCell, { children: row.name }), _jsx(TableCell, { children: row.age })] }, row.id)))] }) }));
}
describe('getTableRowByCellText', () => {
    it('should return the row that contains the matching cell without specified columnHeaderName', () => {
        renderTable();
        const row = getTableRowByCellText(screen.getByRole('table'), 'Name 2');
        expect(row).toHaveAttribute('role', 'row');
        expect(row).toContainElement(screen.getByText('Name 2'));
    });
    it('should return the row that contains the matching cell with specified columnHeaderName', () => {
        renderTable();
        const row = getTableRowByCellText(screen.getByRole('table'), 'Name 2', { columnHeaderName: 'Name' });
        expect(row).toHaveAttribute('role', 'row');
        expect(row).toContainElement(screen.getByText('Name 2'));
    });
    it('should throw an error when no rows match', () => {
        renderTable();
        expect(() => getTableRowByCellText(screen.getByRole('table'), 'Name 404', { columnHeaderName: 'Name' })).toThrowError();
    });
    it('should throw an error when more than one row matches', () => {
        renderTable();
        expect(() => getTableRowByCellText(screen.getByRole('table'), '0', { columnHeaderName: 'Age' })).toThrowError();
    });
    it('should throw an error when the column header does not exist', () => {
        renderTable();
        expect(() => getTableRowByCellText(screen.getByRole('table'), 'Name 1', { columnHeaderName: '404' })).toThrowError();
    });
});
describe('toMarkdownTable', () => {
    it('should return the table in markdown format', () => {
        renderTable();
        const markdownTable = toMarkdownTable(screen.getByRole('table'));
        expect(markdownTable).toEqual(`
| Name | Age |
| --- | --- |
| Name 0 | 0 |
| Name 1 | 0 |
| Name 2 | 0 |
| Name 3 | 0 |
| Name 4 | 0 |
    `.trim());
    });
});
describe('getTableRows', () => {
    it('should return the header row and all body rows in order', () => {
        renderTable();
        const result = getTableRows(screen.getByRole('table'));
        expect(result.headerRow).toBeDefined();
        expect(result.bodyRows).toHaveLength(5);
        result.bodyRows.forEach((row, index) => {
            expect(within(row).getByText(`Name ${index}`)).toBeInTheDocument();
        });
    });
});
describe('getTableCellInRow', () => {
    it('should return the cell for the appropriate column in the specified row', () => {
        renderTable();
        const table = screen.getByRole('table');
        expect(getTableCellInRow(table, { cellText: 'Name 0' }, 'Age').textContent).toEqual('0');
    });
});
//# sourceMappingURL=table.test.js.map