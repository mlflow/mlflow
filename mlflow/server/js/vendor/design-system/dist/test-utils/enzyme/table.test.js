import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it } from '@jest/globals';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17';
// eslint-disable-next-line @databricks/no-restricted-imports-regexp
import { mount, configure as configureEnzyme } from 'enzyme';
import { expect } from '@databricks/config-jest/enzyme';
import { getTableCellInRow, getTableRowByCellText, getTableRows, toMarkdownTable } from './index';
import { findByText } from './utils';
import { DesignSystemProvider, Table, TableCell, TableHeader, TableRow } from '../../design-system';
configureEnzyme({ adapter: new Adapter() });
function mountTable() {
    const data = Array.from({ length: 5 }).map((_, i) => ({
        age: '0',
        id: i,
        name: `Name ${i}`,
    }));
    return mount(_jsx(DesignSystemProvider, { children: _jsxs(Table, { children: [_jsxs(TableRow, { isHeader: true, children: [_jsx(TableHeader, { componentId: "codegen_design-system_src_test-utils_enzyme_table.test.tsx_23", children: "Name" }), _jsx(TableHeader, { componentId: "codegen_design-system_src_test-utils_enzyme_table.test.tsx_24", children: "Age" })] }), data.map((row) => (_jsxs(TableRow, { children: [_jsx(TableCell, { children: row.name }), _jsx(TableCell, { children: row.age })] }, row.id)))] }) }));
}
describe('getTableRowByCellText', () => {
    it('should return the row that contains the matching cell without specified columnHeaderName', () => {
        const wrapper = mountTable();
        const row = getTableRowByCellText(wrapper, 'Name 2');
        expect(row).toHaveProp('role', 'row');
        expect(row).toIncludeText('Name 2');
    });
    it('should return the row that contains the matching cell with specified columnHeaderName', () => {
        const wrapper = mountTable();
        const row = getTableRowByCellText(wrapper, 'Name 2', { columnHeaderName: 'Name' });
        expect(row).toHaveProp('role', 'row');
        expect(row).toIncludeText('Name 2');
    });
    it('should throw an error when no rows match', () => {
        const wrapper = mountTable();
        expect(() => getTableRowByCellText(wrapper, 'Name 404', { columnHeaderName: 'Name' })).toThrowError();
    });
    it('should throw an error when more than one row matches', () => {
        const wrapper = mountTable();
        expect(() => getTableRowByCellText(wrapper, '0', { columnHeaderName: 'Age' })).toThrowError();
    });
    it('should throw an error when the column header does not exist', () => {
        const wrapper = mountTable();
        expect(() => getTableRowByCellText(wrapper, 'Name 1', { columnHeaderName: '404' })).toThrowError();
    });
});
describe('toMarkdownTable', () => {
    it('should return the table in markdown format', () => {
        const wrapper = mountTable();
        const markdownTable = toMarkdownTable(wrapper);
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
        const wrapper = mountTable();
        const result = getTableRows(wrapper);
        expect(result.headerRow).toBeDefined();
        expect(result.bodyRows).toHaveLength(5);
        result.bodyRows.forEach((row, index) => {
            expect(findByText(row, `Name ${index}`)).toExist();
        });
    });
});
describe('getTableCellInRow', () => {
    it('should return the cell for the appropriate column in the specified row', () => {
        const wrapper = mountTable();
        expect(getTableCellInRow(wrapper, { cellText: 'Name 0' }, 'Age').text()).toEqual('0');
    });
});
//# sourceMappingURL=table.test.js.map