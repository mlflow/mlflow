import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, afterEach, jest, it, expect } from '@jest/globals';
import { fireEvent, render } from '@testing-library/react';
import { TableCell, TableHeader, TableRow, TableRowSelectCell, Table, TableFilterInput } from './index';
const Example = () => {
    return (_jsx("div", { children: _jsxs(Table, { children: [_jsxs(TableRow, { isHeader: true, children: [_jsx(TableHeader, { componentId: "codegen_design-system_src_design-system_tableui_tableui.test.tsx_10", children: "Name" }), _jsx(TableHeader, { componentId: "codegen_design-system_src_design-system_tableui_tableui.test.tsx_11", children: "Age" }), _jsx(TableHeader, { componentId: "codegen_design-system_src_design-system_tableui_tableui.test.tsx_12", children: "Address" })] }), _jsxs(TableRow, { children: [_jsx(TableCell, { children: "George" }), _jsx(TableCell, { children: "30" }), _jsx(TableCell, { children: "1313 Mockingbird Lane" })] }), _jsxs(TableRow, { children: [_jsx(TableCell, { children: "Joe" }), _jsx(TableCell, { children: "31" }), _jsx(TableCell, { children: "1313 Mockingbird Lane" })] })] }) }));
};
describe('TableUI', () => {
    afterEach(() => {
        jest.clearAllMocks();
    });
    it('Renders correctly', () => {
        const { queryAllByRole } = render(_jsx(Example, {}));
        // Validate that roles are set correctly.
        expect(queryAllByRole('table')).toHaveLength(1);
        expect(queryAllByRole('row')).toHaveLength(3);
        expect(queryAllByRole('columnheader')).toHaveLength(3);
        expect(queryAllByRole('cell')).toHaveLength(6);
    });
    describe('TableHeader', () => {
        it('throws an error if you try to use TableHeader without TableRow + `isHeader` prop', () => {
            const renderBadTable = () => {
                render(_jsx(Table, { children: _jsx(TableRow, { children: _jsx(TableHeader, { componentId: "codegen_design-system_src_design-system_tableui_tableui.test.tsx_51", children: "Name" }) }) }));
            };
            const renderCorrectTable = () => {
                render(_jsx(Table, { children: _jsx(TableRow, { isHeader: true, children: _jsx(TableHeader, { componentId: "codegen_design-system_src_design-system_tableui_tableui.test.tsx_61", children: "Name" }) }) }));
            };
            // Suppress console.error output.
            jest.spyOn(console, 'error').mockImplementation(() => { });
            expect(() => renderBadTable()).toThrowError();
            expect(() => renderCorrectTable()).not.toThrowError();
        });
    });
    describe('TableRowSelectCell', () => {
        it('throws an error if you try to use `TableRowSelectCell` without providing `someRowsSelected` to `Table`', () => {
            const renderBadTable = () => {
                render(_jsx(Table, { children: _jsx(TableRow, { children: _jsx(TableRowSelectCell, { componentId: "codegen_design-system_src_design-system_tableui_tableui.test.tsx_80", checked: true }) }) }));
            };
            const renderCorrectTable = () => {
                render(_jsx(Table, { someRowsSelected: true, children: _jsx(TableRow, { isHeader: true, children: _jsx(TableRowSelectCell, { componentId: "codegen_design-system_src_design-system_tableui_tableui.test.tsx_90", checked: true }) }) }));
            };
            // Suppress console.error output.
            jest.spyOn(console, 'error').mockImplementation(() => { });
            expect(() => renderBadTable()).toThrowError();
            expect(() => renderCorrectTable()).not.toThrowError();
        });
        it('throws an error if you try to set `TableRowSelectCell` to `indeterminate` outside of a header row', () => {
            const renderBadTable = () => {
                render(_jsx(Table, { someRowsSelected: true, children: _jsx(TableRow, { children: _jsx(TableRowSelectCell, { componentId: "codegen_design-system_src_design-system_tableui_tableui.test.tsx_107", indeterminate: true }) }) }));
            };
            const renderCorrectTable = () => {
                render(_jsx(Table, { someRowsSelected: true, children: _jsx(TableRow, { isHeader: true, children: _jsx(TableRowSelectCell, { componentId: "codegen_design-system_src_design-system_tableui_tableui.test.tsx_117", indeterminate: true }) }) }));
            };
            // Suppress console.error output.
            jest.spyOn(console, 'error').mockImplementation(() => { });
            expect(() => renderBadTable()).toThrowError();
            expect(() => renderCorrectTable()).not.toThrowError();
        });
        it('should call onChange when TableFilterInput is called', () => {
            const onChange = jest.fn();
            const { getByTestId } = render(_jsx(TableFilterInput, { componentId: "codegen_design-system_src_design-system_tableui_tableui.test.tsx_133", "data-testid": "filter-input", onChange: onChange }));
            const input = getByTestId('filter-input');
            fireEvent.change(input, { target: { value: 'test' } });
            expect(onChange).toHaveBeenCalledTimes(1);
        });
        it('should call onClear when clear icon is pressed in TableFilterInput', async () => {
            const onClear = jest.fn();
            const { getByRole } = render(_jsx(TableFilterInput, { componentId: "codegen_design-system_src_design-system_tableui_tableui.test.tsx_143", onClear: onClear, value: "val" }));
            const clearIcon = getByRole('button');
            fireEvent.click(clearIcon);
            expect(onClear).toHaveBeenCalledTimes(1);
        });
        it('should call onChange when onClear is not defined and clear icon is clicked', async () => {
            const onChange = jest.fn();
            const { getByRole } = render(_jsx(TableFilterInput, { componentId: "codegen_design-system_src_design-system_tableui_tableui.test.tsx_154", onChange: onChange, value: "val" }));
            const clearIcon = getByRole('button');
            fireEvent.click(clearIcon);
            expect(onChange).toHaveBeenCalledTimes(1);
        });
    });
});
//# sourceMappingURL=TableUI.test.js.map