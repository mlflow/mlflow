import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { DesignSystemProvider, Table, TableCell, TableHeader, TableRow } from '@databricks/design-system';
import { describe, it, expect } from '@jest/globals';
import { flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
// Define an array of cells corresponding to that data model.
const getTableData = () => [
    {
        firstName: 'Lola',
        lastName: 'Sporer',
        age: 57,
        address: '584 Gerard Plaza',
    },
    {
        firstName: 'Antoinette',
        lastName: 'Watsica',
        age: 21,
        address: '39120 Howe Keys',
    },
    {
        firstName: 'Paul',
        lastName: 'Waters',
        age: 53,
        address: '444 Crist Haven',
    },
];
const columns = [
    {
        id: 'name',
        header: 'Name',
        accessorFn: (row) => `${row.firstName} ${row.lastName}`,
        minSize: 100,
        size: 200,
        maxSize: 400,
    },
    {
        id: 'age',
        header: 'Age',
        accessorKey: 'age',
        minSize: 100,
        size: 200,
        maxSize: 400,
        // Use `cell` to customize the rendering of the cell with JSX
        cell: (row) => _jsx("span", { "data-testid": "foo", children: row.getValue() }),
    },
    {
        id: 'address',
        header: 'Address',
        accessorKey: 'address',
        minSize: 100,
        size: 200,
        maxSize: 400,
    },
];
const Example = () => {
    const [data] = useState(() => getTableData());
    const table = useReactTable({
        data,
        columns,
        enableColumnResizing: true,
        columnResizeMode: 'onChange',
        getCoreRowModel: getCoreRowModel(),
    });
    return (_jsxs(Table, { style: {}, children: [table.getHeaderGroups().map((headerGroup) => (_jsx(TableRow, { isHeader: true, children: headerGroup.headers.map((header) => {
                    return (_jsxs(TableHeader, { componentId: "codegen_design-system_src_design-system_tableui_tableuiadjustwidth.test.tsx_99", style: { maxWidth: header.column.getSize() }, header: header, column: header.column, setColumnSizing: table.setColumnSizing, children: [flexRender(header.column.columnDef.header, header.getContext()), " (maxWidth: ", header.column.getSize(), ")"] }, header.id));
                }) }, headerGroup.id))), table.getRowModel().rows.map((row) => (_jsx(TableRow, { children: row.getAllCells().map((cell) => (_jsx(TableCell, { style: { maxWidth: cell.column.getSize() }, children: flexRender(cell.column.columnDef.cell, cell.getContext()) }, cell.id))) }, row.id)))] }));
};
describe('TableUIAdjustWidth', () => {
    it('Renders correctly and handles properly adjust width from clicking.', async () => {
        render(_jsx(DesignSystemProvider, { children: _jsx(Example, {}) }));
        // Validate the size of the columns
        const headers = screen.getAllByRole('columnheader');
        const resizeHandles = screen.getAllByRole('button');
        const nameColumnResizeHandle = resizeHandles[0];
        const nameHeader = headers[0];
        await userEvent.click(nameColumnResizeHandle);
        expect(resizeHandles[0]).toHaveAttribute('data-state', 'open');
        expect(nameHeader).toHaveStyle({ maxWidth: '200px' });
        // Get buttons from the popover dialog
        const popoverButtons = within(screen.getByRole('dialog')).getAllByRole('button');
        // Validate that the decrease button decreases the width of the first column.
        const decreaseButton = popoverButtons[0];
        expect(decreaseButton).toBeInTheDocument();
        await userEvent.click(decreaseButton);
        expect(nameHeader).toHaveStyle({ maxWidth: '190px' });
        // Validate that the increase button increases the width of the first column.
        const increaseButton = popoverButtons[1];
        expect(increaseButton).toBeInTheDocument();
        await userEvent.click(increaseButton);
        expect(nameHeader).toHaveStyle({ maxWidth: '200px' });
    });
});
//# sourceMappingURL=TableUIAdjustWidth.test.js.map