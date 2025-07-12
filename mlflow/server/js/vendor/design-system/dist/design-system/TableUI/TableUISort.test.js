import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { flexRender, getCoreRowModel, getSortedRowModel, useReactTable, } from '@tanstack/react-table';
import { screen, render, act, fireEvent } from '@testing-library/react';
import { useState } from 'react';
import { Table, TableCell, TableRow, TableHeader } from './index';
import { DesignSystemEventProviderAnalyticsEventTypes, setupDesignSystemEventProviderForTesting, } from '../DesignSystemEventProvider';
import { setupSafexTesting } from '../utils/safex';
const columns = [
    {
        id: 'name',
        header: 'Name',
        accessorKey: 'name',
    },
    {
        id: 'age',
        header: 'Age',
        accessorKey: 'age',
    },
];
const getTableData = () => {
    const data = [];
    data.push({
        name: 'John Doe',
        age: 30,
    }, {
        name: 'Jane Doe',
        age: 25,
    });
    return data;
};
const Example = () => {
    const [data] = useState(() => getTableData());
    const [sorting, setSorting] = useState([{ id: 'name', desc: false }]);
    const table = useReactTable({
        data,
        columns,
        state: {
            sorting,
        },
        onSortingChange: setSorting,
        getCoreRowModel: getCoreRowModel(),
        getSortedRowModel: getSortedRowModel(),
    });
    return (_jsxs(Table, { children: [table.getHeaderGroups().map((headerGroup) => (_jsx(TableRow, { isHeader: true, children: headerGroup.headers.map((header) => {
                    return (_jsx(TableHeader, { componentId: `table_test.header.${header.id}`, sortable: true, sortDirection: header.column.getIsSorted() || 'none', onToggleSort: header.column.getToggleSortingHandler(), analyticsEvents: [
                            DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                            DesignSystemEventProviderAnalyticsEventTypes.OnView,
                        ], children: flexRender(header.column.columnDef.header, header.getContext()) }, header.id));
                }) }, headerGroup.id))), table.getRowModel().rows.map((row) => (_jsx(TableRow, { children: row.getAllCells().map((cell) => {
                    return _jsx(TableCell, { children: flexRender(cell.column.columnDef.cell, cell.getContext()) }, cell.id);
                }) }, row.id)))] }));
};
describe('TableHeader Analytics Events', () => {
    const { setSafex } = setupSafexTesting();
    beforeEach(() => {
        setSafex({
            'databricks.fe.observability.defaultComponentView.tableHeader': true,
        });
        // Disable IntersectionObserver for useNotifyOnFirstView hook to triggeryar
        window.IntersectionObserver = undefined;
    });
    const eventCallback = jest.fn();
    const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
    it('emits onValueChange event when the sort direction changes', async () => {
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Example, {}) }));
        act(() => {
            expect(screen.getByText('Name')).toBeVisible();
            expect(screen.getByText('Age')).toBeVisible();
        });
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenNthCalledWith(1, {
            eventType: 'onView',
            componentId: 'table_test.header.name',
            componentType: 'table_header',
            shouldStartInteraction: false,
            value: 'asc',
        });
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onView',
            componentId: 'table_test.header.age',
            componentType: 'table_header',
            shouldStartInteraction: false,
            value: 'none',
        });
        const buttons = screen.getAllByRole('button');
        const sortName = buttons[0];
        const sortAge = buttons[1];
        expect(sortName).toBeInTheDocument();
        expect(sortAge).toBeInTheDocument();
        // Click on sortName. Header.name should have value desc.
        await act(() => {
            fireEvent.click(sortName);
        });
        expect(eventCallback).toHaveBeenCalledTimes(3);
        expect(eventCallback).toHaveBeenNthCalledWith(3, {
            eventType: 'onValueChange',
            componentId: 'table_test.header.name',
            componentType: 'table_header',
            shouldStartInteraction: false,
            value: 'desc',
        });
        // Click on sortAge. Header.age should have value desc and Header.name should have value none.
        await act(async () => {
            fireEvent.click(sortAge);
        });
        expect(eventCallback).toHaveBeenCalledTimes(5);
        expect(eventCallback).toHaveBeenNthCalledWith(4, {
            eventType: 'onValueChange',
            componentId: 'table_test.header.name',
            componentType: 'table_header',
            shouldStartInteraction: false,
            value: 'none',
        });
        expect(eventCallback).toHaveBeenNthCalledWith(5, {
            eventType: 'onValueChange',
            componentId: 'table_test.header.age',
            componentType: 'table_header',
            shouldStartInteraction: false,
            value: 'desc',
        });
        /// Click on sortName. Header.age should become none again and Header.name should have value asc.
        await act(async () => {
            fireEvent.click(sortName);
        });
        expect(eventCallback).toHaveBeenCalledTimes(7);
        expect(eventCallback).toHaveBeenNthCalledWith(6, {
            eventType: 'onValueChange',
            componentId: 'table_test.header.name',
            componentType: 'table_header',
            shouldStartInteraction: false,
            value: 'asc',
        });
        expect(eventCallback).toHaveBeenNthCalledWith(7, {
            eventType: 'onValueChange',
            componentId: 'table_test.header.age',
            componentType: 'table_header',
            shouldStartInteraction: false,
            value: 'none',
        });
        // Click on sortName twice and Header.name should have value desc and then none.
        await act(async () => {
            fireEvent.click(sortName);
        });
        expect(eventCallback).toHaveBeenCalledTimes(8);
        expect(eventCallback).toHaveBeenNthCalledWith(8, {
            eventType: 'onValueChange',
            componentId: 'table_test.header.name',
            componentType: 'table_header',
            shouldStartInteraction: false,
            value: 'desc',
        });
        await act(async () => {
            fireEvent.click(sortName);
        });
        expect(eventCallback).toHaveBeenCalledTimes(9);
        expect(eventCallback).toHaveBeenNthCalledWith(9, {
            eventType: 'onValueChange',
            componentId: 'table_test.header.name',
            componentType: 'table_header',
            shouldStartInteraction: false,
            value: 'none',
        });
    });
});
//# sourceMappingURL=TableUISort.test.js.map