import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table';
import { screen, render, act, fireEvent } from '@testing-library/react';
import { useState } from 'react';

import { Table, TableCell, TableRow, TableHeader } from './index';
import { DesignSystemEventProvider, DesignSystemEventProviderAnalyticsEventTypes } from '../DesignSystemEventProvider';

type Person = {
  name: string;
  age: number;
};

const columns: ColumnDef<Person>[] = [
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

const getTableData = (): Person[] => {
  const data: Person[] = [];

  data.push(
    {
      name: 'John Doe',
      age: 30,
    },
    {
      name: 'Jane Doe',
      age: 25,
    },
  );

  return data;
};

const Example = (): JSX.Element => {
  const [data] = useState(() => getTableData());
  const [sorting, setSorting] = useState<SortingState>([{ id: 'name', desc: false }]);

  const table = useReactTable<Person>({
    data,
    columns,
    state: {
      sorting,
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <Table>
      {table.getHeaderGroups().map((headerGroup) => (
        <TableRow isHeader key={headerGroup.id}>
          {headerGroup.headers.map((header) => {
            return (
              <TableHeader
                componentId={`table_test.header.${header.id}`}
                key={header.id}
                sortable
                sortDirection={header.column.getIsSorted() || 'none'}
                onToggleSort={header.column.getToggleSortingHandler()}
                analyticsEvents={[DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]}
              >
                {flexRender(header.column.columnDef.header, header.getContext())}
              </TableHeader>
            );
          })}
        </TableRow>
      ))}
      {table.getRowModel().rows.map((row) => (
        <TableRow key={row.id}>
          {row.getAllCells().map((cell) => {
            return <TableCell key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</TableCell>;
          })}
        </TableRow>
      ))}
    </Table>
  );
};

describe('TableHeader Analytics Events', () => {
  const eventCallback = jest.fn();

  it('emits onValueChange event when the sort direction changes', async () => {
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <Example />
      </DesignSystemEventProvider>,
    );

    expect(eventCallback).not.toHaveBeenCalled();

    const buttons = screen.getAllByRole('button');
    const sortName = buttons[0];
    const sortAge = buttons[1];

    expect(sortName).toBeInTheDocument();
    expect(sortAge).toBeInTheDocument();

    // Click on sortName. Header.name should have value desc.
    await act(() => {
      fireEvent.click(sortName);
    });

    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
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

    expect(eventCallback).toHaveBeenCalledTimes(3);
    expect(eventCallback).toHaveBeenNthCalledWith(2, {
      eventType: 'onValueChange',
      componentId: 'table_test.header.name',
      componentType: 'table_header',
      shouldStartInteraction: false,
      value: 'none',
    });

    expect(eventCallback).toHaveBeenNthCalledWith(3, {
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

    expect(eventCallback).toHaveBeenCalledTimes(5);
    expect(eventCallback).toHaveBeenNthCalledWith(4, {
      eventType: 'onValueChange',
      componentId: 'table_test.header.name',
      componentType: 'table_header',
      shouldStartInteraction: false,
      value: 'asc',
    });

    expect(eventCallback).toHaveBeenNthCalledWith(5, {
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

    expect(eventCallback).toHaveBeenCalledTimes(6);
    expect(eventCallback).toHaveBeenNthCalledWith(6, {
      eventType: 'onValueChange',
      componentId: 'table_test.header.name',
      componentType: 'table_header',
      shouldStartInteraction: false,
      value: 'desc',
    });

    await act(async () => {
      fireEvent.click(sortName);
    });

    expect(eventCallback).toHaveBeenCalledTimes(7);
    expect(eventCallback).toHaveBeenNthCalledWith(7, {
      eventType: 'onValueChange',
      componentId: 'table_test.header.name',
      componentType: 'table_header',
      shouldStartInteraction: false,
      value: 'none',
    });
  });
});
