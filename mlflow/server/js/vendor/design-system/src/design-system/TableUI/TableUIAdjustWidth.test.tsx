import { DesignSystemProvider, Table, TableCell, TableHeader, TableRow } from '@databricks/design-system';

import type { ColumnDef } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React, { useState } from 'react';

// Define your data model
type Person = {
  firstName: string;
  lastName: string;
  age: number;
  address: string;
};

// Define an array of cells corresponding to that data model.
const getTableData = (): Person[] => [
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

const columns: ColumnDef<Person>[] = [
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
    cell: (row) => <span data-testid="foo">{row.getValue()}</span>,
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

const Example = (): JSX.Element => {
  const [data] = useState(() => getTableData());

  const table = useReactTable<Person>({
    data,
    columns,
    enableColumnResizing: true,
    columnResizeMode: 'onChange',
    getCoreRowModel: getCoreRowModel(),
  });

  const increaseHandler = (columnId: string, newSize: number) => () => {
    table.setColumnSizing((old) => ({
      ...old,
      [columnId]: newSize,
    }));
  };

  const decreaseHandler = (columnId: string, newSize: number) => () => {
    table.setColumnSizing((old) => ({
      ...old,
      [columnId]: newSize,
    }));
  };

  return (
    <Table style={{}}>
      {table.getHeaderGroups().map((headerGroup) => (
        <TableRow key={headerGroup.id} isHeader>
          {headerGroup.headers.map((header) => {
            return (
              <TableHeader
                componentId="codegen_design-system_src_design-system_tableui_tableuiadjustwidth.test.tsx_99"
                key={header.id}
                style={{ maxWidth: header.column.getSize() }}
                resizable={header.column.getCanResize()}
                resizeHandler={header.getResizeHandler()}
                hasAdjustableWidthHeader={true}
                increaseWidthHandler={increaseHandler(header.column.id, header.column.getSize() + 10)}
                decreaseWidthHandler={decreaseHandler(header.column.id, header.column.getSize() - 10)}
              >
                {flexRender(header.column.columnDef.header, header.getContext())} (maxWidth: {header.column.getSize()})
              </TableHeader>
            );
          })}
        </TableRow>
      ))}
      {table.getRowModel().rows.map((row) => (
        <TableRow key={row.id}>
          {row.getAllCells().map((cell) => (
            <TableCell key={cell.id} style={{ maxWidth: cell.column.getSize() }}>
              {flexRender(cell.column.columnDef.cell, cell.getContext())}
            </TableCell>
          ))}
        </TableRow>
      ))}
    </Table>
  );
};

describe('TableUIAdjustWidth', () => {
  it('Renders correctly and handles properly adjust width from clicking.', async () => {
    render(
      <DesignSystemProvider>
        <Example />
      </DesignSystemProvider>,
    );

    // Validate the size of the columns
    const headers = screen.getAllByRole('columnheader');
    const resizeHandles = screen.getAllByRole('separator');
    const nameColumnResizeHandle = resizeHandles[0];
    const nameHeader = headers[0];
    await userEvent.click(nameColumnResizeHandle);
    expect(resizeHandles[0]).toHaveAttribute('data-state', 'open');
    expect(nameHeader).toHaveStyle({ maxWidth: '200px' });

    const buttons = screen.getAllByRole('button');

    // Validate that the decrease button decreases the width of the first column.
    const decreaseButton = buttons[0];
    expect(decreaseButton).toBeInTheDocument();
    await userEvent.click(decreaseButton);
    expect(nameHeader).toHaveStyle({ maxWidth: '190px' });

    // Validate that the increase button increases the width of the first column.
    const increaseButton = buttons[1];
    expect(increaseButton).toBeInTheDocument();
    await userEvent.click(increaseButton);
    expect(nameHeader).toHaveStyle({ maxWidth: '200px' });
  });
});
