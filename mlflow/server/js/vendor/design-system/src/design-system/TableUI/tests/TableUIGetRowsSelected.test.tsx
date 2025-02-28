import type { ColumnDef, ColumnFiltersState, PaginationState, SortingState } from '@tanstack/react-table';
import {
  filterFns,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table';
import { fireEvent, render, waitFor, within } from '@testing-library/react';
import { useEffect, useState } from 'react';

import {
  Button,
  CheckCircleIcon,
  Empty,
  Header,
  NewWindowIcon,
  Pagination,
  Spacer,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableRowSelectCell,
  useDesignSystemTheme,
  XCircleIcon,
} from '../..';

// Define your data model
type File = {
  status: 'Success' | 'Failure';
  name: string;
  category: string;
  size: string;
  createdBy: string;
};

// Define your columns
const columns: ColumnDef<File>[] = [
  {
    id: 'status',
    header: 'Status',
    accessorKey: 'status',
    filterFn: filterFns.equals,
  },
  {
    id: 'name',
    header: 'Name',
    accessorKey: 'name',
  },
  {
    id: 'category',
    header: 'Category',
    accessorKey: 'category',
    filterFn: filterFns.arrIncludesSome,
  },
  {
    id: 'size',
    header: 'Size',
    accessorKey: 'size',
  },
  {
    id: 'createdBy',
    header: 'Created By',
    accessorKey: 'createdBy',
    filterFn: filterFns.arrIncludesSome,
  },
];

const getTableData = (): (File & { key: string })[] => {
  const authors: string[] = [];
  const categories = ['Financial', 'Retail', 'Healthcare', 'Manufacturing', 'Other'];
  const data: (File & { key: string })[] = [];

  // Generate twenty fake authors with faker
  for (let i = 0; i < 20; i++) {
    authors.push(`firstname_${i}.lastname_${i}@databricks.com`);
  }

  // Generate some fake data
  for (let i = 0; i < 100; i++) {
    // Ensures that the faker data is determinstic by row.
    data.push({
      key: i.toString(),
      status: Math.random() > 0.5 ? 'Success' : 'Failure',
      name: `file_${i}.csv`,
      category: categories[Math.floor(Math.random() * categories.length)],
      size: `10 GB`,
      createdBy: authors[Math.floor(Math.random() * authors.length)],
    });
  }
  return data;
};

const Example = (): JSX.Element => {
  const [data] = useState(() => getTableData());

  // Set up pagination state, so that we can control the pagination from outside the table
  // in our `Pagination` component.
  const [pagination, setPagination] = useState<PaginationState>({
    pageSize: 10,
    pageIndex: 0,
  });

  const { theme } = useDesignSystemTheme();

  const [sorting, setSorting] = useState<SortingState>([{ id: 'status', desc: true }]);
  const [globalFilter, setGlobalFilter] = useState('');
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [rowSelection, setRowSelection] = useState({});

  useEffect(() => {
    // Goto page 1
    setPagination((prev) => ({
      ...prev,
      pageIndex: 0,
    }));
  }, [globalFilter, columnFilters]);

  const table = useReactTable<File>({
    data,
    columns,
    state: {
      pagination,
      sorting,
      columnFilters,
      globalFilter,
      rowSelection,
    },
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onColumnFiltersChange: setColumnFilters,
    onGlobalFilterChange: setGlobalFilter,
    onRowSelectionChange: setRowSelection,
    onSortingChange: setSorting,
    enableRowSelection: (row) => Number.parseInt(row.original.size, 10) > 30,
  });

  const paginationComponent = (
    <Pagination
      componentId="codegen_design-system_src_design-system_tableui_tests_tableuigetrowsselected.test.tsx_144"
      currentPageIndex={pagination.pageIndex + 1}
      numTotal={table.getFilteredRowModel().rows.length}
      onChange={(page, pageSize) => {
        setPagination({
          pageSize: pageSize || pagination.pageSize,
          pageIndex: page - 1,
        });
      }}
      pageSize={pagination.pageSize}
    />
  );

  const emptyComponent = <Empty description="No things found that match your search." />;
  const isEmpty = (): boolean => table.getRowModel().rows.length === 0;

  // Should be a Record type mapping keys of the `File` type above to CSS properties.
  const columnStyles: Partial<Record<keyof File, React.CSSProperties>> = {
    status: {
      maxWidth: 100,
    },
    size: {
      maxWidth: 100,
    },
  };

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Header title="Things" />
        <Button
          componentId="codegen_design-system_src_design-system_tableui_tests_tableuigetrowsselected.test.tsx_174"
          type="link"
        >
          Provide feedback <NewWindowIcon />
        </Button>
      </div>
      <Spacer />

      <Table
        pagination={paginationComponent}
        empty={isEmpty() ? emptyComponent : undefined}
        someRowsSelected={table.getIsSomePageRowsSelected() || table.getIsAllPageRowsSelected()}
      >
        {table.getHeaderGroups().map((headerGroup) => (
          <TableRow key={headerGroup.id} isHeader>
            <TableRowSelectCell
              componentId="codegen_design-system_src_design-system_tableui_tests_tableuigetrowsselected.test.tsx_190"
              checked={table.getIsAllPageRowsSelected()}
              indeterminate={table.getIsSomePageRowsSelected()}
              onChange={table.getToggleAllPageRowsSelectedHandler()}
            />
            {headerGroup.headers.map((header) => {
              return (
                <TableHeader
                  componentId="codegen_design-system_src_design-system_tableui_tests_tableuigetrowsselected.test.tsx_199"
                  style={
                    (header.column.columnDef.id &&
                      // @ts-expect-error-next-line
                      columnStyles[header.column.columnDef.id]) ||
                    {}
                  }
                  key={header.id}
                  ellipsis
                  sortable
                  sortDirection={header.column.getIsSorted() || 'none'}
                  onToggleSort={header.column.getToggleSortingHandler()}
                >
                  {flexRender(header.column.columnDef.header, header.getContext())}
                </TableHeader>
              );
            })}
          </TableRow>
        ))}
        {table.getRowModel().rows.map((row) => (
          <TableRow key={row.id} css={row.getCanSelect() ? undefined : { opacity: 0.5 }}>
            <TableRowSelectCell
              componentId="codegen_design-system_src_design-system_tableui_tests_tableuigetrowsselected.test.tsx_218"
              checked={row.getIsSelected()}
              onChange={row.getCanSelect() ? row.getToggleSelectedHandler() : undefined}
            />
            {row.getAllCells().map((cell) => {
              if (cell.column.columnDef.id === 'status') {
                return (
                  <TableCell
                    ellipsis
                    key={cell.id}
                    style={(cell.column.columnDef.id && columnStyles[cell.column.columnDef.id]) || {}}
                    align="center"
                  >
                    {cell.getValue() === 'Success' ? (
                      <CheckCircleIcon css={{ color: theme.colors.textValidationSuccess }} />
                    ) : (
                      <XCircleIcon css={{ color: theme.colors.textValidationDanger }} />
                    )}
                  </TableCell>
                );
              }

              return (
                <TableCell
                  ellipsis
                  key={cell.id}
                  style={
                    (cell.column.columnDef.id &&
                      // @ts-expect-error-next-line
                      columnStyles[cell.column.columnDef.id]) ||
                    {}
                  }
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}{' '}
                </TableCell>
              );
            })}
          </TableRow>
        ))}
      </Table>
    </div>
  );
};

describe('TableUI GetRowsSelected', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('Selects all correctly when some rows are disabled', async () => {
    const { queryAllByRole } = render(<Example />);

    expect(queryAllByRole('columnheader')).toHaveLength(6);

    const columnHeader = queryAllByRole('columnheader')[0];
    const checkbox = await within(columnHeader).findByRole('checkbox');

    expect(checkbox).toBeInTheDocument();

    fireEvent.click(checkbox);

    // eslint-disable-next-line testing-library/await-async-utils -- FEINF-3005
    waitFor(() => {
      expect(checkbox).toHaveAttribute('aria-checked', 'mixed');
    });

    fireEvent.click(checkbox);
    expect(checkbox).toHaveAttribute('aria-checked', 'false');
  });
});
