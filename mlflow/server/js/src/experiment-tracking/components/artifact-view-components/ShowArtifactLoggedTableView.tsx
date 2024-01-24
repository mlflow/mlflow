import { Pagination, Table, TableCell, TableHeader, TableRow, useDesignSystemTheme } from '@databricks/design-system';
import { getArtifactContent, getArtifactLocationUrl } from '../../../common/utils/ArtifactUtils';
import { useEffect, useMemo, useState } from 'react';
import type { ColumnDef, SortingState, PaginationState } from '@tanstack/react-table';
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  getPaginationRowModel,
  useReactTable,
} from '@tanstack/react-table';
import React from 'react';
import { parseJSONSafe } from 'common/utils/TagUtils';

const DEFAULT_LOGGED_TABLE_PAGE_SIZE = 18;
const MIN_COLUMN_WIDTH = 100;

const LoggedTable = ({ data }: { data: { columns: string[]; data: string[][] } }) => {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [pagination, setPagination] = useState<PaginationState>({
    pageSize: DEFAULT_LOGGED_TABLE_PAGE_SIZE,
    pageIndex: 0,
  });
  const { theme } = useDesignSystemTheme();

  const columns = data['columns'];
  const rows = data['data'];

  const tableColumns = useMemo(
    () =>
      columns.map((col: string) => {
        return {
          id: col,
          header: col,
          accessorKey: col,
        };
      }),
    [columns],
  );
  const tableData = useMemo(
    () =>
      rows.map((row: string[]) => {
        const obj: Record<string, string> = {};
        for (let i = 0; i < columns.length; i++) {
          obj[columns[i]] = row[i];
        }
        return obj;
      }),
    [rows, columns],
  );
  const table = useReactTable({
    columns: tableColumns,
    data: tableData,
    state: {
      pagination,
      sorting,
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
  });

  const paginationComponent = (
    <Pagination
      currentPageIndex={pagination.pageIndex + 1}
      numTotal={rows.length}
      onChange={(page, pageSize) => {
        setPagination({
          pageSize: pageSize || pagination.pageSize,
          pageIndex: page - 1,
        });
      }}
      pageSize={pagination.pageSize}
    />
  );

  return (
    <div style={{ padding: `0px ${theme.spacing.md}px` }}>
      <Table pagination={paginationComponent} scrollable>
        {table.getHeaderGroups().map((headerGroup) => {
          return (
            <TableRow isHeader key={headerGroup.id}>
              {headerGroup.headers.map((header) => {
                return (
                  <TableHeader
                    key={header.id}
                    sortable
                    sortDirection={header.column.getIsSorted() || 'none'}
                    onToggleSort={header.column.getToggleSortingHandler()}
                    css={{ minWidth: MIN_COLUMN_WIDTH }}
                  >
                    {flexRender(header.column.columnDef.header, header.getContext())}
                  </TableHeader>
                );
              })}
            </TableRow>
          );
        })}
        {table.getRowModel().rows.map((row) => (
          <TableRow key={row.id}>
            {row.getAllCells().map((cell) => {
              return (
                <TableCell css={{ minWidth: MIN_COLUMN_WIDTH }} key={cell.id}>
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </TableCell>
              );
            })}
          </TableRow>
        ))}
      </Table>
    </div>
  );
};

type ShowArtifactLoggedTableViewProps = {
  runUuid: string;
  path: string;
};

export const ShowArtifactLoggedTableView = React.memo(({ runUuid, path }: ShowArtifactLoggedTableViewProps) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error>();
  const [curPath, setCurPath] = useState<string | undefined>(undefined);
  const [text, setText] = useState<string>('');

  useEffect(() => {
    setLoading(true);
    const artifactLocation = getArtifactLocationUrl(path, runUuid);
    getArtifactContent(artifactLocation)
      .then((value) => {
        setLoading(false);
        // Check if value is stringified JSON
        if (value && typeof value === 'string') {
          setText(value);
          setError(undefined);
        } else {
          setError(Error('Artifact is not a JSON file'));
        }
      })
      .catch((error: Error) => {
        setError(error);
        setLoading(false);
      });
    setCurPath(path);
  }, [path, runUuid]);

  const data = useMemo(() => parseJSONSafe(text), [text]);

  if (loading || path !== curPath) {
    return <div className="artifact-text-view-loading">Loading...</div>;
  }
  if (error) {
    return <div className="artifact-text-view-error">Oops we couldn't load your file because of an error.</div>;
  } else if (text) {
    if (!data) {
      return <div className="artifact-text-view-error">Unable to parse JSON.</div>;
    }
    return <LoggedTable data={data} />;
  }
  return <div className="artifact-text-view-error">Oops we couldn't load your file because of an error.</div>;
});
