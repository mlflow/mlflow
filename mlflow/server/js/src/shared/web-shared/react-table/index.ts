import { useReactTable as tanstackUseReactTable } from '@tanstack/react-table';
import type { TableOptions, RowData } from '@tanstack/table-core';

export function useReactTable_unverifiedWithReact18<TData extends RowData>(
  filePath: string,
  options: TableOptions<TData>,
): ReturnType<typeof tanstackUseReactTable<TData>> {
  return tanstackUseReactTable(options);
}

export const useReactTable_verifiedWithReact18 = tanstackUseReactTable;
