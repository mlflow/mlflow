import type { Table } from '@tanstack/react-table';
import React, { createContext, useMemo, useState } from 'react';

import type { EvalTraceComparisonEntry } from './types';

type TraceRow = EvalTraceComparisonEntry & { multiline?: boolean };

export interface GenAITracesTableContextValue<T> {
  /** TanStack table instance (may be undefined until grandchild mounts) */
  table: Table<T> | undefined;
  /** Grandchild calls this once when it builds the table */
  setTable: (tbl: Table<T> | undefined) => void;

  selectedRowIds: string[];
  /** Grandchild updates this on every selection change */
  setSelectedRowIds: (rowIds: string[]) => void;
}
export const GenAITracesTableContext = createContext<GenAITracesTableContextValue<TraceRow>>({
  table: undefined,
  setTable: () => {},
  selectedRowIds: [],
  setSelectedRowIds: () => {},
});

interface GenAITracesTableProviderProps {
  children: React.ReactNode;
}

export const GenAITracesTableProvider: React.FC<React.PropsWithChildren<GenAITracesTableProviderProps>> = ({
  children,
}) => {
  const [table, setTable] = useState<Table<TraceRow> | undefined>();
  const [selectedRowIds, setSelectedRowIds] = useState<string[]>([]);

  const value = useMemo(() => ({ table, setTable, selectedRowIds, setSelectedRowIds }), [table, selectedRowIds]);

  return <GenAITracesTableContext.Provider value={value}>{children}</GenAITracesTableContext.Provider>;
};
