import { useMemo } from 'react';

import { useLocalStorage } from '@databricks/web-shared/hooks';

import type { SessionTableColumn } from '../types';

const LOCAL_STORAGE_KEY = 'experiment-chat-sessions-table-column-visibility';
const LOCAL_STORAGE_VERSION = 1;

export const useSessionsTableColumnVisibility = ({
  experimentId,
  columns,
}: {
  experimentId: string;
  columns: SessionTableColumn[];
}) => {
  const defaultColumnVisibility = useMemo(() => {
    return columns.reduce((acc, column) => {
      acc[column.id] = column.defaultVisibility;
      return acc;
    }, {} as Record<string, boolean>);
  }, [columns]);

  const [columnVisibility, setColumnVisibility] = useLocalStorage<Record<string, boolean>>({
    key: `${LOCAL_STORAGE_KEY}-${experimentId}`,
    version: LOCAL_STORAGE_VERSION,
    initialValue: defaultColumnVisibility,
  });

  return {
    columnVisibility,
    setColumnVisibility,
  };
};
