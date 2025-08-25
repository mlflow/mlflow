import { useMemo, useState } from 'react';

import type { EvaluationsOverviewTableSort, TracesTableColumn } from '../types';

export const useTableSort = (
  selectedColumns: TracesTableColumn[],
  initialTableSort?: EvaluationsOverviewTableSort,
): [EvaluationsOverviewTableSort | undefined, (sort: EvaluationsOverviewTableSort | undefined) => void] => {
  const [tableSort, setTableSort] = useState<EvaluationsOverviewTableSort | undefined>(
    initialTableSort && selectedColumns.find((c) => c.id === initialTableSort.key) ? initialTableSort : undefined,
  );

  // This is to keep table sort in sync with selected columns.
  // e.g. if the user deselects the column that is currently used for sorting,
  // we should clear the sort.
  const derivedTableSort = useMemo(() => {
    if (!tableSort) return undefined;

    if (!selectedColumns.find((c) => c.id === tableSort.key)) {
      return undefined;
    }

    return tableSort;
  }, [tableSort, selectedColumns]);

  return [derivedTableSort, setTableSort];
};
