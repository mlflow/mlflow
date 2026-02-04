import { useCallback, useMemo, useState } from 'react';

import { useTableSortURL } from './useTableSortURL';
import { shoudlEnableURLPersistenceForSortAndColumns } from '../../model-trace-explorer/FeatureUtils';
import type { EvaluationsOverviewTableSort, TracesTableColumn } from '../types';

export const useTableSort = (
  selectedColumns: TracesTableColumn[],
  initialTableSort?: EvaluationsOverviewTableSort,
): [EvaluationsOverviewTableSort | undefined, (sort: EvaluationsOverviewTableSort | undefined) => void] => {
  const enableURLPersistence = shoudlEnableURLPersistenceForSortAndColumns();

  const [urlTableSort, setUrlTableSort] = useTableSortURL();

  const [localTableSort, setLocalTableSort] = useState<EvaluationsOverviewTableSort | undefined>(
    initialTableSort && selectedColumns.find((c) => c.id === initialTableSort.key) ? initialTableSort : undefined,
  );

  const derivedTableSort = useMemo(() => {
    let sourceSort: EvaluationsOverviewTableSort | undefined;

    if (enableURLPersistence) {
      // Priority: URL (if valid) → initial → undefined
      if (urlTableSort && selectedColumns.find((c) => c.id === urlTableSort.key)) {
        sourceSort = urlTableSort;
      } else {
        sourceSort = initialTableSort;
      }
    } else {
      // Old behavior: use local state
      sourceSort = localTableSort;
    }

    // Validate: sort column must be visible in selectedColumns
    if (!sourceSort || !selectedColumns.find((c) => c.id === sourceSort.key)) {
      return undefined;
    }

    return sourceSort;
  }, [enableURLPersistence, urlTableSort, initialTableSort, localTableSort, selectedColumns]);

  const setTableSort = useCallback(
    (sort: EvaluationsOverviewTableSort | undefined) => {
      if (enableURLPersistence) {
        setUrlTableSort(sort, false);
      } else {
        setLocalTableSort(sort);
      }
    },
    [enableURLPersistence, setUrlTableSort, setLocalTableSort],
  );

  return [derivedTableSort, setTableSort];
};
