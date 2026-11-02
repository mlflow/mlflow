import type { MutableRefObject } from 'react';
import type { DatasetWithRunType } from '../../components/experiment-page/components/runs/ExperimentViewDatasetDrawer';

// The eval-runs table's TanStack `meta` shape; cells read `table.options.meta` cast to this. Not
// augmenting the global `TableMeta` interface — it's shared app-wide, and fields there break other
// tables' excess-property checks.
export interface ExperimentEvaluationRunsTableMeta {
  setSelectedRunUuid: (runUuid: string) => void;
  setSelectedDatasetWithRun: (datasetWithRun: DatasetWithRunType) => void;
  setIsDrawerOpen: (isOpen: boolean) => void;
  flatRunIndexMap?: Record<string, number>;
  enableNewEvalRunsUi?: boolean;
  hasExpandableRows?: boolean;
  /** Last clicked checkbox row; the start of a shift-click range. */
  selectionAnchorRowIdRef?: MutableRefObject<string | undefined>;
  baselineRunUuid?: string;
}

// TanStack types `meta` as the empty `TableMeta`, so cells narrow it here instead of casting.
export const isEvalRunsTableMeta = (meta: unknown): meta is ExperimentEvaluationRunsTableMeta =>
  typeof meta === 'object' && meta !== null && 'setSelectedRunUuid' in meta;
