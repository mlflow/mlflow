import type { MessageDescriptor } from 'react-intl';
import { defineMessage, FormattedMessage } from 'react-intl';
import {
  CheckboxCell,
  CreatedAtCell,
  DatasetCell,
  DatasetVersionCell,
  ModelVersionCell,
  RunNameCell,
  SortableHeaderCell,
  StatusCell,
  TypeCell,
  VisiblityCell,
} from './ExperimentEvaluationRunsTableCellRenderers';
import type { ColumnDef } from '@tanstack/react-table';
import type { Theme, Interpolation } from '@emotion/react';
import type { RunEntityOrGroupData } from './ExperimentEvaluationRunsPage.utils';
import { ExperimentEvaluationRunsPageMode } from './hooks/useExperimentEvaluationRunsPageMode';
import { EvalRunsVisibilityHeaderCell } from './EvalRunsVisibilityHeaderCell';
import { SelectAllHeaderCell } from './EvalRunsSelectAllHeaderCell';

export enum ExperimentEvaluationRunsTableColumns {
  runName = 'run_name',
  dataset = 'dataset',
  modelVersion = 'model_version',
}

export type EvalRunsTableColumnDef = ColumnDef<RunEntityOrGroupData> & {
  meta?: {
    styles?: Interpolation<Theme>;
  };
};

// One step below the 16px dbui default, for a tighter V2 toolbar control row.
export const EVAL_RUNS_TOOLBAR_ICON_SIZE = 14;

export enum EvalRunsTableColumnId {
  checkbox = 'checkbox',
  visibility = 'visibility',
  runName = 'run_name',
  status = 'status',
  type = 'type',
  dataset = 'dataset',
  datasetVersion = 'dataset_version',
  modelVersion = 'model_version',
  createdAt = 'created_at',
}

export enum EvalRunsTableKeyedColumnPrefix {
  ATTRIBUTE = 'attribute',
  METRIC = 'metric',
  PARAM = 'param',
  TAG = 'tag',
}

export const EVAL_RUNS_UNSELECTABLE_COLUMNS: Set<string> = new Set([
  EvalRunsTableColumnId.checkbox,
  EvalRunsTableColumnId.visibility,
  EvalRunsTableColumnId.runName,
]);

// Icon-only columns that keep tight padding; the roomier cell padding would crowd their fixed width.
export const EVAL_RUNS_TIGHT_PADDING_COLUMN_IDS: Set<string> = new Set([
  EvalRunsTableColumnId.checkbox,
  EvalRunsTableColumnId.visibility,
]);

// Seed/min/max widths for resizable columns; `size` is the initial/reset width (a persisted override
// wins at render). Keyed columns fall back to `EVAL_RUNS_KEYED_COLUMN_SIZE`.
export interface EvalRunsColumnSizeSpec {
  size: number;
  minSize: number;
  maxSize: number;
}

export const EVAL_RUNS_KEYED_COLUMN_SIZE: EvalRunsColumnSizeSpec = { size: 140, minSize: 100, maxSize: 320 };

// Keyed by column id (a plain string, since keyed metric/param/tag ids aren't enum members); the
// callsite's `?? EVAL_RUNS_KEYED_COLUMN_SIZE` supplies the default for ids not listed here.
export const EVAL_RUNS_COLUMN_SIZES: Partial<Record<string, EvalRunsColumnSizeSpec>> = {
  [EvalRunsTableColumnId.runName]: { size: 260, minSize: 160, maxSize: 600 },
  // Icon only; the header truncates when narrow.
  [EvalRunsTableColumnId.status]: { size: 72, minSize: 48, maxSize: 200 },
  [EvalRunsTableColumnId.type]: { size: 100, minSize: 80, maxSize: 160 },
  [EvalRunsTableColumnId.createdAt]: { size: 140, minSize: 100, maxSize: 320 },
  [EvalRunsTableColumnId.dataset]: { size: 180, minSize: 100, maxSize: 480 },
  [EvalRunsTableColumnId.datasetVersion]: { size: 120, minSize: 80, maxSize: 240 },
  [EvalRunsTableColumnId.modelVersion]: { size: 120, minSize: 80, maxSize: 320 },
};

export const EVAL_RUNS_TABLE_BASE_SELECTION_STATE: { [key: string]: boolean } = {
  [EvalRunsTableColumnId.checkbox]: true,
  [EvalRunsTableColumnId.visibility]: true,
  [EvalRunsTableColumnId.runName]: true,
  [EvalRunsTableColumnId.status]: true,
  [EvalRunsTableColumnId.type]: true,
  [EvalRunsTableColumnId.createdAt]: true,
  [EvalRunsTableColumnId.dataset]: true,
  [EvalRunsTableColumnId.modelVersion]: false,
};

// New UI shows the status in Run Name, so the Status column (kept for sorting) and Dataset version default off.
export const getEvalRunsBaseSelectionState = (enableNewEvalRunsUi?: boolean): { [key: string]: boolean } => {
  if (!enableNewEvalRunsUi) {
    return EVAL_RUNS_TABLE_BASE_SELECTION_STATE;
  }
  return {
    ...EVAL_RUNS_TABLE_BASE_SELECTION_STATE,
    [EvalRunsTableColumnId.status]: false,
    [EvalRunsTableColumnId.datasetVersion]: false,
  };
};

export const EVAL_RUNS_COLUMN_LABELS: Record<EvalRunsTableColumnId, MessageDescriptor> = {
  [EvalRunsTableColumnId.checkbox]: defineMessage({
    defaultMessage: 'Select',
    description: 'Label for the checkbox column in the evaluation runs table',
  }),
  [EvalRunsTableColumnId.visibility]: defineMessage({
    defaultMessage: 'Visibility',
    description: 'Label for the visibility icon column in the evaluation runs table',
  }),
  [EvalRunsTableColumnId.runName]: defineMessage({
    defaultMessage: 'Run Name',
    description: 'Column header for run name in the evaluation runs table',
  }),
  [EvalRunsTableColumnId.status]: defineMessage({
    defaultMessage: 'Status',
    description: 'Column header for run status in the evaluation runs table',
  }),
  [EvalRunsTableColumnId.type]: defineMessage({
    defaultMessage: 'Type',
    description: 'Column header for run type (Eval vs Test) in the evaluation runs table',
  }),
  [EvalRunsTableColumnId.createdAt]: defineMessage({
    defaultMessage: 'Created at',
    description: 'Column header for created timestamp in the evaluation runs table',
  }),
  [EvalRunsTableColumnId.dataset]: defineMessage({
    defaultMessage: 'Dataset',
    description: 'Label for the dataset column in the evaluation runs table',
  }),
  [EvalRunsTableColumnId.datasetVersion]: defineMessage({
    defaultMessage: 'Dataset version',
    description: 'Column header for the dataset version (digest) in the evaluation runs table',
  }),
  [EvalRunsTableColumnId.modelVersion]: defineMessage({
    defaultMessage: 'Version',
    description: 'Column header for model versions in the evaluation runs table',
  }),
};

export const EVAL_RUNS_COLUMN_TYPE_LABELS: Record<EvalRunsTableKeyedColumnPrefix, MessageDescriptor> = {
  [EvalRunsTableKeyedColumnPrefix.ATTRIBUTE]: defineMessage({
    defaultMessage: 'Attributes',
    description: 'Header for attribute columns in the evaluation runs table column configuration',
  }),
  [EvalRunsTableKeyedColumnPrefix.METRIC]: defineMessage({
    defaultMessage: 'Metrics',
    description: 'Header for metric columns in the evaluation runs table column configuration',
  }),
  [EvalRunsTableKeyedColumnPrefix.PARAM]: defineMessage({
    defaultMessage: 'Params',
    description: 'Header for parameter columns in the evaluation runs table column configuration',
  }),
  [EvalRunsTableKeyedColumnPrefix.TAG]: defineMessage({
    defaultMessage: 'Tags',
    description: 'Header for tag columns in the evaluation runs table column configuration',
  }),
};

export const getExperimentEvalRunsDefaultColumns = (
  viewMode: ExperimentEvaluationRunsPageMode,
  enableNewEvalRunsUi?: boolean,
): EvalRunsTableColumnDef[] => {
  const unselectableColumns: EvalRunsTableColumnDef[] = [
    {
      id: EvalRunsTableColumnId.checkbox,
      ...(enableNewEvalRunsUi ? { header: SelectAllHeaderCell } : {}),
      cell: CheckboxCell,
      enableResizing: false,
      enableSorting: false,
      size: 32,
      meta: { styles: { minWidth: 32, maxWidth: 32 } },
    },
  ];

  if (viewMode === ExperimentEvaluationRunsPageMode.CHARTS) {
    unselectableColumns.push({
      id: EvalRunsTableColumnId.visibility,
      header: EvalRunsVisibilityHeaderCell,
      cell: VisiblityCell,
      enableResizing: false,
      enableSorting: false,
      size: 40,
      meta: { styles: { minWidth: 40, maxWidth: 40, padding: '0 4px' } },
    });
  }

  const restColumns: EvalRunsTableColumnDef[] = [
    {
      id: EvalRunsTableColumnId.runName,
      header: () => <FormattedMessage {...EVAL_RUNS_COLUMN_LABELS[EvalRunsTableColumnId.runName]} />,
      cell: RunNameCell,
      enableResizing: true,
      ...(enableNewEvalRunsUi
        ? {
            enableSorting: true,
            sortingFn: 'alphanumeric' as const,
            accessorFn: (row: RunEntityOrGroupData) => ('subRuns' in row ? undefined : row.info.runName),
          }
        : {}),
      meta: {
        styles: {
          // The new UI adds the status icon to this cell, so it needs room for the name next to it.
          minWidth: enableNewEvalRunsUi ? 240 : 100,
        },
      },
    },
    {
      id: EvalRunsTableColumnId.status,
      header: () => <FormattedMessage {...EVAL_RUNS_COLUMN_LABELS[EvalRunsTableColumnId.status]} />,
      accessorFn: (row: RunEntityOrGroupData) => {
        if ('subRuns' in row) {
          return undefined;
        }
        return row.info.status;
      },
      cell: StatusCell,
      enableResizing: Boolean(enableNewEvalRunsUi),
      ...(enableNewEvalRunsUi ? { enableSorting: true, sortingFn: 'alphanumeric' as const } : {}),
      meta: {
        styles: {
          minWidth: 60,
          maxWidth: 80,
        },
      },
    } satisfies EvalRunsTableColumnDef,
    {
      id: EvalRunsTableColumnId.type,
      header: () => <FormattedMessage {...EVAL_RUNS_COLUMN_LABELS[EvalRunsTableColumnId.type]} />,
      cell: TypeCell,
      enableResizing: Boolean(enableNewEvalRunsUi),
      meta: {
        styles: {
          minWidth: 80,
          maxWidth: 110,
        },
      },
    },
    {
      id: EvalRunsTableColumnId.createdAt,
      header: (context) => (
        <SortableHeaderCell
          {...context}
          title={<FormattedMessage {...EVAL_RUNS_COLUMN_LABELS[EvalRunsTableColumnId.createdAt]} />}
        />
      ),
      accessorFn: (row) => {
        if ('subRuns' in row) {
          return undefined;
        }
        return row.info.startTime;
      },
      cell: CreatedAtCell,
      enableSorting: true,
      sortingFn: 'alphanumeric',
      enableResizing: true,
      meta: {
        styles: {
          minWidth: 100,
        },
      },
    },
    {
      id: EvalRunsTableColumnId.dataset,
      header: () => <FormattedMessage {...EVAL_RUNS_COLUMN_LABELS[EvalRunsTableColumnId.dataset]} />,
      cell: DatasetCell,
      enableResizing: true,
      ...(enableNewEvalRunsUi
        ? {
            enableSorting: true,
            sortingFn: 'alphanumeric' as const,
            accessorFn: (row: RunEntityOrGroupData) =>
              'subRuns' in row ? undefined : row.inputs?.datasetInputs?.[0]?.dataset?.name,
          }
        : {}),
      meta: {
        styles: {
          minWidth: 100,
        },
      },
    },
    ...(enableNewEvalRunsUi
      ? [
          {
            id: EvalRunsTableColumnId.datasetVersion,
            header: () => <FormattedMessage {...EVAL_RUNS_COLUMN_LABELS[EvalRunsTableColumnId.datasetVersion]} />,
            cell: DatasetVersionCell,
            enableResizing: true,
            enableSorting: true,
            sortingFn: 'alphanumeric',
            accessorFn: (row: RunEntityOrGroupData) =>
              'subRuns' in row ? undefined : row.inputs?.datasetInputs?.[0]?.dataset?.digest,
            meta: {
              styles: {
                minWidth: 80,
              },
            },
          } satisfies EvalRunsTableColumnDef,
        ]
      : []),
    {
      id: EvalRunsTableColumnId.modelVersion,
      header: () => <FormattedMessage {...EVAL_RUNS_COLUMN_LABELS[EvalRunsTableColumnId.modelVersion]} />,
      cell: ModelVersionCell,
      enableResizing: true,
      meta: {
        styles: {
          minWidth: 100,
        },
      },
    },
  ];

  return [...unselectableColumns, ...restColumns];
};
