import type { MessageDescriptor } from 'react-intl';
import { defineMessage, FormattedMessage } from 'react-intl';
import {
  CheckboxCell,
  CreatedAtCell,
  DatasetCell,
  ModelVersionCell,
  RunNameCell,
  SortableHeaderCell,
  VisiblityCell,
} from './ExperimentEvaluationRunsTableCellRenderers';
import type { ColumnDef } from '@tanstack/react-table';
import type { Theme, Interpolation } from '@emotion/react';
import type { RunEntityOrGroupData } from './ExperimentEvaluationRunsPage.utils';
import { ExperimentEvaluationRunsPageMode } from './hooks/useExperimentEvaluationRunsPageMode';

export interface ExperimentEvaluationRunsTableMeta {
  setSelectedRunUuid: (runUuid: string) => void;
}

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

export enum EvalRunsTableColumnId {
  checkbox = 'checkbox',
  visibility = 'visibility',
  runName = 'run_name',
  dataset = 'dataset',
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

export const EVAL_RUNS_TABLE_BASE_SELECTION_STATE: { [key: string]: boolean } = {
  [EvalRunsTableColumnId.checkbox]: true,
  [EvalRunsTableColumnId.visibility]: true,
  [EvalRunsTableColumnId.runName]: true,
  [EvalRunsTableColumnId.createdAt]: true,
  [EvalRunsTableColumnId.dataset]: true,
  [EvalRunsTableColumnId.modelVersion]: true,
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
  [EvalRunsTableColumnId.createdAt]: defineMessage({
    defaultMessage: 'Created at',
    description: 'Column header for created timestamp in the evaluation runs table',
  }),
  [EvalRunsTableColumnId.dataset]: defineMessage({
    defaultMessage: 'Dataset',
    description: 'Label for the dataset column in the evaluation runs table',
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
): EvalRunsTableColumnDef[] => {
  const unselectableColumns: EvalRunsTableColumnDef[] = [
    {
      id: EvalRunsTableColumnId.checkbox,
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
      cell: VisiblityCell,
      enableResizing: true,
      size: 32,
      meta: { styles: { minWidth: 32, maxWidth: 32 } },
    });
  }

  const restColumns: EvalRunsTableColumnDef[] = [
    {
      id: EvalRunsTableColumnId.runName,
      header: () => <FormattedMessage {...EVAL_RUNS_COLUMN_LABELS[EvalRunsTableColumnId.runName]} />,
      cell: RunNameCell,
      enableResizing: true,
      meta: {
        styles: {
          minWidth: 100,
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
      meta: {
        styles: {
          minWidth: 100,
        },
      },
    },
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
