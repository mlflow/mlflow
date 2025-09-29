import type { ColDef, ColGroupDef } from '@ag-grid-community/core';
import { useMemo, useRef } from 'react';
import { useIntl } from 'react-intl';
import { ExperimentLoggedModelTableNameCell } from '../ExperimentLoggedModelTableNameCell';
import { ExperimentLoggedModelTableDateCell } from '../ExperimentLoggedModelTableDateCell';
import { ExperimentLoggedModelStatusIndicator } from '../ExperimentLoggedModelStatusIndicator';
import { ExperimentLoggedModelTableDatasetCell } from '../ExperimentLoggedModelTableDatasetCell';
import type { LoggedModelProto } from '../../../types';
import { compact, isEqual, values, uniq, orderBy, isObject } from 'lodash';
import { ExperimentLoggedModelTableSourceRunCell } from '../ExperimentLoggedModelTableSourceRunCell';
import {
  ExperimentLoggedModelActionsCell,
  ExperimentLoggedModelActionsHeaderCell,
} from '../ExperimentLoggedModelActionsCell';
import { ExperimentLoggedModelTableRegisteredModelsCell } from '../ExperimentLoggedModelTableRegisteredModelsCell';
import {
  createLoggedModelDatasetColumnGroupId,
  ExperimentLoggedModelTableDatasetColHeader,
} from '../ExperimentLoggedModelTableDatasetColHeader';
import { ExperimentLoggedModelTableSourceCell } from '../ExperimentLoggedModelTableSourceCell';
import { shouldUnifyLoggedModelsAndRegisteredModels } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import {
  LoggedModelsTableGroupHeaderRowClass,
  type LoggedModelsTableRow,
} from '../ExperimentLoggedModelListPageTable.utils';

/**
 * Utility hook that memoizes value based on deep comparison.
 * Helps to regenerate columns only if underlying dependencies change.
 */
const useMemoizeColumns = <T,>(factory: () => T, deps: unknown[], disable?: boolean): T => {
  const ref = useRef<{ deps: unknown[]; value: T }>();

  if (!ref.current || (!isEqual(deps, ref.current.deps) && !disable)) {
    ref.current = { deps, value: factory() };
  }

  return ref.current.value;
};

export enum ExperimentLoggedModelListPageKnownColumnGroups {
  Attributes = 'attributes',
  Params = 'params',
}

export enum ExperimentLoggedModelListPageKnownColumns {
  RelationshipType = 'relationship_type',
  Step = 'step',
  Select = 'select',
  Name = 'name',
  Status = 'status',
  CreationTime = 'creation_time',
  Source = 'source',
  SourceRun = 'source_run_id',
  RegisteredModels = 'registered_models',
  Dataset = 'dataset',
}

export const LOGGED_MODEL_LIST_METRIC_COLUMN_PREFIX = 'metrics.';

export const ExperimentLoggedModelListPageStaticColumns: string[] = [
  ExperimentLoggedModelListPageKnownColumns.Select,
  ExperimentLoggedModelListPageKnownColumns.Name,
  ExperimentLoggedModelListPageKnownColumns.CreationTime,
];

const createDatasetHash = (datasetName?: string, datasetDigest?: string) => {
  if (!datasetName || !datasetDigest) {
    return '';
  }
  return JSON.stringify([datasetName, datasetDigest]);
};

// Creates a metric column ID based on the metric key and optional dataset name and digest.
// The ID format is:
// - `metrics.<datasetHash>.<metricKey>` for metrics grouped by dataset
// - `metrics.<metricKey>` for ungrouped metrics
// The dataset hash is created using the dataset name and digest: [datasetName, datasetDigest]
const createLoggedModelMetricOrderByColumnId = (metricKey: string, datasetName?: string, datasetDigest?: string) => {
  const isUngroupedMetricColumn = !datasetName || !datasetDigest;
  if (isUngroupedMetricColumn) {
    return `${LOGGED_MODEL_LIST_METRIC_COLUMN_PREFIX}${metricKey}`;
  }
  return `${LOGGED_MODEL_LIST_METRIC_COLUMN_PREFIX}${createDatasetHash(datasetName, datasetDigest)}.${metricKey}`;
};

// Parse `metrics.<datasetHash>.<metricKey>` format
// and return dataset name, digest and metric key.
// Make it fall back to default values on error.
export const parseLoggedModelMetricOrderByColumnId = (metricColumnId: string) => {
  const match = metricColumnId.match(/metrics\.(.*?)(?:\.(.*))?$/);
  try {
    if (match) {
      const [, datasetHashOrMetricKey, metricKey] = match;
      if (!metricKey) {
        return { datasetName: undefined, datasetDigest: undefined, metricKey: datasetHashOrMetricKey };
      }
      const [datasetName, datasetDigest] = JSON.parse(datasetHashOrMetricKey);
      return { datasetName, datasetDigest, metricKey };
    }
  } catch (error) {
    // eslint-disable-next-line no-console
    console.error('Failed to parse metric column ID', error);
  }
  return { datasetName: undefined, datasetDigest: undefined, metricKey: metricColumnId };
};

/**
 * Iterate through all logged models and metrics grouped by datasets.
 * Each group is identified by a hashed combination of dataset name and digest.
 * For metrics without dataset, use empty string as a key.
 * The result is a map of dataset hashes to an object containing the dataset name, digest, metrics
 * and the first run ID found for that dataset.
 */
const extractMetricGroups = (loggedModels: LoggedModelProto[]) => {
  const result: Record<string, { datasetDigest?: string; datasetName?: string; runId?: string; metrics: string[] }> =
    {};
  for (const loggedModel of orderBy(loggedModels, (model) => model.info?.model_id)) {
    for (const metric of loggedModel?.data?.metrics ?? []) {
      if (!metric.key) {
        continue;
      }
      const datasetHash =
        metric.dataset_name && metric.dataset_digest
          ? createDatasetHash(metric.dataset_name, metric.dataset_digest)
          : '';

      if (!result[datasetHash]) {
        result[datasetHash] = {
          datasetName: metric.dataset_name,
          datasetDigest: metric.dataset_digest,
          // We use first found run ID, as it will be used for dataset fetching.
          runId: metric.run_id,
          metrics: [],
        };
      }
      if (result[datasetHash] && !result[datasetHash].metrics.includes(metric.key)) {
        result[datasetHash].metrics.push(metric.key);
      }
    }
  }
  return result;
};

const defaultColumnSet = [
  ExperimentLoggedModelListPageKnownColumns.Name,
  ExperimentLoggedModelListPageKnownColumns.Status,
  ExperimentLoggedModelListPageKnownColumns.CreationTime,
  ExperimentLoggedModelListPageKnownColumns.Source,
  ExperimentLoggedModelListPageKnownColumns.SourceRun,
  ExperimentLoggedModelListPageKnownColumns.RegisteredModels,
  ExperimentLoggedModelListPageKnownColumns.Dataset,
];

/**
 * Returns the columns for the logged model list table.
 * Metric column IDs follow the structure:
 * - `metrics.<datasetName>.<metricKey>` for metrics grouped by dataset
 * - `metrics.<metricKey>` for ungrouped metrics
 */
export const useExperimentLoggedModelListPageTableColumns = ({
  columnVisibility = {},
  supportedAttributeColumnKeys = defaultColumnSet,
  loggedModels = [],
  disablePinnedColumns = false,
  disableOrderBy = false,
  enableSortingByMetrics,
  orderByColumn,
  orderByAsc,
  isLoading,
}: {
  loggedModels?: LoggedModelProto[];
  columnVisibility?: Record<string, boolean>;
  disablePinnedColumns?: boolean;
  supportedAttributeColumnKeys?: string[];
  disableOrderBy?: boolean;
  enableSortingByMetrics?: boolean;
  orderByColumn?: string;
  orderByAsc?: boolean;
  isLoading?: boolean;
}) => {
  const datasetMetricGroups = useMemo(() => extractMetricGroups(loggedModels), [loggedModels]);

  const parameterKeys = useMemo(
    () => compact(uniq(loggedModels.map((loggedModel) => loggedModel?.data?.params?.map((param) => param.key)).flat())),
    [loggedModels],
  );

  const intl = useIntl();

  return useMemoizeColumns(
    () => {
      const isUnifiedLoggedModelsEnabled = shouldUnifyLoggedModelsAndRegisteredModels();

      const attributeColumns: ColDef[] = [
        {
          colId: ExperimentLoggedModelListPageKnownColumns.RelationshipType,
          headerName: 'Type',
          sortable: false,
          valueGetter: ({ data }) => {
            return data.direction === 'input'
              ? intl.formatMessage({
                  defaultMessage: 'Input',
                  description:
                    'Label indicating that the logged model was the input of the experiment run. Displayed in logged model list table on the run page.',
                })
              : intl.formatMessage({
                  defaultMessage: 'Output',
                  description:
                    'Label indicating that the logged model was the output of the experiment run Displayed in logged model list table on the run page.',
                });
          },
          pinned: !disablePinnedColumns ? 'left' : undefined,
          resizable: false,
          width: 100,
        },
        {
          colId: ExperimentLoggedModelListPageKnownColumns.Step,
          headerName: intl.formatMessage({
            defaultMessage: 'Step',
            description:
              'Header title for the step column in the logged model list table. Step indicates the run step where the model was logged.',
          }),
          field: 'step',
          valueGetter: ({ data }) => data.step ?? '-',
          pinned: !disablePinnedColumns ? 'left' : undefined,
          resizable: false,
          width: 60,
        },
        {
          headerName: intl.formatMessage({
            defaultMessage: 'Model name',
            description: 'Header title for the model name column in the logged model list table',
          }),
          colId: ExperimentLoggedModelListPageKnownColumns.Name,
          cellRenderer: ExperimentLoggedModelTableNameCell,
          cellClass: ({ data }: { data: LoggedModelsTableRow }) => {
            return isObject(data) && 'isGroup' in data ? LoggedModelsTableGroupHeaderRowClass : '';
          },
          resizable: true,
          pinned: !disablePinnedColumns ? 'left' : undefined,
          minWidth: 140,
          flex: 1,
        },
        {
          headerName: intl.formatMessage({
            defaultMessage: 'Status',
            description: 'Header title for the status column in the logged model list table',
          }),
          cellRenderer: ExperimentLoggedModelStatusIndicator,
          colId: ExperimentLoggedModelListPageKnownColumns.Status,
          pinned: !disablePinnedColumns ? 'left' : undefined,
          width: 140,
          resizable: false,
        },
        {
          headerName: intl.formatMessage({
            defaultMessage: 'Created',
            description: 'Header title for the creation timestamp column in the logged model list table',
          }),
          field: 'info.creation_timestamp_ms',
          colId: ExperimentLoggedModelListPageKnownColumns.CreationTime,
          cellRenderer: ExperimentLoggedModelTableDateCell,
          resizable: true,
          pinned: !disablePinnedColumns ? 'left' : undefined,
          sortable: !disableOrderBy,
          sortingOrder: ['desc', 'asc'],
          comparator: () => 0,
        },
        {
          headerName: intl.formatMessage({
            defaultMessage: 'Logged from',
            description: "Header title for the 'Logged from' column in the logged model list table",
          }),
          colId: ExperimentLoggedModelListPageKnownColumns.Source,
          cellRenderer: ExperimentLoggedModelTableSourceCell,
          resizable: true,
        },
        {
          headerName: intl.formatMessage({
            defaultMessage: 'Source run',
            description: 'Header title for the source run column in the logged model list table',
          }),
          colId: ExperimentLoggedModelListPageKnownColumns.SourceRun,
          cellRenderer: ExperimentLoggedModelTableSourceRunCell,
          resizable: true,
        },
        {
          headerName: intl.formatMessage({
            defaultMessage: 'Registered models',
            description: 'Header title for the registered models column in the logged model list table',
          }),
          colId: ExperimentLoggedModelListPageKnownColumns.RegisteredModels,
          cellRenderer: ExperimentLoggedModelTableRegisteredModelsCell,
          resizable: true,
        },

        {
          headerName: intl.formatMessage({
            defaultMessage: 'Dataset',
            description: 'Header title for the dataset column in the logged model list table',
          }),
          colId: ExperimentLoggedModelListPageKnownColumns.Dataset,
          cellRenderer: ExperimentLoggedModelTableDatasetCell,
          resizable: true,
        },
      ];

      const columnDefs: ColGroupDef[] = [
        {
          groupId: 'attributes',
          headerName: intl.formatMessage({
            defaultMessage: 'Model attributes',
            description: 'Header title for the model attributes section of the logged model list table',
          }),
          children: attributeColumns.filter((column) => {
            // Exclude registered models column when unified logged models feature is enabled
            if (
              isUnifiedLoggedModelsEnabled &&
              column.colId === ExperimentLoggedModelListPageKnownColumns.RegisteredModels
            ) {
              return false;
            }
            return !column.colId || supportedAttributeColumnKeys.includes(column.colId);
          }),
        },
      ];

      const metricGroups = orderBy(values(datasetMetricGroups), (group) => group?.datasetName);

      metricGroups.forEach(({ datasetDigest, datasetName, runId, metrics }) => {
        const isUngroupedMetricColumn = !datasetName || !datasetDigest;
        const headerName = isUngroupedMetricColumn ? '' : `${datasetName} (#${datasetDigest})`;
        columnDefs.push({
          headerName,
          groupId: createLoggedModelDatasetColumnGroupId(datasetName, datasetDigest, runId),
          headerGroupComponent: ExperimentLoggedModelTableDatasetColHeader,
          children:
            metrics?.map((metricKey) => {
              const metricColumnId = createLoggedModelMetricOrderByColumnId(metricKey, datasetName, datasetDigest);
              return {
                headerName: metricKey,
                hide: columnVisibility[metricColumnId] === false,
                colId: metricColumnId,
                valueGetter: ({ data }: { data: LoggedModelProto }) => {
                  // NB: Looping through metric values might not seem to be most efficient, but considering the number
                  // metrics we render on the screen it might be more efficient than creating a lookup table.
                  // Might be revisited if performance becomes an issue.
                  for (const metric of data.data?.metrics ?? []) {
                    if (metric.key === metricKey) {
                      if (metric.dataset_name === datasetName || (!datasetName && !metric.dataset_name)) {
                        return metric.value;
                      }
                    }
                  }
                  return undefined;
                },
                resizable: true,
                sortable: enableSortingByMetrics && !disableOrderBy,
                sortingOrder: ['desc', 'asc'],
                comparator: () => 0,
                sort: enableSortingByMetrics && metricColumnId === orderByColumn ? (orderByAsc ? 'asc' : 'desc') : null,
              };
            }) ?? [],
        });
      });

      if (parameterKeys.length > 0) {
        columnDefs.push({
          headerName: intl.formatMessage({
            defaultMessage: 'Parameters',
            description: 'Header title for the parameters section of the logged model list table',
          }),
          groupId: 'params',
          children: parameterKeys.map((paramKey) => ({
            headerName: paramKey,
            colId: `params.${paramKey}`,
            hide: columnVisibility[`params.${paramKey}`] === false,
            valueGetter: ({ data }: { data: LoggedModelProto }) => {
              for (const param of data.data?.params ?? []) {
                if (param.key === paramKey) {
                  return param.value;
                }
              }
              return undefined;
            },
            resizable: true,
          })),
        });
      }

      const compactColumnDefs = [
        {
          headerCheckboxSelection: false,
          checkboxSelection: false,
          width: 40,
          maxWidth: 40,
          resizable: false,
          colId: ExperimentLoggedModelListPageKnownColumns.Select,
          cellRenderer: ExperimentLoggedModelActionsCell,
          headerComponent: ExperimentLoggedModelActionsHeaderCell,
          flex: undefined,
        },
        {
          headerName: intl.formatMessage({
            defaultMessage: 'Model name',
            description: 'Header title for the model name column in the logged model list table',
          }),
          colId: ExperimentLoggedModelListPageKnownColumns.Name,
          cellRenderer: ExperimentLoggedModelTableNameCell,
          resizable: true,
          flex: 1,
        },
      ];

      return { columnDefs, compactColumnDefs };
    },
    [datasetMetricGroups, parameterKeys, supportedAttributeColumnKeys],
    // Do not recreate column definitions if logged models are being loaded, e.g. due to changing sort order
    isLoading,
  );
};
