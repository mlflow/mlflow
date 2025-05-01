import type { ColDef, ColGroupDef } from '@ag-grid-community/core';
import { useMemo, useRef } from 'react';
import { useIntl } from 'react-intl';
import { ExperimentLoggedModelTableNameCell } from '../ExperimentLoggedModelTableNameCell';
import { ExperimentLoggedModelTableDateCell } from '../ExperimentLoggedModelTableDateCell';
import { ExperimentLoggedModelStatusIndicator } from '../ExperimentLoggedModelStatusIndicator';
import { ExperimentLoggedModelTableDatasetCell } from '../ExperimentLoggedModelTableDatasetCell';
import { LoggedModelProto } from '../../../types';
import { compact, isEqual, keys, uniq } from 'lodash';
import { ExperimentLoggedModelTableSourceRunCell } from '../ExperimentLoggedModelTableSourceRunCell';
import {
  ExperimentLoggedModelActionsCell,
  ExperimentLoggedModelActionsHeaderCell,
} from '../ExperimentLoggedModelActionsCell';

/**
 * Utility hook that memoizes value based on deep comparison.
 * Helps to regenerate columns only if underlying dependencies change.
 */
const useMemoizeColumns = <T,>(factory: () => T, deps: unknown[]): T => {
  const ref = useRef<{ deps: unknown[]; value: T }>();

  if (!ref.current || !isEqual(deps, ref.current.deps)) {
    ref.current = { deps, value: factory() };
  }

  return ref.current.value;
};

const ungroupedMetricColumnKey = Symbol('ungroupedMetricColumnKey');

const isUngroupedMetricColumnKey = (key: string | symbol): key is symbol => key === ungroupedMetricColumnKey;

export enum ExperimentLoggedModelListPageKnownColumnGroups {
  Attributes = 'attributes',
  Params = 'params',
}

export enum ExperimentLoggedModelListPageKnownColumns {
  RelationshipType = 'relationship_type',
  Select = 'select',
  Name = 'name',
  Status = 'status',
  CreationTime = 'creation_time',
  SourceRun = 'source_run_id',
  Dataset = 'dataset',
}

export const ExperimentLoggedModelListPageStaticColumns: string[] = [
  ExperimentLoggedModelListPageKnownColumns.Select,
  ExperimentLoggedModelListPageKnownColumns.Name,
  ExperimentLoggedModelListPageKnownColumns.CreationTime,
];

const extractMetricGroups = (loggedModels: LoggedModelProto[]) => {
  const result: Record<string | symbol, string[]> = {};
  for (const loggedModel of loggedModels) {
    for (const metric of loggedModel?.data?.metrics ?? []) {
      const datasetName = metric.dataset_name || ungroupedMetricColumnKey;

      if (!result[datasetName]) {
        result[datasetName] = [];
      }
      if (metric.key && !result[datasetName].includes(metric.key)) {
        result[datasetName].push(metric.key);
      }
    }
  }
  return result;
};

const defaultColumnSet = [
  ExperimentLoggedModelListPageKnownColumns.Name,
  ExperimentLoggedModelListPageKnownColumns.Status,
  ExperimentLoggedModelListPageKnownColumns.CreationTime,
  ExperimentLoggedModelListPageKnownColumns.SourceRun,
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
  isCompactMode = false,
  loggedModels = [],
  disablePinnedColumns = false,
  disableOrderBy = false,
}: {
  loggedModels?: LoggedModelProto[];
  columnVisibility?: Record<string, boolean>;
  isCompactMode?: boolean;
  disablePinnedColumns?: boolean;
  supportedAttributeColumnKeys?: string[];
  disableOrderBy?: boolean;
}) => {
  const datasetMetricGroups = useMemo(() => extractMetricGroups(loggedModels), [loggedModels]);
  const datasetMetricKeys = useMemo(
    () => [...keys(datasetMetricGroups), ...Object.getOwnPropertySymbols(datasetMetricGroups)],
    [datasetMetricGroups],
  );
  const parameterKeys = useMemo(
    () => compact(uniq(loggedModels.map((loggedModel) => loggedModel?.data?.params?.map((param) => param.key)).flat())),
    [loggedModels],
  );

  const intl = useIntl();

  return useMemoizeColumns(() => {
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
        headerName: intl.formatMessage({
          defaultMessage: 'Model name',
          description: 'Header title for the model name column in the logged model list table',
        }),
        colId: ExperimentLoggedModelListPageKnownColumns.Name,
        cellRenderer: ExperimentLoggedModelTableNameCell,
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
          defaultMessage: 'Source run',
          description: 'Header title for the source run column in the logged model list table',
        }),
        colId: ExperimentLoggedModelListPageKnownColumns.SourceRun,
        cellRenderer: ExperimentLoggedModelTableSourceRunCell,
        resizable: true,
        pinned: !disablePinnedColumns ? 'left' : undefined,
      },

      {
        headerName: intl.formatMessage({
          defaultMessage: 'Dataset',
          description: 'Header title for the dataset column in the logged model list table',
        }),
        colId: ExperimentLoggedModelListPageKnownColumns.Dataset,
        cellRenderer: ExperimentLoggedModelTableDatasetCell,
        resizable: true,
        pinned: !disablePinnedColumns ? 'left' : undefined,
      },
    ];

    const columns: ColGroupDef[] = [
      {
        groupId: 'attributes',
        headerName: intl.formatMessage({
          defaultMessage: 'Model attributes',
          description: 'Header title for the model attributes section of the logged model list table',
        }),
        children: attributeColumns.filter(
          (column) => !column.colId || supportedAttributeColumnKeys.includes(column.colId),
        ),
      },
    ];

    datasetMetricKeys.forEach((datasetName) => {
      const headerName = typeof datasetName === 'symbol' ? '' : datasetName;
      columns.push({
        headerName,
        groupId: isUngroupedMetricColumnKey(datasetName) ? 'metrics' : `metrics.${datasetName}`,
        children: datasetMetricGroups[datasetName].map((metricKey) => ({
          headerName: metricKey,
          hide:
            columnVisibility[
              isUngroupedMetricColumnKey(datasetName) ? `metrics.${metricKey}` : `metrics.${datasetName}.${metricKey}`
            ] === false,
          colId: isUngroupedMetricColumnKey(datasetName)
            ? `metrics.${metricKey}`
            : `metrics.${datasetName}.${metricKey}`,
          valueGetter: ({ data }: { data: LoggedModelProto }) => {
            // NB: Looping through metric values might not seem to be most efficient, but considering the number
            // metrics we render on the screen it might be more efficient than creating a lookup table.
            // Might be revisited if performance becomes an issue.
            for (const metric of data.data?.metrics ?? []) {
              if (metric.key === metricKey) {
                if (metric.dataset_name === datasetName || (typeof datasetName === 'symbol' && !metric.dataset_name)) {
                  return metric.value;
                }
              }
            }
            return undefined;
          },
          resizable: true,
        })),
      });
    });

    if (parameterKeys.length > 0) {
      columns.push({
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

    if (isCompactMode) {
      return [
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
    }

    return columns;
  }, [datasetMetricGroups, datasetMetricKeys, parameterKeys, isCompactMode, supportedAttributeColumnKeys]);
};
