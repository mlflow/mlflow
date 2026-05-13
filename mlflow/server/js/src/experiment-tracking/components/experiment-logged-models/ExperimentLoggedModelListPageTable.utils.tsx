import { first, groupBy, isEmpty, isObject, orderBy } from 'lodash';
import type { LoggedModelProto, RunEntity } from '../../types';
import { useMemo } from 'react';

export enum LoggedModelsTableGroupByMode {
  RUNS = 'runs',
}

export interface LoggedModelDataGroupDataRow {
  isGroup: true;
  groupUuid: string;
  groupData?: {
    sourceRun?: RunEntity;
  };
}

export const isLoggedModelDataGroupDataRow = (data?: LoggedModelsTableDataRow): data is LoggedModelDataGroupDataRow => {
  return isObject(data) && 'isGroup' in data && data.isGroup === true;
};

export const isLoggedModelRow = (data?: LoggedModelsTableDataRow | symbol): data is LoggedModelDataWithSourceRun => {
  return isObject(data) && !isLoggedModelDataGroupDataRow(data);
};

export const LoggedModelsTableLoadMoreRowSymbol = Symbol('LoadMoreRow');

/**
 * Represents a logged model entity enriched with source run
 */
export interface LoggedModelDataWithSourceRun extends LoggedModelProto {
  sourceRun?: RunEntity;
}

/**
 * Represents a >data< row in the logged models table.
 * It's defined to distinguish it from the special "Load more" row.
 */
export type LoggedModelsTableDataRow = LoggedModelDataWithSourceRun | LoggedModelDataGroupDataRow;

/**
 * All possible types of rows in the logged models table.
 */
export type LoggedModelsTableRow = LoggedModelsTableDataRow | typeof LoggedModelsTableLoadMoreRowSymbol;

export const LoggedModelsTableGroupingEnabledClass = 'mlflow-logged-models-table-grouped';
export const LoggedModelsTableGroupHeaderRowClass = 'mlflow-logged-models-table-group-cell';

export enum LoggedModelsTableSpecialRowID {
  LOAD_MORE = 'LOAD_MORE',
  REMAINING_MODELS_GROUP = 'REMAINING_MODELS_GROUP',
}

/**
 * Returns the ID of the logged models table row.
 */
export const getLoggedModelsTableRowID = ({ data }: { data: LoggedModelsTableRow }) => {
  if (!isObject(data)) {
    return LoggedModelsTableSpecialRowID.LOAD_MORE;
  }
  if ('isGroup' in data) {
    return data.groupUuid;
  }
  return data?.info?.model_id ?? '';
};

/**
 * Generates the data rows for the logged models table based on the provided parameters.
 * Supports grouping by source runs if specified.
 */
export const useLoggedModelTableDataRows = ({
  groupModelsBy,
  loggedModelsWithSourceRuns,
  expandedGroups,
}: {
  loggedModelsWithSourceRuns?: LoggedModelDataWithSourceRun[];
  groupModelsBy?: LoggedModelsTableGroupByMode;
  expandedGroups: string[];
}) => {
  return useMemo<LoggedModelsTableDataRow[] | undefined>(() => {
    // If grouping is unavailable or not set, return the original list
    if (!groupModelsBy || isEmpty(loggedModelsWithSourceRuns)) {
      return loggedModelsWithSourceRuns;
    }

    const groups = groupBy(
      loggedModelsWithSourceRuns,
      (loggedModel) => loggedModel.info?.source_run_id ?? LoggedModelsTableSpecialRowID.REMAINING_MODELS_GROUP,
    );

    // Place ungrouped models in a special group at the end
    const sortedGroups = orderBy(
      Object.entries(groups),
      ([groupId]) => groupId !== LoggedModelsTableSpecialRowID.REMAINING_MODELS_GROUP,
      'desc',
    );

    const rows: LoggedModelsTableDataRow[] = [];

    sortedGroups.forEach(([runUuid, models]) => {
      rows.push({
        isGroup: true,
        groupUuid: runUuid,
        groupData: {
          sourceRun: first(models)?.sourceRun,
        },
      });
      if (expandedGroups.includes(runUuid)) {
        rows.push(...models);
      }
    });

    return rows;
  }, [loggedModelsWithSourceRuns, expandedGroups, groupModelsBy]);
};
