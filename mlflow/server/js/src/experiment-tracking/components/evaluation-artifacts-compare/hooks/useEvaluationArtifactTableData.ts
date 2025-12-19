import { fromPairs, isNil, isObject, isString, sortBy } from 'lodash';
import { useMemo } from 'react';

import type { ArtifactLogTableImageObject, EvaluateCellImage, EvaluationArtifactTableEntry } from '../../../types';
import { PendingEvaluationArtifactTableEntry } from '../../../types';
import type { EvaluationDataReduxState } from '../../../reducers/EvaluationDataReducer';
import { shouldEnablePromptLab } from '../../../../common/utils/FeatureUtils';
import {
  PROMPTLAB_METADATA_COLUMN_LATENCY,
  PROMPTLAB_METADATA_COLUMN_TOTAL_TOKENS,
} from '../../prompt-engineering/PromptEngineering.utils';
import { LOG_TABLE_IMAGE_COLUMN_TYPE } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { getArtifactLocationUrl } from '@mlflow/mlflow/src/common/utils/ArtifactUtils';

type ArtifactsByRun = EvaluationDataReduxState['evaluationArtifactsByRunUuid'];
type PendingDataByRun = EvaluationDataReduxState['evaluationPendingDataByRunUuid'];
type DraftInputValues = EvaluationDataReduxState['evaluationDraftInputValues'];

export type UseEvaluationArtifactTableDataResult = {
  // Unique key for every result row
  key: string;

  // Values of "group by" columns. The column name is the key.
  groupByCellValues: Record<string, string>;

  // Values of output columns. The run uuid is the key.
  cellValues: Record<string, string | EvaluateCellImage>;

  // Contains data describing additional metadata for output: evaluation time, total tokens and a flag
  // indicating if the run was evaluated in this session and is unsynced
  outputMetadataByRunUuid?: Record<string, { isPending: boolean; evaluationTime: number; totalTokens?: number }>;

  isPendingInputRow?: boolean;
}[];

const extractGroupByValuesFromEntry = (entry: EvaluationArtifactTableEntry, groupByCols: string[]) => {
  const groupByMappings = groupByCols.map<[string, string]>((groupBy) => {
    const value = entry[groupBy];
    return [groupBy, isString(value) ? value : JSON.stringify(value)];
  });

  // Next, let's calculate a unique hash for values of those columns - it will serve as
  // an identifier of each result row.
  const groupByHashKey = groupByMappings.map(([, keyValue]) => String(keyValue)).join('.');

  return { key: groupByHashKey, groupByValues: fromPairs(groupByMappings) };
};

/**
 * Consumes table artifact data and based on provided dimensions,
 * generates the data ready to be displayed in the comparison data grid.
 *
 * @param artifactsByRun artifacts-by-run data (extracted from the redux store)
 * @param comparedRunsUuids UUIDs of runs we want to compare
 * @param tableNames table names we want to include in the comparison
 * @param groupByCols list of columns that will be used to group the results by
 * @param outputColumn selects the column to be displayed in the run
 */
export const useEvaluationArtifactTableData = (
  artifactsByRun: ArtifactsByRun,
  pendingDataByRun: PendingDataByRun,
  draftInputValues: DraftInputValues,
  comparedRunsUuids: string[],
  tableNames: string[],
  groupByCols: string[],
  outputColumn: string,
): UseEvaluationArtifactTableDataResult =>
  // eslint-disable-next-line complexity
  useMemo(() => {
    /**
     * End results, i.e. table rows
     */
    const results: UseEvaluationArtifactTableDataResult = [];

    /**
     * An aggregate object containing all output column values.
     * The first level key is the combined hash of all group by values,
     * the second level key is the run UUID. A leaf of this tree corresponds to the output cell value.
     */
    const outputCellsValueMap: Record<string, Record<string, any>> = {};

    /**
     * An aggregate object containing values of the "group by" columns.
     * The first level key is the combined hash of all group by values,
     * the second level key is the "group by" column name. A leaf of this tree corresponds to the cell value.
     */
    const groupByCellsValueMap: Record<string, Record<string, any>> = {};

    /**
     * This array contains all "group by" keys that were freshly added or evaluated, i.e. they are not found
     * in the original evaluation data. This helps to identify them, place them on the top and indicate
     * they're yet to be synchronized.
     */
    const pendingRowKeys: string[] = [];

    /**
     * Start with populating the table with the draft rows created from the draft input sets
     */
    for (const draftInputValueSet of draftInputValues) {
      const visibleGroupByValues = groupByCols.map((colName) => [colName, draftInputValueSet[colName]]);

      const draftInputRowKey = visibleGroupByValues.map(([, value]) => value).join('.');

      // Register new "group by" values combination and mark it as an artificial row
      groupByCellsValueMap[draftInputRowKey] = fromPairs(visibleGroupByValues);
      pendingRowKeys.push(draftInputRowKey);
    }

    const outputMetadataByCellsValueMap: Record<
      string,
      Record<string, { isPending: boolean; evaluationTime: number; totalTokens?: number }>
    > = {};

    // Search through artifact tables and get all entries corresponding to a particular run
    const runsWithEntries = comparedRunsUuids.map<[string, EvaluationArtifactTableEntry[]]>((runUuid) => {
      const baseEntries = Object.values(artifactsByRun[runUuid] || {})
        .filter(({ path }) => tableNames.includes(path))
        .map(({ entries }) => entries)
        .flat();
      return [runUuid, baseEntries];
    });

    // Iterate through all entries and assign them to the corresponding groups.
    for (const [runUuid, entries] of runsWithEntries) {
      for (const entry of entries) {
        const { key, groupByValues } = extractGroupByValuesFromEntry(entry, groupByCols);

        // Do not process the entry that have empty values for all active "group by" columns
        if (Object.values(groupByValues).every((value) => !value)) {
          continue;
        }

        // Assign "group by" column cell values
        if (!groupByCellsValueMap[key]) {
          groupByCellsValueMap[key] = groupByValues;
        }

        // Check if there are values in promptlab metadata columns
        if (entry[PROMPTLAB_METADATA_COLUMN_LATENCY] || entry[PROMPTLAB_METADATA_COLUMN_TOTAL_TOKENS]) {
          if (!outputMetadataByCellsValueMap[key]) {
            outputMetadataByCellsValueMap[key] = {};
          }

          // If true, save it to the record containing output metadata at the index
          // corresponding to a current "group by" key (row) and the run uuid (column)
          // Show the metadata of the most recent value
          if (!outputMetadataByCellsValueMap[key][runUuid]) {
            outputMetadataByCellsValueMap[key][runUuid] = {
              isPending: false,
              evaluationTime: parseFloat(entry[PROMPTLAB_METADATA_COLUMN_LATENCY]),
              totalTokens: entry[PROMPTLAB_METADATA_COLUMN_TOTAL_TOKENS]
                ? parseInt(entry[PROMPTLAB_METADATA_COLUMN_TOTAL_TOKENS], 10)
                : undefined,
            };
          }
        }

        // Assign output column cell values
        if (!outputCellsValueMap[key]) {
          outputCellsValueMap[key] = {};
        }

        const cellsEntry = outputCellsValueMap[key];

        // Use the data from the other set if present, but only if there
        // is no value assigned already. This way we will proritize prepended values.
        cellsEntry[runUuid] = cellsEntry[runUuid] || entry[outputColumn];
      }
    }

    for (const [runUuid, pendingEntries] of Object.entries(pendingDataByRun)) {
      for (const pendingEntry of pendingEntries) {
        const { entryData, ...metadata } = pendingEntry;
        const { key, groupByValues } = extractGroupByValuesFromEntry(entryData, groupByCols);

        // Do not process the entry that have empty values for all active "group by" columns
        if (Object.values(groupByValues).every((value) => !value)) {
          continue;
        }

        // Assign "group by" column cell values
        if (!groupByCellsValueMap[key]) {
          groupByCellsValueMap[key] = groupByValues;

          // If the key was not found in the original set, mark entire row as pending
          pendingRowKeys.push(key);
        }

        if (!outputMetadataByCellsValueMap[key]) {
          outputMetadataByCellsValueMap[key] = {};
        }

        // code pointer for where the metadat is stored
        outputMetadataByCellsValueMap[key][runUuid] = metadata;

        // Assign output column cell values
        if (!outputCellsValueMap[key]) {
          outputCellsValueMap[key] = {};
        }

        const cellsEntry = outputCellsValueMap[key];
        // Use pending data to overwrite already existing result
        cellsEntry[runUuid] = entryData[outputColumn] || cellsEntry[runUuid];
      }
    }

    /**
     * Extract all "group by" keys, i.e. effectively row keys.
     * Hoist all rows that were created during the pending evaluation to the top.
     */
    const allRowKeys = sortBy(Object.entries(groupByCellsValueMap), ([key]) => !pendingRowKeys.includes(key));

    // In the final step, iterate through all found combinations of "group by" values and
    // assign the cells
    for (const [key, groupByCellValues] of allRowKeys) {
      const existingTableRow = results.find(({ key: existingKey }) => key === existingKey);
      if (existingTableRow && outputCellsValueMap[key]) {
        existingTableRow.cellValues = outputCellsValueMap[key];
        existingTableRow.outputMetadataByRunUuid = outputMetadataByCellsValueMap[key];
      } else {
        const cellsEntry = outputCellsValueMap[key];
        Object.keys(cellsEntry || {}).forEach((runUuid: string) => {
          if (cellsEntry[runUuid] !== null && typeof cellsEntry[runUuid] === 'object') {
            try {
              const { type, filepath, compressed_filepath } = cellsEntry[runUuid] as ArtifactLogTableImageObject;
              if (type === LOG_TABLE_IMAGE_COLUMN_TYPE) {
                cellsEntry[runUuid] = {
                  url: getArtifactLocationUrl(filepath, runUuid),
                  compressed_url: getArtifactLocationUrl(compressed_filepath, runUuid),
                };
              } else {
                cellsEntry[runUuid] = JSON.stringify(cellsEntry[runUuid]);
              }
            } catch {
              cellsEntry[runUuid] = '';
            }
          } else if (!isNil(cellsEntry[runUuid]) && !isString(cellsEntry[runUuid])) {
            // stringify non-empty values so that the value
            // doesn't appear as (empty) in the output cell
            // also don't stringify strings, since they'll have
            // an extra quote around them
            cellsEntry[runUuid] = JSON.stringify(cellsEntry[runUuid]);
          }
        });

        results.push({
          key,
          groupByCellValues,
          cellValues: outputCellsValueMap[key] || {},
          outputMetadataByRunUuid: outputMetadataByCellsValueMap[key],
          isPendingInputRow: pendingRowKeys.includes(key),
        });
      }
    }

    return results;
  }, [comparedRunsUuids, artifactsByRun, groupByCols, draftInputValues, tableNames, outputColumn, pendingDataByRun]);
