import { fromPairs } from 'lodash';
import { useMemo } from 'react';

import { EvaluationArtifactTableEntry } from '../../../types';
import type { EvaluationDataReduxState } from '../../../reducers/EvaluationDataReducer';

type ArtifactsByRun = EvaluationDataReduxState['evaluationArtifactsByRunUuid'];

export type UseEvaluationArtifactTableDataResult = {
  // Unique key for every result row
  key: string;

  // Values of "group by" columns. The column name is the key.
  groupByCellValues: Record<string, string>;

  // Values of output columns. The run uuid is the key.
  cellValues: Record<string, string>;
}[];

/**
 * Consumes table artifact data and based on provided dimensions,
 * generates the data ready to be displayed in the comparison data grid.
 *
 * @param artifactsByRun artifacts-by-run data (extracted from the redux store)
 * @param comparedRunsUuids UUIDs of runs we want to compare
 * @param tableNames table names we want to include in the comparison
 * @param groupByCols list of columns that will be used to group the results by
 * @param outputColumn selects the column to be displayed in the run
 * @param intersectingOnly if set to true, only generate rows where the output column has a value for every run
 */
export const useEvaluationArtifactTableData = (
  artifactsByRun: ArtifactsByRun,
  comparedRunsUuids: string[],
  tableNames: string[],
  groupByCols: string[],
  outputColumn: string,
  intersectingOnly = false,
): UseEvaluationArtifactTableDataResult =>
  useMemo(() => {
    /**
     * An aggregate object containing all result values.
     * The first level key is the combined hash of all group by values,
     * the second level key is the run UUID. A leaf of this tree corresponds to the output cell value.
     */
    const outputCellsValueMap: Record<string, Record<string, string>> = {};

    /**
     * Similar object containing values of the "group by" columns.
     * The first level key is the combined hash of all group by values,
     * the second level key is the "group by" column name. A leaf of this tree corresponds to the cell value.
     */
    const groupByCellsValueMap: Record<string, Record<string, string>> = {};

    // Search through artifact tables and get all entries corresponding to a particular run
    const runsWithEntries = comparedRunsUuids.map<[string, EvaluationArtifactTableEntry[]]>(
      (runUuid) => {
        const selectedTablesForRun = Object.values(artifactsByRun[runUuid] || {})
          .filter(({ path }) => tableNames.includes(path))
          .map(({ entries }) => entries)
          .flat();
        return [runUuid, selectedTablesForRun];
      },
    );

    // Iterate through all entries and assign them to the corresponding groups.
    for (const [runUuid, entries] of runsWithEntries) {
      for (const entry of entries) {
        // For each entry in the input tables, find values of columns selected to be grouped by.
        const groupByMappings = groupByCols.map<[string, string]>((groupBy) => [
          groupBy,
          entry[groupBy]?.toString(),
        ]);

        // Next, let's calculate a unique hash for values of those columns - it will serve as
        // an identifier of each result row.
        const groupByHashKey = groupByMappings.map(([, keyValue]) => keyValue).join('.');

        // Assign "group by" column cell values
        if (!groupByCellsValueMap[groupByHashKey]) {
          groupByCellsValueMap[groupByHashKey] = fromPairs(groupByMappings);
        }

        // Assignoutput column cell values
        if (!outputCellsValueMap[groupByHashKey]) {
          outputCellsValueMap[groupByHashKey] = {};
        }

        const cellsEntry = outputCellsValueMap[groupByHashKey];
        if (cellsEntry) {
          // Override with the data from the other set if present.
          // If not, retain previous value.
          cellsEntry[runUuid] = entry[outputColumn]?.toString() || cellsEntry[runUuid];
        }
      }
    }

    // In the final step, iterate through all found combinations of "group by" values and
    // assign the cells
    const results: UseEvaluationArtifactTableDataResult = [];

    for (const [key, groupByCellValues] of Object.entries(groupByCellsValueMap)) {
      const cellValues = outputCellsValueMap[key];
      // If `intersectingOnly` is set to true, check if every compared run has the value
      // for this particular group by row.
      const shouldBeAddedToResultSet =
        !intersectingOnly || comparedRunsUuids.every((runUuid) => cellValues[runUuid]);
      if (shouldBeAddedToResultSet) {
        results.push({
          key,
          groupByCellValues,
          cellValues: outputCellsValueMap[key] || {},
        });
      }
    }

    return results;
  }, [comparedRunsUuids, artifactsByRun, groupByCols, intersectingOnly, tableNames, outputColumn]);
