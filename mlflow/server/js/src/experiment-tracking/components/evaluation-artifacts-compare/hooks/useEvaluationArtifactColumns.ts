import { useMemo } from 'react';
import type { EvaluationDataReduxState } from '../../../reducers/EvaluationDataReducer';
import type { ArtifactLogTableImageObject } from '@mlflow/mlflow/src/experiment-tracking/types';

type ArtifactsByRun = EvaluationDataReduxState['evaluationArtifactsByRunUuid'];

/**
 * Consumes artifacts data (extracted from the redux store) and based on the
 * provided list of tables and run UUIDs, returns:
 * - list of all columns found in the tables data
 * - list of columns that are present in every matching table
 */
export const useEvaluationArtifactColumns = (
  artifactsByRun: ArtifactsByRun,
  comparedRunUuids: string[],
  tableNames: string[],
) =>
  useMemo(() => {
    // Do not proceed if there are no tables or runs selected
    if (tableNames.length === 0 || comparedRunUuids.length === 0) {
      return { columns: [], columnsIntersection: [], imageColumns: [] };
    }

    // Extract all matching table objects from the store data
    const allTableEntries = comparedRunUuids
      .map((runUuid) => Object.values(artifactsByRun[runUuid] || {}).filter(({ path }) => tableNames.includes(path)))
      .flat();

    // Extract all valid column names
    const allColumnsForAllTables = allTableEntries
      .filter(({ path }) => tableNames.includes(path))
      .map(({ columns, entries }) => {
        return columns.map((column) => {
          const column_string = String(column);
          if (entries.length > 0) {
            const entry = entries[0][column];
            if (typeof entry === 'object' && (entry as ArtifactLogTableImageObject)?.type === 'image') {
              return { name: column_string, type: 'image' };
            } else {
              return { name: column_string, type: 'text' };
            }
          } else {
            return { name: column_string, type: 'text' };
          }
        });
      })
      .flat();

    // Remove duplicates
    const columns = Array.from(
      new Set(allColumnsForAllTables.filter((col) => col.type === 'text').map((col) => col.name)),
    );
    const imageColumns = Array.from(
      new Set(allColumnsForAllTables.filter((col) => col.type === 'image').map((col) => col.name)),
    );
    // Find the intersection
    const columnsIntersection = columns.filter((column) =>
      allTableEntries.every(({ columns: tableColumns }) => tableColumns.includes(column)),
    );

    return {
      columns,
      columnsIntersection,
      imageColumns,
    };
  }, [comparedRunUuids, artifactsByRun, tableNames]);
