import { useMemo } from 'react';
import type { EvaluationDataReduxState } from '../../../reducers/EvaluationDataReducer';

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
      return { columns: [], columnsIntersection: [] };
    }

    // Extract all matching table objects from the store data
    const allTableEntries = comparedRunUuids
      .map((runUuid) =>
        Object.values(artifactsByRun[runUuid] || {}).filter(({ path }) =>
          tableNames.includes(path),
        ),
      )
      .flat();

    // Extract all valid column names
    const allColumnsForAllTables = allTableEntries
      .filter(({ path }) => tableNames.includes(path))
      .map(({ columns }) => columns)
      .flat();

    // Remove duplicates
    const columns = Array.from(new Set(allColumnsForAllTables));

    // Find the intersection
    const columnsIntersection = columns.filter((column) =>
      allTableEntries.every(({ columns: tableColumns }) => tableColumns.includes(column)),
    );

    return {
      columns,
      columnsIntersection,
    };
  }, [comparedRunUuids, artifactsByRun, tableNames]);
