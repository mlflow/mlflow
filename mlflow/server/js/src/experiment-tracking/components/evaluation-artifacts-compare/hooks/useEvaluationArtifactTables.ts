import { fromPairs } from 'lodash';
import { useMemo } from 'react';
import type { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import { extractLoggedTablesFromRunTags } from '../../../utils/ArtifactUtils';

/**
 * Consumes an array of experiment run entities and extracts names of
 * all table artifacts defined by their tags.
 */
export const useEvaluationArtifactTables = (comparedRunRows: RunRowType[]) =>
  useMemo(() => {
    const tablesByRun = fromPairs(
      comparedRunRows
        .map<[string, string[]]>((run) => {
          const tablesInRun = run.tags ? extractLoggedTablesFromRunTags(run.tags) : [];
          return [run.runUuid, tablesInRun];
        })
        // Filter entries with no tables reported
        .filter(([, tables]) => tables.length > 0),
    );

    const allUniqueTables = Array.from(new Set(Object.values(tablesByRun).flat()));

    const tablesIntersection = allUniqueTables.filter((tableName) =>
      comparedRunRows.every(({ runUuid }) => tablesByRun[runUuid]?.includes(tableName)),
    );

    const noEvalTablesLogged = allUniqueTables.length === 0;

    return {
      tables: allUniqueTables,
      tablesByRun,
      tablesIntersection,
      noEvalTablesLogged,
    };
  }, [comparedRunRows]);
