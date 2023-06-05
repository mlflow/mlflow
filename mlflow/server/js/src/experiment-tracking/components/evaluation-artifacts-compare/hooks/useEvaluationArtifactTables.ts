import { fromPairs } from 'lodash';
import { useMemo } from 'react';
import { RunLoggedArtifactType, RunLoggedArtifactsDeclaration } from '../../../types';
import { MLFLOW_LOGGED_ARTIFACTS_TAG } from '../../../constants';
import { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';

/**
 * Consumes an array of experiment run entities and extracts names of
 * all table artifacts defined by their tags.
 */
export const useEvaluationArtifactTables = (comparedRunRows: RunRowType[]) =>
  useMemo(() => {
    const tablesByRun = fromPairs(
      comparedRunRows
        .map<[string, string[]]>((run) => {
          const rawLoggedArtifactsDeclaration = run.tags?.[MLFLOW_LOGGED_ARTIFACTS_TAG]?.value;
          const tablesInRun: Set<string> = new Set();
          if (rawLoggedArtifactsDeclaration) {
            try {
              const loggedArtifacts: RunLoggedArtifactsDeclaration = JSON.parse(
                rawLoggedArtifactsDeclaration,
              );

              loggedArtifacts
                .filter(({ type }) => type === RunLoggedArtifactType.TABLE)
                .forEach(({ path }) => {
                  tablesInRun.add(path);
                });
            } catch (error) {
              if (error instanceof SyntaxError) {
                throw new SyntaxError(
                  `The "${MLFLOW_LOGGED_ARTIFACTS_TAG}" tag in "${run.runName}" run is malformed!`,
                );
              }
              throw error;
            }
          }
          return [run.runUuid, Array.from(tablesInRun)];
        })
        // Filter entries with no tables reported
        .filter(([, tables]) => tables.length > 0),
    );

    const allUniqueTables = Array.from(new Set(Object.values(tablesByRun).flat()));

    const tablesIntersection = allUniqueTables.filter((tableName) =>
      comparedRunRows.every(({ runUuid }) => tablesByRun[runUuid]?.includes(tableName)),
    );

    return {
      tables: allUniqueTables,
      tablesByRun,
      tablesIntersection,
    };
  }, [comparedRunRows]);
