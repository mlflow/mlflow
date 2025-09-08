import { getArtifactChunkedText, getArtifactLocationUrl } from '../../common/utils/ArtifactUtils';
import type { EvaluationArtifactTable, EvaluationArtifactTableEntry } from '../types';

// Reflects structure logged by mlflow.log_table()
export interface RawEvaluationArtifact {
  columns: string[];
  data: (string | number | null | boolean | Record<string, any>)[][];
}

export class EvaluationTableParseError extends Error {}

/**
 * Service function that fetches and parses evaluation artifact table.
 */
export const fetchEvaluationTableArtifact = async (
  runUuid: string,
  artifactPath: string,
): Promise<EvaluationArtifactTable> => {
  const fullArtifactSrcPath = getArtifactLocationUrl(artifactPath, runUuid);

  return getArtifactChunkedText(fullArtifactSrcPath)
    .then((artifactContent) => {
      try {
        return JSON.parse(artifactContent);
      } catch {
        throw new EvaluationTableParseError(`Artifact ${artifactPath} is malformed and/or not valid JSON`);
      }
    })
    .then((data) => parseEvaluationTableArtifact(artifactPath, data));
};

export const parseEvaluationTableArtifact = (
  path: string,
  rawEvaluationArtifact: RawEvaluationArtifact,
): EvaluationArtifactTable => {
  const { columns, data } = rawEvaluationArtifact;
  if (!columns) {
    throw new SyntaxError(`Artifact ${path} is malformed, it does not contain "columns" field`);
  }
  if (!data) {
    throw new SyntaxError(`Artifact ${path} is malformed, it does not contain "data" field`);
  }
  const columnsToIndex = columns.reduce<Record<string, number>>(
    (currentMap, columnName, index) => ({
      ...currentMap,
      [columnName]: index,
    }),
    {},
  );

  const entries: EvaluationArtifactTableEntry[] = [];
  for (const rawDataEntry of data) {
    const entry: EvaluationArtifactTableEntry = {};
    for (const column of columns) {
      entry[column] = rawDataEntry[columnsToIndex[column]];
    }
    entries.push(entry);
  }
  return {
    columns,
    path,
    entries,
    rawArtifactFile: rawEvaluationArtifact,
  };
};
