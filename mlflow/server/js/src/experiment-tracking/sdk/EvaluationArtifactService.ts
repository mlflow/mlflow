import { getArtifactChunkedText } from '../../common/utils/ArtifactUtils';
import { EvaluationArtifactTable, EvaluationArtifactTableEntry } from '../types';
import { getSrc as getArtifactSrcPath } from '../components/artifact-view-components/ShowArtifactPage';

// Reflects structure logged by mlflow.log_table()
interface RawEvaluationArtifact {
  columns: string[];
  data: string[][];
}

/**
 * Service function that fetches and parses evaluation artifact table.
 */
export const fetchEvaluationTableArtifact = async (
  runUuid: string,
  artifactPath: string,
): Promise<EvaluationArtifactTable> => {
  // This will be improved after moving `/get-artifact` to the /ajax-api/ service prefix
  const fullArtifactSrcPath = getArtifactSrcPath(artifactPath, runUuid);

  return getArtifactChunkedText(fullArtifactSrcPath)
    .then((artifactContent) => {
      try {
        return JSON.parse(artifactContent);
      } catch {
        throw new Error(`Artifact ${artifactPath} is malformed and/or not valid JSON`);
      }
    })
    .then((data) => parseEvaluationTableArtifact(artifactPath, data));
};

const parseEvaluationTableArtifact = (
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
  };
};
