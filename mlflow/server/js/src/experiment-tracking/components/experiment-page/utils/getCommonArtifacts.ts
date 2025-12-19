import type { ArtifactFileInfo, ArtifactListFilesResponse } from '../../../types';

/**
 * Gets the list of artifacts that are present in all runs in the given list of runs.
 * @param artifactsKeyedByRun Object containing artifacts keyed by run UUID
 * @returns List of common artifacts
 */
export const getCommonArtifacts = (artifactsKeyedByRun: Record<string, ArtifactListFilesResponse>) => {
  const runUuids = Object.keys(artifactsKeyedByRun);

  if (runUuids.length === 0) return [];

  let commonArtifacts = artifactsKeyedByRun[runUuids[0]]?.files
    ?.map((file: ArtifactFileInfo) => (file.is_dir ? null : file.path))
    ?.filter((path: string | null) => path !== null);

  if (!commonArtifacts || commonArtifacts.length === 0) return [];

  for (let i = 1; i < runUuids.length; i++) {
    const currentRunArtifacts = artifactsKeyedByRun[runUuids[i]]?.files?.map((file: any) => file.path);
    commonArtifacts = commonArtifacts?.filter((path: any) => currentRunArtifacts.includes(path));
    if (commonArtifacts.length === 0) {
      break;
    }
  }

  return commonArtifacts;
};
