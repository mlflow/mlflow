import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

import { SkillRegistryApi } from '../api';
import { listSkillFiles, MAX_PREVIEW_BYTES, type SkillFile } from '../skillFiles';
import { SKILL_QUERY_KEYS } from '../utils';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';

// A stored version's content never changes, so neither query needs refetching.
export const useSkillVersionFilesQuery = (artifactPath: string | undefined) => {
  const activeWorkspace = useActiveWorkspace();
  return useQuery([SKILL_QUERY_KEYS.SKILL_VERSION_FILES, artifactPath, activeWorkspace], {
    queryFn: () => listSkillFiles(artifactPath as string),
    enabled: Boolean(artifactPath),
    retry: false,
    staleTime: Infinity,
  });
};

export const useSkillFileContentQuery = (artifactPath: string | undefined, file: SkillFile | undefined) => {
  const activeWorkspace = useActiveWorkspace();
  const tooLarge = (file?.size ?? 0) > MAX_PREVIEW_BYTES;
  const query = useQuery([SKILL_QUERY_KEYS.SKILL_FILE_CONTENT, artifactPath, file?.path, activeWorkspace], {
    queryFn: () => SkillRegistryApi.getArtifactText(`${artifactPath}/${file?.path}`),
    enabled: Boolean(artifactPath && file) && !tooLarge,
    retry: false,
    staleTime: Infinity,
  });
  return { ...query, tooLarge };
};
