import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { SkillRegistryApi } from '../api';
import type { Skill } from '../types';
import { SKILL_QUERY_KEYS } from '../utils';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';

export const useSkillQuery = (name: string, organization = '') => {
  const activeWorkspace = useActiveWorkspace();

  return useQuery<Skill, Error>([SKILL_QUERY_KEYS.SKILL, name, organization, activeWorkspace], {
    queryFn: () => SkillRegistryApi.getSkill(name, organization),
    retry: false,
    enabled: Boolean(name),
  });
};
