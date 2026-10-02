import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { SKILL_QUERY_KEYS } from '../utils';

export const useInvalidateSkillQueries = () => {
  const queryClient = useQueryClient();
  return (name: string, organization = '') => {
    queryClient.invalidateQueries([SKILL_QUERY_KEYS.SKILLS_LIST]);
    queryClient.invalidateQueries([SKILL_QUERY_KEYS.SKILL, name, organization]);
    queryClient.invalidateQueries([SKILL_QUERY_KEYS.SKILL_VERSIONS, name, organization]);
    queryClient.invalidateQueries([SKILL_QUERY_KEYS.SKILL_VERSION, name, organization]);
  };
};
