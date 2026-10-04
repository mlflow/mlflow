import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { SKILL_QUERY_KEYS } from '../utils';

/** Invalidates one skill's queries and the catalog, or every skill query when no name is given. */
export const useInvalidateSkillQueries = () => {
  const queryClient = useQueryClient();
  return (name?: string, organization = '') => {
    const scope = name == null ? [] : [name, organization];
    queryClient.invalidateQueries([SKILL_QUERY_KEYS.SKILLS_LIST]);
    queryClient.invalidateQueries([SKILL_QUERY_KEYS.SKILL, ...scope]);
    queryClient.invalidateQueries([SKILL_QUERY_KEYS.SKILL_VERSIONS, ...scope]);
    queryClient.invalidateQueries([SKILL_QUERY_KEYS.SKILL_VERSION, ...scope]);
  };
};
