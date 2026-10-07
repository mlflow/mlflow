import { useMemo } from 'react';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { SkillRegistryApi } from '../api';
import type { SearchSkillVersionsResponse, SkillVersion } from '../types';
import { SKILL_QUERY_KEYS, visibleSkillVersions } from '../utils';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';

export const SKILL_VERSION_LIST_LIMIT = 100;

export const useSkillVersionsQuery = (name: string, organization = '') => {
  const activeWorkspace = useActiveWorkspace();
  const queryResult = useQuery<SearchSkillVersionsResponse, Error>(
    [SKILL_QUERY_KEYS.SKILL_VERSIONS, name, organization, activeWorkspace],
    {
      queryFn: () =>
        SkillRegistryApi.searchSkillVersions(
          name,
          { order_by: ['version DESC'], max_results: SKILL_VERSION_LIST_LIMIT },
          organization,
        ),
      retry: false,
      enabled: Boolean(name),
    },
  );

  const returned = queryResult.data?.skill_versions;
  // The server already omits deleted versions; filtering keeps them hidden if one is ever returned.
  const data = useMemo(() => visibleSkillVersions(returned), [returned]);

  return {
    ...queryResult,
    data,
    hasMoreVersions: Boolean(queryResult.data?.next_page_token),
  };
};

export const useSkillVersionQuery = (name: string, organization = '', version?: number, enabled = true) => {
  const activeWorkspace = useActiveWorkspace();

  return useQuery<SkillVersion, Error>([SKILL_QUERY_KEYS.SKILL_VERSION, name, organization, version, activeWorkspace], {
    queryFn: () => SkillRegistryApi.getSkillVersion(name, version as number, organization),
    retry: false,
    enabled: Boolean(name) && version != null && enabled,
  });
};
