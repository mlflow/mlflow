import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { SkillRegistryApi } from '../api';
import type { SearchSkillVersionsResponse, SkillVersion } from '../types';
import { SKILL_QUERY_KEYS, visibleSkillVersions } from '../utils';
import { useCursorPaginatedQuery } from '../../common/hooks/useCursorPaginatedQuery';
import { useActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';

export const useSkillVersionsQuery = (name: string, organization = '') => {
  const activeWorkspace = useActiveWorkspace();

  return useCursorPaginatedQuery<SearchSkillVersionsResponse, SkillVersion[]>({
    queryKeyPrefix: SKILL_QUERY_KEYS.SKILL_VERSIONS,
    extraQueryKeys: { name, organization, activeWorkspace },
    storageKey: 'skill_registry.versions.page_size',
    queryFn: ({ pageToken, pageSize }) => {
      return SkillRegistryApi.searchSkillVersions(
        name,
        {
          order_by: ['version DESC'],
          page_token: pageToken,
          max_results: pageSize,
        },
        organization,
      );
    },
    extractData: (response) => visibleSkillVersions(response.skill_versions),
    enabled: Boolean(name),
    keepPreviousData: false,
  });
};

export const useSkillVersionQuery = (name: string, organization = '', version?: number, enabled = true) => {
  const activeWorkspace = useActiveWorkspace();

  return useQuery<SkillVersion, Error>([SKILL_QUERY_KEYS.SKILL_VERSION, name, organization, version, activeWorkspace], {
    queryFn: () => SkillRegistryApi.getSkillVersion(name, version as number, organization),
    retry: false,
    enabled: Boolean(name) && version != null && enabled,
  });
};
