import { useMemo } from 'react';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { SkillRegistryApi } from '../api';
import type { SearchSkillVersionsResponse, SkillVersion } from '../types';
import { SKILL_QUERY_KEYS, withDeletedVersionPlaceholders } from '../utils';
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
  const hasNextPage = Boolean(queryResult.data?.next_page_token);
  const data = useMemo(
    () =>
      withDeletedVersionPlaceholders(returned, {
        completeHistory: !hasNextPage,
        maxRows: SKILL_VERSION_LIST_LIMIT,
      }),
    [returned, hasNextPage],
  );
  const lowestShown = data[data.length - 1]?.version;

  return {
    ...queryResult,
    data,
    hasMoreVersions: hasNextPage || (lowestShown != null && lowestShown > 1 && data.length >= SKILL_VERSION_LIST_LIMIT),
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
