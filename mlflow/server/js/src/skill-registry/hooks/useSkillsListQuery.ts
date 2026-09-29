import { SkillRegistryApi } from '../api';
import type { SearchSkillsResponse, Skill } from '../types';
import { buildSkillCatalogFilterString, SKILL_QUERY_KEYS, type SkillCatalogFilters } from '../utils';
import { useCursorPaginatedQuery } from '../../common/hooks/useCursorPaginatedQuery';

export const useSkillsListQuery = ({
  searchText,
  filterActive = false,
  organization,
  tagKey,
  tagValue,
  sourceType = '',
  enabled = true,
}: SkillCatalogFilters & { enabled?: boolean } = {}) => {
  return useCursorPaginatedQuery<SearchSkillsResponse, Skill[]>({
    queryKeyPrefix: SKILL_QUERY_KEYS.SKILLS_LIST,
    searchFilter: searchText,
    extraQueryKeys: { filterActive, organization, tagKey, tagValue, sourceType },
    storageKey: 'skill_registry.page_size',
    queryFn: ({ searchFilter, pageToken, pageSize }) => {
      return SkillRegistryApi.searchSkills({
        filter_string: buildSkillCatalogFilterString({
          searchText: searchFilter,
          filterActive,
          organization,
          tagKey,
          tagValue,
          sourceType,
        }),
        page_token: pageToken,
        max_results: pageSize,
      });
    },
    extractData: (response) => response.skills,
    enabled,
  });
};
