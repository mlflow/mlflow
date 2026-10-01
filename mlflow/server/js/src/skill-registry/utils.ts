import type { TagProps } from '@databricks/design-system';
import { buildSearchFilterClause } from '../common/utils/SearchUtils';
import { SkillAction, SkillStatus, type Skill, type SkillSourceType } from './types';

export const SKILL_QUERY_KEYS = {
  SKILLS_LIST: 'skills_list',
  SKILL: 'skill',
} as const;

export const STATUS_TAG_COLOR: Record<SkillStatus, TagProps['color']> = {
  [SkillStatus.DRAFT]: 'charcoal',
  [SkillStatus.ACTIVE]: 'lime',
  [SkillStatus.DEPRECATED]: 'lemon',
  [SkillStatus.DELETED]: 'coral',
};

export const SKILL_CATALOG_SOURCE_TYPE_OPTIONS: SkillSourceType[] = ['git', 'oci', 'zip', 'mlflow'];

export const formatSkillIdentity = (name: string, organization = ''): string =>
  organization ? `@${organization}/${name}` : name;

export const formatSkillOrganization = (organization = ''): string => (organization ? `@${organization}` : '');

export const formatSkillSourceLabel = (sourceType?: SkillSourceType | null): string => {
  switch (sourceType) {
    case 'git':
      return 'Git';
    case 'oci':
      return 'OCI image';
    case 'zip':
      return 'ZIP archive';
    case 'mlflow':
      return 'MLflow artifacts';
    default:
      return '';
  }
};

export const parseSkillRouteParams = (params: { organization?: string; skillName?: string }) => ({
  name: decodeURIComponent(params.skillName ?? ''),
  organization: decodeURIComponent(params.organization ?? '').replace(/^@/, ''),
});

export const isSkillDimmed = (skill: Skill): boolean => skill.status !== SkillStatus.ACTIVE;

const hasAction = (actions: SkillAction[] | undefined, action: SkillAction) =>
  actions === undefined || actions.includes(action);

export const getSkillPermissions = (skill?: Pick<Skill, 'allowed_actions'>) => {
  if (!skill) {
    return { canUse: false, canUpdate: false, canDelete: false, canManage: false };
  }
  const actions = skill.allowed_actions;
  return {
    canUse: hasAction(actions, SkillAction.USE),
    canUpdate: hasAction(actions, SkillAction.UPDATE),
    canDelete: hasAction(actions, SkillAction.DELETE),
    canManage: hasAction(actions, SkillAction.MANAGE),
  };
};

export const escapeFilterLiteral = (value: string): string => value.replace(/'/g, "''");

export const formatTagFilterIdentifier = (key: string): string => {
  if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
    return `tags.${key}`;
  }
  return `tags.\`${key.replace(/`/g, '')}\``;
};

export interface SkillCatalogFilters {
  searchText?: string;
  filterActive?: boolean;
  organization?: string;
  tagKey?: string;
  tagValue?: string;
  sourceType?: SkillSourceType | '';
}

export const buildSkillCatalogFilterString = ({
  searchText,
  filterActive,
  organization,
  tagKey,
  tagValue,
  sourceType,
}: SkillCatalogFilters = {}): string | undefined => {
  const clauses = [
    buildSearchFilterClause(searchText?.trim(), 'search_text'),
    filterActive ? `status = '${SkillStatus.ACTIVE}'` : undefined,
    organization?.trim() ? `organization = '${escapeFilterLiteral(organization.trim())}'` : undefined,
    tagKey?.trim() && tagValue?.trim()
      ? `${formatTagFilterIdentifier(tagKey.trim())} = '${escapeFilterLiteral(tagValue.trim())}'`
      : undefined,
    // Parent source_type is the latest-resolved version's source_type (RFC UI contract).
    sourceType ? `source_type = '${sourceType}'` : undefined,
  ].filter(Boolean);

  return clauses.length ? clauses.join(' AND ') : undefined;
};

export const hasSkillCatalogFilters = ({
  searchText,
  filterActive,
  organization,
  tagKey,
  tagValue,
  sourceType,
}: SkillCatalogFilters = {}): boolean =>
  Boolean(
    searchText?.trim() || filterActive || organization?.trim() || (tagKey?.trim() && tagValue?.trim()) || sourceType,
  );
