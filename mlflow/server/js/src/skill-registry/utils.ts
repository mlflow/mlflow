import type { TagProps } from '@databricks/design-system';
import { PermissionError, NotFoundError } from '@databricks/web-shared/errors';
import { buildSearchFilterClause } from '../common/utils/SearchUtils';
import { sanitizeHref } from '../common/utils/registryIcons';
import {
  SkillAction,
  SkillStatus,
  type Skill,
  type SkillAlias,
  type SkillSourceType,
  type SkillVersion,
} from './types';

export const SKILL_QUERY_KEYS = {
  SKILLS_LIST: 'skills_list',
  SKILL: 'skill',
  SKILL_VERSIONS: 'skill_versions',
  SKILL_VERSION: 'skill_version',
} as const;

// Destinations are provisional until MLflow CLI confirms whether --destination
// names the parent skills directory or a skill-specific folder.
export const SKILL_INSTALL_TARGETS = [
  { id: 'claude-code', label: 'Claude Code', destination: '.claude/skills' },
  { id: 'codex', label: 'Codex', destination: '.agents/skills' },
  { id: 'cursor', label: 'Cursor', destination: '.cursor/skills' },
  { id: 'copilot', label: 'GitHub Copilot', destination: '.github/skills' },
  { id: 'custom', label: 'Any directory', destination: './skills' },
] as const;

export type SkillInstallTargetId = (typeof SKILL_INSTALL_TARGETS)[number]['id'];

export const formatSkillStatusLabel = (status: SkillStatus) => status.charAt(0).toUpperCase() + status.slice(1);

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

export interface SkillSourcePresentation {
  label: string;
  locator: string | null;
  locatorHref?: string;
  path: string | null;
  ref: string | null;
  browseHref?: string;
  showExternalWarning: boolean;
}

/** `git@host:owner/repo.git`: host in group 1, path in group 2. */
export const SCP_GIT_REMOTE_PATTERN = /^git@([^:]+):(.+)$/;

const stripGitSuffix = (url: string) => url.replace(/\.git\/?$/i, '').replace(/\/$/, '');

const encodeGitPath = (value: string) =>
  value
    .split('/')
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join('/');

/** Turn an https clone URL or git/SSH remote into the https repository URL. */
export const toHttpsGitRepoUrl = (source: string): string | undefined => {
  const trimmed = source.trim();
  const https = sanitizeHref(trimmed);
  if (https) {
    return sanitizeHref(stripGitSuffix(https));
  }
  const scp = trimmed.match(SCP_GIT_REMOTE_PATTERN);
  if (scp) {
    return sanitizeHref(`https://${scp[1]}/${stripGitSuffix(scp[2]).replace(/^\/+/, '')}`);
  }
  const ssh = trimmed.match(/^ssh:\/\/(?:[^@]+@)?([^/]+)\/(.+)$/);
  if (ssh) {
    return sanitizeHref(`https://${ssh[1]}/${stripGitSuffix(ssh[2])}`);
  }
  return undefined;
};

export const buildGitBrowseHref = (
  repoUrl: string,
  ref?: string | null,
  subpath?: string | null,
): string | undefined => {
  const revision = ref?.trim();
  if (!revision || /\/(tree|blob|src)\//.test(repoUrl)) {
    return undefined;
  }
  let prefix = 'tree';
  try {
    const host = new URL(repoUrl).hostname;
    if (host === 'bitbucket.org' || host.endsWith('.bitbucket.org')) {
      prefix = 'src';
    } else if (host.split('.').includes('gitlab')) {
      // GitLab, including self-managed hosts such as gitlab.example.com, scopes repository pages under /-/.
      prefix = '-/tree';
    }
  } catch {
    return undefined;
  }
  const path = subpath ? encodeGitPath(subpath) : '';
  const href = path
    ? `${repoUrl}/${prefix}/${encodeGitPath(revision)}/${path}`
    : `${repoUrl}/${prefix}/${encodeGitPath(revision)}`;
  return sanitizeHref(href);
};

export const describeSkillSource = (
  version: Pick<SkillVersion, 'source_type' | 'source' | 'ref' | 'subpath'>,
): SkillSourcePresentation => {
  const label = formatSkillSourceLabel(version.source_type);
  const source = version.source?.trim() || null;
  const path = version.subpath?.trim() || null;
  const ref = version.source_type === 'git' ? version.ref?.trim() || null : null;
  if (!source) {
    return { label, locator: null, path, ref, showExternalWarning: false };
  }

  if (version.source_type === 'git') {
    const repoUrl = toHttpsGitRepoUrl(source);
    const browseHref = repoUrl ? buildGitBrowseHref(repoUrl, ref, path) : undefined;
    return {
      label,
      locator: repoUrl ?? source,
      locatorHref: repoUrl,
      path,
      ref,
      browseHref,
      showExternalWarning: Boolean(repoUrl),
    };
  }

  const href = sanitizeHref(source);
  return {
    label,
    locator: source,
    locatorHref: href,
    path,
    ref: null,
    showExternalWarning: Boolean(href),
  };
};

export const safeDecode = (value: string) => {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
};

export const parseSkillIdentityKey = (skillKey: string) => {
  const decoded = safeDecode(skillKey);
  if (!decoded.startsWith('@')) {
    return { name: decoded, organization: '' };
  }
  const slash = decoded.indexOf('/');
  if (slash === -1) {
    return { name: '', organization: decoded.slice(1) };
  }
  return { name: decoded.slice(slash + 1), organization: decoded.slice(1, slash) };
};

export const parseSkillRouteParams = (params: { skillKey?: string; organization?: string; skillName?: string }) => {
  if (params.skillKey) {
    return parseSkillIdentityKey(params.skillKey);
  }
  return {
    name: safeDecode(params.skillName ?? ''),
    organization: safeDecode(params.organization ?? '').replace(/^@/, ''),
  };
};

export const formatSkillUri = (name: string, organization = '', version?: number) => {
  const identity = formatSkillIdentity(name, organization);
  return version == null ? `skills:/${identity}` : `skills:/${identity}/${version}`;
};

export const formatSkillAliasUri = (name: string, organization = '', alias: string) =>
  `${formatSkillUri(name, organization)}@${alias}`;

export const formatSkillReferenceUris = (name: string, organization = '', version: number, aliases: string[] = []) => [
  formatSkillUri(name, organization, version),
  ...aliases.map((alias) => formatSkillAliasUri(name, organization, alias)),
];

export const aliasesForVersion = (
  skill: Pick<Skill, 'aliases'> | undefined,
  version: Pick<SkillVersion, 'version' | 'aliases'> | undefined,
): string[] => {
  if (!version) {
    return [];
  }
  const fromParent =
    skill?.aliases?.filter((alias) => alias.version === version.version).map((alias) => alias.alias) ?? [];
  return [...new Set([...fromParent, ...(version.aliases ?? [])])];
};

export const visibleSkillVersions = (versions: SkillVersion[] | undefined): SkillVersion[] =>
  (versions ?? []).filter((version) => version.status !== SkillStatus.DELETED);

export const skillVersionStatusTransitions = (status: SkillStatus): SkillStatus[] => {
  switch (status) {
    case SkillStatus.DRAFT:
      return [SkillStatus.ACTIVE];
    case SkillStatus.ACTIVE:
      return [SkillStatus.DRAFT, SkillStatus.DEPRECATED];
    case SkillStatus.DEPRECATED:
      return [SkillStatus.ACTIVE];
    default:
      return [];
  }
};

export const canSoftDeleteSkillVersion = (status: SkillStatus) =>
  status === SkillStatus.DRAFT || status === SkillStatus.DEPRECATED;

export const resolveDefaultSkillVersion = (
  skill?: Pick<Skill, 'latest_version'> | null,
  versions?: SkillVersion[],
): number | undefined => {
  if (skill?.latest_version != null) {
    return skill.latest_version;
  }
  return visibleSkillVersions(versions)[0]?.version;
};

export const parseSkillVersionParam = (value: string | null): number | undefined => {
  if (!value || !/^[1-9][0-9]*$/.test(value)) {
    return undefined;
  }
  return Number(value);
};

export const isPermissionDeniedError = (error: Error | null | undefined) =>
  error instanceof PermissionError || error?.name === 'PermissionError';

export const isNotFoundError = (error: Error | null | undefined) =>
  error instanceof NotFoundError || error?.name === 'NotFoundError';

export const isSkillDimmed = (skill: Skill): boolean => skill.status !== SkillStatus.ACTIVE;

const hasAction = (actions: SkillAction[] | undefined, action: SkillAction) =>
  actions === undefined || actions.includes(action);

/**
 * RFC-0008 gives skills READ, EDIT (UPDATE) and MANAGE (DELETE). Anyone who can see a skill can read and
 * pull it, so there is no separate use check. `allowed_actions` is optional (basic-auth adds it, other
 * auth modes may not), and a missing list leaves enforcement to the server.
 */
export const getSkillPermissions = (skill?: Pick<Skill, 'allowed_actions'>) => {
  if (!skill) {
    return { canUpdate: false, canDelete: false };
  }
  const actions = skill.allowed_actions;
  return {
    canUpdate: hasAction(actions, SkillAction.UPDATE),
    canDelete: hasAction(actions, SkillAction.DELETE),
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
