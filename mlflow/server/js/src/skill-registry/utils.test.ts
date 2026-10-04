import { describe, expect, it } from '@jest/globals';
import { SkillAction, SkillStatus } from './types';
import {
  aliasesForVersion,
  buildSkillCatalogFilterString,
  buildGitBrowseHref,
  describeSkillSource,
  escapeFilterLiteral,
  formatSkillIdentity,
  formatSkillOrganization,
  formatSkillReferenceUris,
  formatSkillSourceLabel,
  formatSkillUri,
  formatTagFilterIdentifier,
  getSkillPermissions,
  hasSkillCatalogFilters,
  canSoftDeleteSkillVersion,
  isSkillDimmed,
  parseSkillRouteParams,
  parseSkillVersionParam,
  resolveDefaultSkillVersion,
  skillVersionStatusTransitions,
  SKILL_CATALOG_SOURCE_TYPE_OPTIONS,
  visibleSkillVersions,
  withDeletedVersionPlaceholders,
} from './utils';
import { formatSkillPullCli, formatSkillPullPython } from './snippets';
import { createMockSkill, createMockSkillVersion } from './test-utils';

describe('formatSkillIdentity', () => {
  it('prefixes organization-qualified names and leaves the empty organization bare', () => {
    expect(formatSkillIdentity('code-review', 'acme')).toBe('@acme/code-review');
    expect(formatSkillIdentity('prompt-style-guide', '')).toBe('prompt-style-guide');
  });
});

describe('formatSkillOrganization', () => {
  it('prefixes a non-empty organization', () => {
    expect(formatSkillOrganization('ocp-admin')).toBe('@ocp-admin');
    expect(formatSkillOrganization('')).toBe('');
  });
});

describe('formatSkillSourceLabel', () => {
  it('maps stored source types to catalog labels', () => {
    expect(formatSkillSourceLabel('git')).toBe('Git');
    expect(formatSkillSourceLabel('oci')).toBe('OCI image');
    expect(formatSkillSourceLabel('zip')).toBe('ZIP archive');
    expect(formatSkillSourceLabel('mlflow')).toBe('MLflow artifacts');
    expect(formatSkillSourceLabel(null)).toBe('');
  });
});

describe('buildGitBrowseHref', () => {
  it("uses each host family's tree URL", () => {
    expect(buildGitBrowseHref('https://github.com/acme/skills', 'main', 'skills/review')).toBe(
      'https://github.com/acme/skills/tree/main/skills/review',
    );
    expect(buildGitBrowseHref('https://gitlab.com/acme/platform/skills', 'v1.0', 'review')).toBe(
      'https://gitlab.com/acme/platform/skills/-/tree/v1.0/review',
    );
    expect(buildGitBrowseHref('https://gitlab.example.com/acme/skills', 'main')).toBe(
      'https://gitlab.example.com/acme/skills/-/tree/main',
    );
    expect(buildGitBrowseHref('https://bitbucket.org/acme/skills', 'main')).toBe(
      'https://bitbucket.org/acme/skills/src/main',
    );
    expect(buildGitBrowseHref('https://github.com/acme/skills', null)).toBeUndefined();
  });
});

describe('describeSkillSource', () => {
  it('maps a git remote to a repo link, path, ref, and browse URL', () => {
    expect(
      describeSkillSource({
        source_type: 'git',
        source: 'https://github.com/RHEcosystemAppEng/agentic-plugins.git',
        ref: 'main',
        subpath: 'ocp-admin/skills/network-policy-architect',
      }),
    ).toEqual({
      label: 'Git',
      locator: 'https://github.com/RHEcosystemAppEng/agentic-plugins',
      locatorHref: 'https://github.com/RHEcosystemAppEng/agentic-plugins',
      path: 'ocp-admin/skills/network-policy-architect',
      ref: 'main',
      browseHref:
        'https://github.com/RHEcosystemAppEng/agentic-plugins/tree/main/ocp-admin/skills/network-policy-architect',
      showExternalWarning: true,
    });
    expect(
      describeSkillSource({ source_type: 'git', source: 'git@github.com:acme/skills.git', ref: 'main', subpath: null }),
    ).toMatchObject({
      locatorHref: 'https://github.com/acme/skills',
      browseHref: 'https://github.com/acme/skills/tree/main',
    });
  });

  it('links zip URLs, keeps artifact URIs as text, and shows OCI image references with a path', () => {
    expect(
      describeSkillSource({
        source_type: 'zip',
        source: 'https://example.com/skills.zip',
        ref: 'main',
        subpath: 'skills/code-review',
      }),
    ).toMatchObject({
      label: 'ZIP archive',
      locatorHref: 'https://example.com/skills.zip',
      path: 'skills/code-review',
      ref: null,
      showExternalWarning: true,
    });
    expect(
      describeSkillSource({
        source_type: 'mlflow',
        source: 'mlflow-artifacts:/skills/@acme/code-review/2',
        ref: null,
        subpath: null,
      }),
    ).toMatchObject({
      locator: 'mlflow-artifacts:/skills/@acme/code-review/2',
      locatorHref: undefined,
      showExternalWarning: false,
    });
    expect(
      describeSkillSource({
        source_type: 'oci',
        source: 'ghcr.io/acme/skills:v1',
        ref: null,
        subpath: 'skills/code-review',
      }),
    ).toMatchObject({
      label: 'OCI image',
      locator: 'ghcr.io/acme/skills:v1',
      locatorHref: undefined,
      path: 'skills/code-review',
      showExternalWarning: false,
    });
  });
});

describe('parseSkillRouteParams', () => {
  it('decodes organization-qualified and default identities', () => {
    expect(parseSkillRouteParams({ organization: 'acme', skillName: 'code-review' })).toEqual({
      name: 'code-review',
      organization: 'acme',
    });
    expect(parseSkillRouteParams({ organization: '@acme', skillName: 'code-review' })).toEqual({
      name: 'code-review',
      organization: 'acme',
    });
    expect(parseSkillRouteParams({ skillName: 'prompt-style-guide' })).toEqual({
      name: 'prompt-style-guide',
      organization: '',
    });
  });
});

describe('isSkillDimmed', () => {
  it('dims skills whose derived parent status is not active', () => {
    expect(isSkillDimmed(createMockSkill({ status: SkillStatus.ACTIVE }))).toBe(false);
    expect(isSkillDimmed(createMockSkill({ status: SkillStatus.DRAFT }))).toBe(true);
    expect(isSkillDimmed(createMockSkill({ status: null }))).toBe(true);
  });
});

describe('skill version lifecycle', () => {
  it('offers only the stored transitions and never a direct active deletion', () => {
    expect(skillVersionStatusTransitions(SkillStatus.DRAFT)).toEqual([SkillStatus.ACTIVE]);
    expect(skillVersionStatusTransitions(SkillStatus.ACTIVE)).toEqual([SkillStatus.DRAFT, SkillStatus.DEPRECATED]);
    expect(skillVersionStatusTransitions(SkillStatus.DEPRECATED)).toEqual([SkillStatus.ACTIVE]);
    expect(skillVersionStatusTransitions(SkillStatus.DELETED)).toEqual([]);
    expect(canSoftDeleteSkillVersion(SkillStatus.DRAFT)).toBe(true);
    expect(canSoftDeleteSkillVersion(SkillStatus.DEPRECATED)).toBe(true);
    expect(canSoftDeleteSkillVersion(SkillStatus.ACTIVE)).toBe(false);
    expect(canSoftDeleteSkillVersion(SkillStatus.DELETED)).toBe(false);
  });
});

describe('getSkillPermissions', () => {
  it('treats missing allowed_actions as unrestricted', () => {
    expect(getSkillPermissions(createMockSkill())).toEqual({ canUpdate: true, canDelete: true });
  });

  it('treats an empty allowed_actions list as read-only', () => {
    expect(getSkillPermissions(createMockSkill({ allowed_actions: [] }))).toEqual({
      canUpdate: false,
      canDelete: false,
    });
  });

  it('exposes matching parent actions', () => {
    expect(getSkillPermissions(createMockSkill({ allowed_actions: [SkillAction.USE, SkillAction.UPDATE] }))).toEqual({
      canUpdate: true,
      canDelete: false,
    });
  });
});

describe('buildSkillCatalogFilterString', () => {
  it('returns undefined when no filters are set', () => {
    expect(buildSkillCatalogFilterString({})).toBeUndefined();
  });

  it('wraps free-text search in a search_text ILIKE clause', () => {
    expect(buildSkillCatalogFilterString({ searchText: 'review' })).toBe("search_text ILIKE '%review%'");
  });

  it('keeps ordinary phrases containing SQL keywords on the free-text path', () => {
    expect(buildSkillCatalogFilterString({ searchText: 'write in python' })).toBe(
      "search_text ILIKE '%write in python%'",
    );
    expect(buildSkillCatalogFilterString({ searchText: 'write like Shakespeare' })).toBe(
      "search_text ILIKE '%write like Shakespeare%'",
    );
  });

  it('escapes ILIKE wildcards and quotes in free-text search', () => {
    expect(buildSkillCatalogFilterString({ searchText: "O'Brien_100%" })).toBe(
      "search_text ILIKE '%O''Brien\\_100\\%%'",
    );
  });

  it('builds equality clauses for active status, organization, tags, and source type', () => {
    expect(
      buildSkillCatalogFilterString({
        searchText: 'review',
        filterActive: true,
        organization: 'acme',
        tagKey: 'team',
        tagValue: 'platform',
        sourceType: 'git',
      }),
    ).toBe(
      "search_text ILIKE '%review%' AND status = 'active' AND organization = 'acme' AND tags.team = 'platform' AND source_type = 'git'",
    );
  });

  it('backticks tag keys that are not simple identifiers', () => {
    expect(formatTagFilterIdentifier('mlflow.organization')).toBe('tags.`mlflow.organization`');
    expect(buildSkillCatalogFilterString({ tagKey: 'team/name', tagValue: "O'Brien" })).toBe(
      "tags.`team/name` = 'O''Brien'",
    );
  });

  it('does not emit a tag clause until both key and value are present', () => {
    expect(buildSkillCatalogFilterString({ tagKey: 'team' })).toBeUndefined();
    expect(buildSkillCatalogFilterString({ tagValue: 'platform' })).toBeUndefined();
  });

  it('escapes organization literals', () => {
    expect(escapeFilterLiteral("acme's")).toBe("acme''s");
    expect(buildSkillCatalogFilterString({ organization: "acme's" })).toBe("organization = 'acme''s'");
  });
});

describe('hasSkillCatalogFilters', () => {
  it('is true when any catalog filter is active', () => {
    expect(hasSkillCatalogFilters({})).toBe(false);
    expect(hasSkillCatalogFilters({ searchText: 'x' })).toBe(true);
    expect(hasSkillCatalogFilters({ filterActive: true })).toBe(true);
    expect(hasSkillCatalogFilters({ sourceType: 'oci' })).toBe(true);
    expect(hasSkillCatalogFilters({ tagKey: 'team' })).toBe(false);
    expect(hasSkillCatalogFilters({ tagKey: 'team', tagValue: 'platform' })).toBe(true);
  });
});

describe('catalog filter options', () => {
  it('exposes latest-resolved source types for the catalog source filter', () => {
    expect(SKILL_CATALOG_SOURCE_TYPE_OPTIONS).toEqual(['git', 'oci', 'zip', 'mlflow']);
  });
});

describe('parseSkillIdentityKey and parseSkillRouteParams', () => {
  it('parses encoded and decoded skillKey identities', () => {
    expect(parseSkillRouteParams({ skillKey: encodeURIComponent('@acme/code-review') })).toEqual({
      name: 'code-review',
      organization: 'acme',
    });
    expect(parseSkillRouteParams({ skillKey: '@acme/code-review' })).toEqual({
      name: 'code-review',
      organization: 'acme',
    });
    expect(parseSkillRouteParams({ skillKey: 'prompt-style-guide' })).toEqual({
      name: 'prompt-style-guide',
      organization: '',
    });
  });
});

describe('skill URI and pull snippets', () => {
  it('formats pinned and unpinned skills URIs', () => {
    expect(formatSkillUri('code-review', 'acme')).toBe('skills:/@acme/code-review');
    expect(formatSkillUri('code-review', 'acme', 2)).toBe('skills:/@acme/code-review/2');
    expect(formatSkillUri('prompt-style-guide')).toBe('skills:/prompt-style-guide');
    expect(formatSkillReferenceUris('code-review', 'acme', 2, ['production', 'stable'])).toEqual([
      'skills:/@acme/code-review/2',
      'skills:/@acme/code-review@production',
      'skills:/@acme/code-review@stable',
    ]);
  });

  it('formats CLI and Python pull examples for a destination', () => {
    expect(formatSkillPullCli('skills:/@acme/code-review/2', '.claude/skills')).toBe(
      'mlflow skills pull skills:/@acme/code-review/2 \\\n    --destination .claude/skills',
    );
    expect(
      formatSkillPullPython({
        name: 'code-review',
        organization: 'acme',
        version: 2,
        destination: '.cursor/skills',
      }),
    ).toBe(
      'import mlflow.genai\n\nmlflow.genai.pull(\n    name="code-review",\n    organization="acme",\n    version=2,\n    destination=".cursor/skills",\n)',
    );
  });
});

describe('version helpers', () => {
  it('fills missing version numbers with deleted placeholders', () => {
    expect(
      withDeletedVersionPlaceholders([
        createMockSkillVersion({ version: 3, status: SkillStatus.ACTIVE }),
        createMockSkillVersion({ version: 1, status: SkillStatus.DEPRECATED }),
      ]).map((version) => [version.version, version.status]),
    ).toEqual([
      [3, SkillStatus.ACTIVE],
      [2, SkillStatus.DELETED],
      [1, SkillStatus.DEPRECATED],
    ]);
  });

  it('fills deleted numbers below the oldest returned version only for a complete history', () => {
    const versions = [createMockSkillVersion({ version: 3, status: SkillStatus.ACTIVE })];
    expect(withDeletedVersionPlaceholders(versions).map((version) => version.version)).toEqual([3]);
    expect(
      withDeletedVersionPlaceholders(versions, { completeHistory: true }).map((version) => [
        version.version,
        version.status,
      ]),
    ).toEqual([
      [3, SkillStatus.ACTIVE],
      [2, SkillStatus.DELETED],
      [1, SkillStatus.DELETED],
    ]);
  });

  it('caps the rows produced for a long run of deleted numbers', () => {
    const rows = withDeletedVersionPlaceholders(
      [
        createMockSkillVersion({ version: 100000, status: SkillStatus.ACTIVE }),
        createMockSkillVersion({ version: 1, status: SkillStatus.ACTIVE }),
      ],
      { maxRows: 100 },
    );
    expect(rows).toHaveLength(100);
    expect(rows[0].version).toBe(100000);
  });

  it('omits deleted versions from ordinary results', () => {
    expect(
      visibleSkillVersions([
        createMockSkillVersion({ version: 2, status: SkillStatus.ACTIVE }),
        createMockSkillVersion({ version: 1, status: SkillStatus.DELETED }),
      ]).map((version) => version.version),
    ).toEqual([2]);
  });

  it('defaults to latest_version when present', () => {
    expect(resolveDefaultSkillVersion(createMockSkill({ latest_version: 4 }), [])).toBe(4);
    expect(
      resolveDefaultSkillVersion(createMockSkill({ latest_version: null }), [
        createMockSkillVersion({ version: 3, status: SkillStatus.DELETED }),
        createMockSkillVersion({ version: 2, status: SkillStatus.ACTIVE }),
      ]),
    ).toBe(2);
  });

  it('parses positive integer version query params', () => {
    expect(parseSkillVersionParam('2')).toBe(2);
    expect(parseSkillVersionParam('01')).toBeUndefined();
    expect(parseSkillVersionParam('latest')).toBeUndefined();
    expect(parseSkillVersionParam(null)).toBeUndefined();
  });

  it('links only safe http(s) sources', () => {
    const locatorHref = (source: string) =>
      describeSkillSource({ source_type: 'zip', source, ref: null, subpath: null }).locatorHref;
    expect(locatorHref('https://example.com/skill.zip')).toBe('https://example.com/skill.zip');
    expect(locatorHref('mlflow-artifacts:/skills/@acme/code-review/2')).toBeUndefined();
    expect(locatorHref(`${'javascript'}:alert(1)`)).toBeUndefined();
  });

  it('collects aliases that target a version from parent and version records', () => {
    const skill = createMockSkill({
      aliases: [
        { alias: 'prod', version: 2 },
        { alias: 'stable', version: 1 },
      ],
    });
    expect(aliasesForVersion(skill, createMockSkillVersion({ version: 2, aliases: ['current'] }))).toEqual([
      'prod',
      'current',
    ]);
  });
});
