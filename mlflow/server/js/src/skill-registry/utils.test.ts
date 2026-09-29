import { describe, expect, it } from '@jest/globals';
import { SkillAction, SkillStatus } from './types';
import {
  buildSkillCatalogFilterString,
  escapeFilterLiteral,
  formatSkillIdentity,
  formatSkillOrganization,
  formatSkillSourceLabel,
  formatTagFilterIdentifier,
  getSkillPermissions,
  hasSkillCatalogFilters,
  isSkillDimmed,
  parseSkillRouteParams,
  SKILL_CATALOG_SOURCE_TYPE_OPTIONS,
} from './utils';
import { createMockSkill } from './test-utils';

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

describe('getSkillPermissions', () => {
  it('treats missing allowed_actions as unrestricted', () => {
    expect(getSkillPermissions(createMockSkill())).toEqual({
      canUse: true,
      canUpdate: true,
      canDelete: true,
      canManage: true,
    });
  });

  it('treats an empty allowed_actions list as read-only', () => {
    expect(getSkillPermissions(createMockSkill({ allowed_actions: [] }))).toEqual({
      canUse: false,
      canUpdate: false,
      canDelete: false,
      canManage: false,
    });
  });

  it('exposes matching parent actions', () => {
    expect(getSkillPermissions(createMockSkill({ allowed_actions: [SkillAction.USE, SkillAction.UPDATE] }))).toEqual({
      canUse: true,
      canUpdate: true,
      canDelete: false,
      canManage: false,
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
