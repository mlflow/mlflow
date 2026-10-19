import { describe, expect, it } from '@jest/globals';
import { matchPath } from '../common/utils/RoutingUtils';
import SkillRegistryRoutes, { SkillRegistryRoutePaths } from './routes';
import { parseSkillRouteParams } from './utils';

describe('Skill Registry routes', () => {
  it('exposes catalog, encoded-identity, and two-segment detail paths', () => {
    expect(SkillRegistryRoutePaths.skillRegistryPage).toBe('/skills');
    expect(SkillRegistryRoutePaths.skillDetailPage).toBe('/skills/:skillKey');
    expect(SkillRegistryRoutePaths.skillDetailPageWithOrganization).toBe('/skills/:organization/:skillName');
  });

  it('builds readable detail URLs for organization-scoped skills', () => {
    expect(SkillRegistryRoutes.skillRegistryPageRoute).toBe('/skills');
    expect(SkillRegistryRoutes.getSkillDetailRoute('code-review')).toBe('/skills/code-review');
    expect(SkillRegistryRoutes.getSkillDetailRoute('code-review', 'acme')).toBe('/skills/@acme/code-review');
    expect(SkillRegistryRoutes.getSkillDetailRoute('code-review', 'acme.labs', 2)).toBe(
      '/skills/@acme.labs/code-review?version=2',
    );
  });

  it('falls back to one encoded segment when a name or organization is not route-safe', () => {
    expect(SkillRegistryRoutes.getSkillDetailRoute('name/with space', 'org/with space')).toBe(
      `/skills/${encodeURIComponent('@org/with space/name/with space')}`,
    );
    expect(SkillRegistryRoutes.getSkillDetailRoute('..', 'acme')).toBe(`/skills/${encodeURIComponent('@acme/..')}`);
  });

  it('matches encoded identity, two-segment, and default-organization URLs', () => {
    const encoded = matchPath(SkillRegistryRoutePaths.skillDetailPage, '/skills/%40acme%2Fcode-review');
    expect(encoded).not.toBeNull();
    expect(parseSkillRouteParams(encoded?.params ?? {})).toEqual({ name: 'code-review', organization: 'acme' });
    const twoSegment = matchPath(SkillRegistryRoutePaths.skillDetailPageWithOrganization, '/skills/@acme/code-review');
    expect(twoSegment).toMatchObject({ params: { organization: '@acme', skillName: 'code-review' } });
    expect(parseSkillRouteParams(twoSegment?.params ?? {})).toEqual({ name: 'code-review', organization: 'acme' });
    expect(matchPath(SkillRegistryRoutePaths.skillDetailPage, '/skills/prompt-style-guide')).toMatchObject({
      params: { skillKey: 'prompt-style-guide' },
    });
  });
});
