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

  it('builds encoded detail URLs so @ and / cannot split the path', () => {
    expect(SkillRegistryRoutes.skillRegistryPageRoute).toBe('/skills');
    expect(SkillRegistryRoutes.getSkillDetailRoute('code-review')).toBe('/skills/code-review');
    expect(SkillRegistryRoutes.getSkillDetailRoute('code-review', 'acme')).toBe('/skills/%40acme%2Fcode-review');
    expect(SkillRegistryRoutes.getSkillDetailRoute('code-review', 'acme', 2)).toBe(
      '/skills/%40acme%2Fcode-review?version=2',
    );
    expect(SkillRegistryRoutes.getSkillDetailRoute('name/with space', 'org/with space')).toBe(
      `/skills/${encodeURIComponent('@org/with space/name/with space')}`,
    );
  });

  it('matches encoded identity, two-segment, and default-organization URLs', () => {
    const encoded = matchPath(SkillRegistryRoutePaths.skillDetailPage, '/skills/%40acme%2Fcode-review');
    expect(encoded).not.toBeNull();
    expect(parseSkillRouteParams(encoded?.params ?? {})).toEqual({ name: 'code-review', organization: 'acme' });
    expect(
      matchPath(SkillRegistryRoutePaths.skillDetailPageWithOrganization, '/skills/@acme/code-review'),
    ).toMatchObject({ params: { organization: '@acme', skillName: 'code-review' } });
    expect(matchPath(SkillRegistryRoutePaths.skillDetailPage, '/skills/prompt-style-guide')).toMatchObject({
      params: { skillKey: 'prompt-style-guide' },
    });
  });
});
