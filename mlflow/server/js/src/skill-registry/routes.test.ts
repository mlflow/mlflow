import { describe, expect, it } from '@jest/globals';
import { matchPath } from '../common/utils/RoutingUtils';
import SkillRegistryRoutes, { SkillRegistryRoutePaths } from './routes';

describe('Skill Registry routes', () => {
  it('exposes catalog and organization-qualified detail paths', () => {
    expect(SkillRegistryRoutePaths.skillRegistryPage).toBe('/skills');
    expect(SkillRegistryRoutePaths.skillDetailPage).toBe('/skills/:skillName');
    expect(SkillRegistryRoutePaths.skillDetailPageWithOrganization).toBe('/skills/:organization/:skillName');
  });

  it('builds default and organization-qualified detail URLs', () => {
    expect(SkillRegistryRoutes.skillRegistryPageRoute).toBe('/skills');
    expect(SkillRegistryRoutes.getSkillDetailRoute('code-review')).toBe('/skills/code-review');
    expect(SkillRegistryRoutes.getSkillDetailRoute('code-review', 'acme')).toBe('/skills/@acme/code-review');
    expect(SkillRegistryRoutes.getSkillDetailRoute('name/with space', 'org/with space')).toBe(
      '/skills/@org%2Fwith%20space/name%2Fwith%20space',
    );
  });

  it('matches organization-qualified and default identities', () => {
    expect(
      matchPath(SkillRegistryRoutePaths.skillDetailPageWithOrganization, '/skills/@acme/code-review'),
    ).toMatchObject({ params: { organization: '@acme', skillName: 'code-review' } });
    expect(matchPath(SkillRegistryRoutePaths.skillDetailPage, '/skills/prompt-style-guide')).toMatchObject({
      params: { skillName: 'prompt-style-guide' },
    });
  });
});
