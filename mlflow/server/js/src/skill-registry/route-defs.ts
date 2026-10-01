import type { DocumentTitleHandle } from '../common/utils/RoutingUtils';
import { createLazyRouteElement } from '../common/utils/RoutingUtils';
import { SkillRegistryPageId, SkillRegistryRoutePaths } from './routes';

export const getSkillRegistryRouteDefs = () => {
  return [
    {
      path: SkillRegistryRoutePaths.skillRegistryPage,
      element: createLazyRouteElement(() => import('./pages/SkillRegistryPage')),
      pageId: SkillRegistryPageId.skillRegistryPage,
      handle: { getPageTitle: () => 'Skills' } satisfies DocumentTitleHandle,
    },
    {
      path: SkillRegistryRoutePaths.skillDetailPageWithOrganization,
      element: createLazyRouteElement(() => import('./pages/SkillDetailPage')),
      pageId: SkillRegistryPageId.skillDetailPage,
      handle: {
        getPageTitle: (params) => {
          const organization = decodeURIComponent(params['organization'] || '').replace(/^@/, '');
          const name = decodeURIComponent(params['skillName'] || '');
          return `Skill: @${organization}/${name}`;
        },
      } satisfies DocumentTitleHandle,
    },
    {
      path: SkillRegistryRoutePaths.skillDetailPage,
      element: createLazyRouteElement(() => import('./pages/SkillDetailPage')),
      pageId: SkillRegistryPageId.skillDetailPage,
      handle: {
        getPageTitle: (params) => `Skill: ${decodeURIComponent(params['skillName'] || '')}`,
      } satisfies DocumentTitleHandle,
    },
  ];
};
