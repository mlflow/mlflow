import type { DocumentTitleHandle } from '../common/utils/RoutingUtils';
import { createLazyRouteElement } from '../common/utils/RoutingUtils';
import { SkillRegistryPageId, SkillRegistryRoutePaths } from './routes';
import { formatSkillIdentity, parseSkillRouteParams } from './utils';

const skillDetailTitle: DocumentTitleHandle['getPageTitle'] = (params) => {
  const { name, organization } = parseSkillRouteParams(params);
  return `Skill: ${formatSkillIdentity(name, organization)}`;
};

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
      handle: { getPageTitle: skillDetailTitle } satisfies DocumentTitleHandle,
    },
    {
      path: SkillRegistryRoutePaths.skillDetailPage,
      element: createLazyRouteElement(() => import('./pages/SkillDetailPage')),
      pageId: SkillRegistryPageId.skillDetailPage,
      handle: { getPageTitle: skillDetailTitle } satisfies DocumentTitleHandle,
    },
  ];
};
