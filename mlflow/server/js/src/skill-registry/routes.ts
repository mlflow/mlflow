import { createMLflowRoutePath, generatePath } from '../common/utils/RoutingUtils';
import { formatSkillIdentity } from './utils';

// Validated names and organizations fit this, so they need no encoding in a readable path.
const ROUTE_SAFE_SEGMENT = /^[a-z0-9][a-z0-9.-]*$/;

export enum SkillRegistryPageId {
  skillRegistryPage = 'mlflow.skill-registry',
  skillDetailPage = 'mlflow.skill-registry.skill-detail',
}

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
export class SkillRegistryRoutePaths {
  static get skillRegistryPage() {
    return createMLflowRoutePath('/skills');
  }

  static get skillDetailPage() {
    // Default-organization skills, or the whole identity encoded into one segment so `@` and `/`
    // cannot split the path (`%40acme%2Fcode-review`), which older links use.
    return createMLflowRoutePath('/skills/:skillKey');
  }

  static get skillDetailPageWithOrganization() {
    // Organization-scoped skills (`/skills/@acme/code-review`).
    return createMLflowRoutePath('/skills/:organization/:skillName');
  }
}

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
class SkillRegistryRoutes {
  static get skillRegistryPageRoute() {
    return SkillRegistryRoutePaths.skillRegistryPage;
  }

  static getSkillDetailRoute(name: string, organization = '', version?: number) {
    const path =
      organization && ROUTE_SAFE_SEGMENT.test(organization) && ROUTE_SAFE_SEGMENT.test(name)
        ? generatePath(SkillRegistryRoutePaths.skillDetailPageWithOrganization, {
            organization: `@${organization}`,
            skillName: name,
          })
        : generatePath(SkillRegistryRoutePaths.skillDetailPage, {
            skillKey: encodeURIComponent(formatSkillIdentity(name, organization)),
          });
    if (version != null) {
      return `${path}?version=${encodeURIComponent(String(version))}`;
    }
    return path;
  }
}

export default SkillRegistryRoutes;
