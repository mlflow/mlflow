import { createMLflowRoutePath, generatePath } from '../common/utils/RoutingUtils';
import { formatSkillIdentity } from './utils';

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
    // One encoded identity segment so `@` and `/` cannot split the path
    // (`%40acme%2Fcode-review`). Matches the Skill Registry prototype.
    return createMLflowRoutePath('/skills/:skillKey');
  }

  static get skillDetailPageWithOrganization() {
    // Compatibility for unencoded two-segment URLs (`/skills/@acme/code-review`).
    return createMLflowRoutePath('/skills/:organization/:skillName');
  }
}

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
class SkillRegistryRoutes {
  static get skillRegistryPageRoute() {
    return SkillRegistryRoutePaths.skillRegistryPage;
  }

  static getSkillDetailRoute(name: string, organization = '', version?: number) {
    const path = generatePath(SkillRegistryRoutePaths.skillDetailPage, {
      skillKey: encodeURIComponent(formatSkillIdentity(name, organization)),
    });
    if (version != null) {
      return `${path}?version=${encodeURIComponent(String(version))}`;
    }
    return path;
  }
}

export default SkillRegistryRoutes;
