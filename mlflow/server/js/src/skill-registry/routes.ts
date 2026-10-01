import { createMLflowRoutePath } from '../common/utils/RoutingUtils';

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
    return createMLflowRoutePath('/skills/:skillName');
  }

  static get skillDetailPageWithOrganization() {
    // `@` is a literal prefix on the organization segment (`@acme`). React Router
    // does not treat `/skills/@:organization` as a named parameter.
    return createMLflowRoutePath('/skills/:organization/:skillName');
  }
}

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
class SkillRegistryRoutes {
  static get skillRegistryPageRoute() {
    return SkillRegistryRoutePaths.skillRegistryPage;
  }

  static getSkillDetailRoute(name: string, organization = '') {
    const encodedName = encodeURIComponent(name);
    if (organization) {
      return `/skills/@${encodeURIComponent(organization)}/${encodedName}`;
    }
    return `/skills/${encodedName}`;
  }
}

export default SkillRegistryRoutes;
