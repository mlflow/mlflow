import { createMLflowRoutePath } from '../common/utils/RoutingUtils';

export enum AdminPageId {
  accountPage = 'mlflow.account',
}

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
export class AdminRoutePaths {
  static get accountPage() {
    return createMLflowRoutePath('/account');
  }
}

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
class AdminRoutes {
  static get accountPageRoute() {
    return AdminRoutePaths.accountPage;
  }
}

export default AdminRoutes;
