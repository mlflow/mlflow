import { createMLflowRoutePath } from '../common/utils/RoutingUtils';

export enum AccountPageId {
  accountPage = 'mlflow.account',
}

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
export class AccountRoutePaths {
  static get accountPage() {
    return createMLflowRoutePath('/account');
  }
}

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
class AccountRoutes {
  static get accountPageRoute() {
    return AccountRoutePaths.accountPage;
  }
}

export default AccountRoutes;
