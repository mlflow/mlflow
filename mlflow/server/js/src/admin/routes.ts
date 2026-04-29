import { createMLflowRoutePath, generatePath } from '../common/utils/RoutingUtils';

export enum AdminPageId {
  adminPage = 'mlflow.admin',
  roleDetailPage = 'mlflow.admin.role-detail',
  userPermissionsPage = 'mlflow.admin.user-permissions',
  accountPage = 'mlflow.account',
}

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
export class AdminRoutePaths {
  static get adminPage() {
    return createMLflowRoutePath('/admin');
  }

  static get roleDetailPage() {
    return createMLflowRoutePath('/admin/roles/:roleId');
  }

  static get userPermissionsPage() {
    return createMLflowRoutePath('/admin/users/:username/permissions');
  }

  static get accountPage() {
    return createMLflowRoutePath('/account');
  }
}

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
class AdminRoutes {
  static get adminPageRoute() {
    return AdminRoutePaths.adminPage;
  }

  static get accountPageRoute() {
    return AdminRoutePaths.accountPage;
  }

  static getRoleDetailRoute(roleId: number) {
    return generatePath(AdminRoutePaths.roleDetailPage, { roleId: roleId.toString() });
  }

  static getUserPermissionsRoute(username: string) {
    // The backend's username validation is permissive (non-empty), so the
    // value can contain ``/``, ``?``, or ``%``. URL-encode it so those
    // characters don't break routing or generate ambiguous URLs.
    return generatePath(AdminRoutePaths.userPermissionsPage, { username: encodeURIComponent(username) });
  }
}

export default AdminRoutes;
