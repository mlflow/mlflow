import { createMLflowRoutePath, generatePath } from '../common/utils/RoutingUtils';

export enum AdminPageId {
  adminPage = 'mlflow.admin',
  workspaceManagementPage = 'mlflow.admin.workspace',
  roleDetailPage = 'mlflow.admin.role-detail',
  userDetailPage = 'mlflow.admin.user-detail',
}

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
export class AdminRoutePaths {
  static get adminPage() {
    return createMLflowRoutePath('/admin');
  }

  // Per-workspace management view. Same component as ``adminPage``; the
  // pathname is the discriminator and the workspace value still travels in
  // the ``?workspace=`` query param so ``WorkspaceRouterSync`` keeps the
  // global ``activeWorkspace`` in sync.
  static get workspaceManagementPage() {
    return createMLflowRoutePath('/admin/ws');
  }

  static get roleDetailPage() {
    return createMLflowRoutePath('/admin/roles/:roleId');
  }

  static get userDetailPage() {
    return createMLflowRoutePath('/admin/users/:username');
  }
}

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
class AdminRoutes {
  static get adminPageRoute() {
    return AdminRoutePaths.adminPage;
  }

  static getWorkspaceManagementRoute(workspaceName: string) {
    // Workspace name validation runs upstream; URL-encode anyway so a name
    // containing ``&`` or ``=`` doesn't corrupt the query string.
    return `${AdminRoutePaths.workspaceManagementPage}?workspace=${encodeURIComponent(workspaceName)}`;
  }

  static getRoleDetailRoute(roleId: number) {
    return generatePath(AdminRoutePaths.roleDetailPage, { roleId: roleId.toString() });
  }

  static getUserDetailRoute(username: string) {
    // The backend's username validation is permissive (non-empty), so the
    // value can contain ``/``, ``?``, or ``%``. URL-encode it so those
    // characters don't break routing or generate ambiguous URLs.
    return generatePath(AdminRoutePaths.userDetailPage, { username: encodeURIComponent(username) });
  }
}

export default AdminRoutes;
