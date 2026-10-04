import { createMLflowRoutePath, generatePath } from '../common/utils/RoutingUtils';

export enum AdminPageId {
  adminPage = 'mlflow.admin',
  workspaceManagementPage = 'mlflow.admin.workspace',
  roleDetailPage = 'mlflow.admin.role-detail',
  userDetailPage = 'mlflow.admin.user-detail',
  roleConditionsPage = 'mlflow.admin.role-conditions',
  userConditionsPage = 'mlflow.admin.user-conditions',
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

  // Conditions get their own path rather than a ``?tab=`` on the detail pages: a
  // condition narrows access where a grant widens it, and a peer tab would imply the
  // two combine the same way.
  static get roleConditionsPage() {
    return createMLflowRoutePath('/admin/roles/:roleId/conditions');
  }

  static get userConditionsPage() {
    return createMLflowRoutePath('/admin/users/:username/conditions');
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

  static getRoleConditionsRoute(roleId: number) {
    return generatePath(AdminRoutePaths.roleConditionsPage, { roleId: roleId.toString() });
  }

  static getUserConditionsRoute(username: string) {
    // Same encoding as ``getUserDetailRoute`` -- the backend's username validation is
    // permissive, so the value can contain ``/``, ``?``, or ``%``.
    return generatePath(AdminRoutePaths.userConditionsPage, { username: encodeURIComponent(username) });
  }

  static getUserDetailRoute(username: string) {
    // The backend's username validation is permissive (non-empty), so the
    // value can contain ``/``, ``?``, or ``%``. URL-encode it so those
    // characters don't break routing or generate ambiguous URLs.
    return generatePath(AdminRoutePaths.userDetailPage, { username: encodeURIComponent(username) });
  }
}

export default AdminRoutes;
