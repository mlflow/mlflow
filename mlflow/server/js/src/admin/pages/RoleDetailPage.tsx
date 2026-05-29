import { useMemo, useState } from 'react';
import {
  Alert,
  Breadcrumb,
  Button,
  Empty,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tabs,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { Link, useParams, useSearchParams } from '../../common/utils/RoutingUtils';
import AdminRoutes from '../routes';
import { useRoleDetailQuery, useRoleUsersQuery, useUsersQuery, useWithSettingsReturnTo } from '../hooks';
import { formatResourcePattern, isWorkspaceAdminRole } from '../types';
import { EditRoleModal } from '../components/EditRoleModal';

const PermissionsSection = ({ roleId }: { roleId: number }) => {
  const { theme } = useDesignSystemTheme();
  const { data: roleData } = useRoleDetailQuery(roleId);
  const permissions = useMemo(() => roleData?.role?.permissions ?? [], [roleData]);

  const emptyState =
    permissions.length === 0 ? (
      <Empty title="No permissions" description="Use Edit role to add permissions to this role." />
    ) : null;

  return (
    <Table
      scrollable
      noMinHeight
      empty={emptyState}
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.general.borderRadiusBase,
        overflow: 'hidden',
      }}
    >
      <TableRow isHeader>
        <TableHeader componentId="admin.role.perm_resource_type_header" css={{ flex: 1 }}>
          Resource Type
        </TableHeader>
        <TableHeader componentId="admin.role.perm_resource_pattern_header" css={{ flex: 1 }}>
          Resource Pattern
        </TableHeader>
        <TableHeader componentId="admin.role.perm_permission_header" css={{ flex: 1 }}>
          Permission
        </TableHeader>
      </TableRow>
      {permissions.map((perm) => (
        <TableRow key={perm.id}>
          <TableCell css={{ flex: 1 }}>
            <Tag componentId="admin.role.permission_resource_type_tag">{perm.resource_type}</Tag>
          </TableCell>
          <TableCell css={{ flex: 1 }}>
            <code>{formatResourcePattern(perm.resource_pattern)}</code>
          </TableCell>
          <TableCell css={{ flex: 1 }}>
            <Tag componentId="admin.role.permission_level_tag" color="indigo">
              {perm.permission}
            </Tag>
          </TableCell>
        </TableRow>
      ))}
    </Table>
  );
};

const AssignedUsersSection = ({ roleId }: { roleId: number }) => {
  const { theme } = useDesignSystemTheme();
  const { data: assignmentsData, isLoading, error: queryError } = useRoleUsersQuery(roleId);
  const { data: usersData, error: usersError } = useUsersQuery();

  const assignments = useMemo(() => assignmentsData?.assignments ?? [], [assignmentsData]);
  const userMap = useMemo(() => {
    const map = new Map<number, string>();
    for (const user of usersData?.users ?? []) {
      map.set(user.id, user.username);
    }
    return map;
  }, [usersData]);

  if (isLoading) {
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: theme.spacing.sm,
          padding: theme.spacing.lg,
          minHeight: 200,
        }}
      >
        <Spinner size="small" />
      </div>
    );
  }

  const fetchError = queryError || usersError;
  if (fetchError) {
    return (
      <Alert
        componentId="admin.role.assignments.fetch_error"
        type="error"
        message="Failed to load assigned users"
        description={(fetchError as Error)?.message || 'An error occurred while fetching data.'}
      />
    );
  }

  const emptyState =
    assignments.length === 0 ? (
      <Empty title="No users assigned" description="Use Edit role to assign users to this role." />
    ) : null;

  return (
    <Table
      scrollable
      noMinHeight
      empty={emptyState}
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.general.borderRadiusBase,
        overflow: 'hidden',
      }}
    >
      <TableRow isHeader>
        <TableHeader componentId="admin.role.assigned_user_id_header" css={{ flex: 1 }}>
          User ID
        </TableHeader>
        <TableHeader componentId="admin.role.assigned_username_header" css={{ flex: 2 }}>
          Username
        </TableHeader>
      </TableRow>
      {assignments.map((assignment) => {
        const username = userMap.get(assignment.user_id);
        return (
          <TableRow key={assignment.id}>
            <TableCell css={{ flex: 1 }}>{assignment.user_id}</TableCell>
            <TableCell css={{ flex: 2 }}>{username ?? 'Unknown'}</TableCell>
          </TableRow>
        );
      })}
    </Table>
  );
};

const RoleDetailPage = () => {
  const { theme } = useDesignSystemTheme();
  const { roleId: roleIdParam } = useParams<{ roleId: string }>();
  const roleId = Number(roleIdParam);
  const isValidRoleId = Number.isFinite(roleId);

  const { data: roleData, isLoading, error: loadError } = useRoleDetailQuery(roleId);
  const withReturnTo = useWithSettingsReturnTo();
  const [searchParams, setSearchParams] = useSearchParams();
  const tabFromUrl = searchParams.get('tab');
  const activeTab = tabFromUrl === 'users' ? 'users' : 'permissions';
  const [editRoleOpen, setEditRoleOpen] = useState(false);

  const role = roleData?.role;

  if (!isValidRoleId) {
    return (
      <ScrollablePageWrapper>
        <div css={{ padding: theme.spacing.md }}>
          <Alert
            componentId="admin.role.invalid_id"
            type="error"
            message="Invalid role ID"
            description="The requested role could not be loaded because the URL contains an invalid role ID."
          />
        </div>
      </ScrollablePageWrapper>
    );
  }

  if (isLoading) {
    return (
      <ScrollablePageWrapper>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: theme.spacing.sm,
            padding: theme.spacing.lg,
            minHeight: 200,
          }}
        >
          <Spinner size="small" />
        </div>
      </ScrollablePageWrapper>
    );
  }

  if (loadError || !role) {
    return (
      <ScrollablePageWrapper>
        <div css={{ padding: theme.spacing.md }}>
          <Alert
            componentId="admin.role.load_error"
            type="error"
            message={(loadError as Error)?.message || 'Role not found'}
          />
        </div>
      </ScrollablePageWrapper>
    );
  }

  return (
    <ScrollablePageWrapper>
      <div css={{ padding: theme.spacing.md, display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Breadcrumb includeTrailingCaret>
            <Breadcrumb.Item>
              {/* Preserve the Roles tab when going "back to Admin" — without
                  the query param, AdminPage falls back to the Users tab.
                  Also carry through the Settings ``returnTo`` param so the
                  Settings exit link still goes back to the originating page
                  if the user entered Admin from Settings. */}
              <Link
                componentId="admin.role.breadcrumb_admin"
                to={withReturnTo(`${AdminRoutes.adminPageRoute}?tab=roles`)}
              >
                Platform Admin
              </Link>
            </Breadcrumb.Item>
            <Breadcrumb.Item>{role.name}</Breadcrumb.Item>
          </Breadcrumb>
          <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                <Typography.Title withoutMargins level={2}>
                  {role.name}
                </Typography.Title>
                {isWorkspaceAdminRole(role) && (
                  // Match the rolesTable column header ("Workspace Manager")
                  // and the per-row tag text ("Manager") used elsewhere.
                  <Tag componentId="admin.role.admin_tag" color="indigo">
                    Manager
                  </Tag>
                )}
              </div>
              <Typography.Text color="secondary">
                {role.description || 'No description'} · Workspace: {role.workspace}
              </Typography.Text>
            </div>
            <Button componentId="admin.role.edit_button" type="primary" onClick={() => setEditRoleOpen(true)}>
              Edit role
            </Button>
          </div>
        </div>

        <Tabs.Root
          componentId="admin.role_detail.tabs"
          valueHasNoPii
          value={activeTab}
          onValueChange={(value) => {
            const next = new URLSearchParams(searchParams);
            if (value === 'permissions') {
              next.delete('tab');
            } else {
              next.set('tab', value);
            }
            setSearchParams(next, { replace: true });
          }}
        >
          <Tabs.List>
            <Tabs.Trigger value="permissions">Permissions</Tabs.Trigger>
            <Tabs.Trigger value="users">Assigned users</Tabs.Trigger>
          </Tabs.List>
          <Tabs.Content value="permissions" css={{ paddingTop: theme.spacing.md }}>
            <PermissionsSection roleId={roleId} />
          </Tabs.Content>
          <Tabs.Content value="users" css={{ paddingTop: theme.spacing.md }}>
            <AssignedUsersSection roleId={roleId} />
          </Tabs.Content>
        </Tabs.Root>

        <EditRoleModal open={editRoleOpen} onClose={() => setEditRoleOpen(false)} roleId={roleId} />
      </div>
    </ScrollablePageWrapper>
  );
};

export default RoleDetailPage;
