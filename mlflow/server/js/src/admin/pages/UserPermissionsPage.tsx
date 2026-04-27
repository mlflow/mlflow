import { useState } from 'react';
import {
  Alert,
  Breadcrumb,
  Button,
  Input,
  PlusIcon,
  SimpleSelect,
  SimpleSelectOption,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { Link, useParams } from '../../common/utils/RoutingUtils';
import AdminRoutes from '../routes';
import { useGrantUserPermission, useUserRolesQuery } from '../hooks';
import { PERMISSIONS } from '../types';

// Resource types supported by the legacy per-user permission CRUD endpoints.
// `workspace` is intentionally excluded — workspace grants flow through roles
// (a role's `roles.workspace` column is the implicit grant target). `scorer`
// is also excluded for now: its identifier is composite (`experiment_id` +
// `scorer_name`) and the form below assumes a single string identifier.
const USER_PERMISSION_RESOURCE_TYPES = [
  'experiment',
  'registered_model',
  'gateway_secret',
  'gateway_endpoint',
  'gateway_model_definition',
] as const;

const RESOURCE_TYPE_LABEL: Record<(typeof USER_PERMISSION_RESOURCE_TYPES)[number], string> = {
  experiment: 'Experiment ID',
  registered_model: 'Registered model name',
  gateway_secret: 'Gateway secret ID',
  gateway_endpoint: 'Gateway endpoint ID',
  gateway_model_definition: 'Gateway model definition ID',
};

const UserPermissionsPage = () => {
  const { theme } = useDesignSystemTheme();
  const { username = '' } = useParams<{ username: string }>();

  const grantPermission = useGrantUserPermission();
  const { data: rolesData, isLoading: rolesLoading } = useUserRolesQuery(username);
  const roles = rolesData?.roles ?? [];

  const [resourceType, setResourceType] = useState<(typeof USER_PERMISSION_RESOURCE_TYPES)[number]>('experiment');
  const [resourceId, setResourceId] = useState('');
  const [permission, setPermission] = useState<string>(PERMISSIONS[0]);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleGrant = async () => {
    setError(null);
    setSuccess(null);
    if (!resourceId.trim()) {
      setError(`${RESOURCE_TYPE_LABEL[resourceType]} is required`);
      return;
    }
    try {
      await grantPermission.mutateAsync({
        resource_type: resourceType,
        resource_id: resourceId.trim(),
        username,
        permission,
      });
      setSuccess(`Granted ${permission} on ${resourceType} "${resourceId.trim()}" to ${username}.`);
      setResourceId('');
    } catch (e: any) {
      setError(e?.message || 'Failed to grant permission');
    }
  };

  return (
    <ScrollablePageWrapper>
      <div css={{ padding: theme.spacing.md, display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        <Breadcrumb includeTrailingCaret>
          <Breadcrumb.Item>
            <Link componentId="admin.user_permissions.breadcrumb_admin" to={AdminRoutes.adminPageRoute}>
              Admin
            </Link>
          </Breadcrumb.Item>
          <Breadcrumb.Item>{username}</Breadcrumb.Item>
        </Breadcrumb>

        <div>
          <Typography.Title level={2} withoutMargins>
            Permissions for <code>{username}</code>
          </Typography.Title>
          <Typography.Text color="secondary">
            Grant per-resource permissions directly to this user. Prefer assigning a role if multiple users need the
            same permissions.
          </Typography.Text>
        </div>

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <Typography.Title level={4} withoutMargins>
            Grant a permission
          </Typography.Title>
          {error && (
            <Alert
              componentId="admin.user_permissions.grant_error"
              type="error"
              message={error}
              closable
              onClose={() => setError(null)}
            />
          )}
          {success && (
            <Alert
              componentId="admin.user_permissions.grant_success"
              type="success"
              message={success}
              closable
              onClose={() => setSuccess(null)}
            />
          )}
          <div>
            <Typography.Text bold>Resource type</Typography.Text>
            <SimpleSelect
              id="admin-user-permissions-resource-type"
              componentId="admin.user_permissions.resource_type"
              value={resourceType}
              onChange={({ target }) => {
                setResourceType(target.value as (typeof USER_PERMISSION_RESOURCE_TYPES)[number]);
                setResourceId('');
              }}
            >
              {USER_PERMISSION_RESOURCE_TYPES.map((rt) => (
                <SimpleSelectOption key={rt} value={rt}>
                  {rt}
                </SimpleSelectOption>
              ))}
            </SimpleSelect>
          </div>
          <div>
            <Typography.Text bold>{RESOURCE_TYPE_LABEL[resourceType]}</Typography.Text>
            <Input
              componentId="admin.user_permissions.resource_id"
              value={resourceId}
              onChange={(e) => setResourceId(e.target.value)}
              placeholder={RESOURCE_TYPE_LABEL[resourceType]}
            />
          </div>
          <div>
            <Typography.Text bold>Permission</Typography.Text>
            <SimpleSelect
              id="admin-user-permissions-level"
              componentId="admin.user_permissions.permission_level"
              value={permission}
              onChange={({ target }) => setPermission(target.value)}
            >
              {PERMISSIONS.map((p) => (
                <SimpleSelectOption key={p} value={p}>
                  {p}
                </SimpleSelectOption>
              ))}
            </SimpleSelect>
          </div>
          <div>
            <Button
              componentId="admin.user_permissions.grant_button"
              type="primary"
              icon={<PlusIcon />}
              onClick={handleGrant}
              loading={grantPermission.isLoading}
            >
              Grant permission
            </Button>
          </div>
        </div>

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <Typography.Title level={4} withoutMargins>
            Roles assigned
          </Typography.Title>
          {rolesLoading ? (
            <Spinner size="small" />
          ) : (
            <Table
              scrollable
              noMinHeight
              empty={<Typography.Text color="secondary">This user has no role assignments yet.</Typography.Text>}
              css={{
                border: `1px solid ${theme.colors.border}`,
                borderRadius: theme.general.borderRadiusBase,
                overflow: 'hidden',
              }}
            >
              <TableRow isHeader>
                <TableHeader componentId="admin.user_permissions.role_name_header" css={{ flex: 2 }}>
                  Role
                </TableHeader>
                <TableHeader componentId="admin.user_permissions.role_workspace_header" css={{ flex: 1 }}>
                  Workspace
                </TableHeader>
              </TableRow>
              {roles.map((role) => (
                <TableRow key={role.id}>
                  <TableCell css={{ flex: 2 }}>
                    <Link componentId="admin.user_permissions.role_link" to={AdminRoutes.getRoleDetailRoute(role.id)}>
                      {role.name}
                    </Link>
                  </TableCell>
                  <TableCell css={{ flex: 1 }}>{role.workspace}</TableCell>
                </TableRow>
              ))}
            </Table>
          )}
        </div>
      </div>
    </ScrollablePageWrapper>
  );
};

export default UserPermissionsPage;
