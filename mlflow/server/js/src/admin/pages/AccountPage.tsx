import { useState } from 'react';
import {
  Alert,
  Button,
  Empty,
  Input,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tag,
  Typography,
  UserIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useCurrentUserQuery, useUpdatePassword, useUserRolesQuery } from '../hooks';
import { isWorkspaceAdminRole } from '../types';

const AccountPage = () => {
  const { theme } = useDesignSystemTheme();
  const queryClient = useQueryClient();
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const updatePassword = useUpdatePassword();
  const { data: currentUserData } = useCurrentUserQuery();
  const username = currentUserData?.user?.username ?? '';

  const { data: rolesData, isLoading: rolesLoading } = useUserRolesQuery(username);
  const roles = rolesData?.roles ?? [];

  const handleChangePassword = async () => {
    setError(null);
    setSuccessMessage(null);

    if (!newPassword) {
      setError('Password cannot be empty');
      return;
    }

    if (newPassword !== confirmPassword) {
      setError('New password and confirmation do not match');
      return;
    }

    try {
      await updatePassword.mutateAsync({ username, password: newPassword });
      setSuccessMessage('Password updated successfully');
      setNewPassword('');
      setConfirmPassword('');
    } catch (e: any) {
      setError(e.message || 'Failed to update password');
    }
  };

  const handleLogout = () => {
    const expired = 'expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
    document.cookie = `mlflow_user=; ${expired}`;
    // Also clear the dev-switcher auth header cookie if it was set
    document.cookie = `mlflow-request-header-Authorization=; ${expired}`;

    // Drop cached React Query data so stale admin/user info doesn't linger
    // if the user navigates back.
    queryClient.clear();

    // Navigate (not fetch) to the prefix-aware /logout route. The server
    // returns a 200 HTML page that runs a synchronous XHR with bogus
    // credentials against /ajax-api/2.0/mlflow/users/current. That XHR
    // receives 401 with WWW-Authenticate, which causes the browser to drop
    // its cached Basic Auth credentials. The page then shows a link back to
    // the app, where the next request will prompt for fresh credentials.
    //
    // Resolving 'logout' relative to ``window.location.href`` preserves any
    // static prefix (the backend registers /logout via ``_add_static_prefix``)
    // so deployments served under a sub-path continue to work.
    window.location.href = new URL('logout', window.location.href).toString();
  };

  const rolesEmptyState =
    roles.length === 0 ? <Empty title="No roles" description="You have not been assigned to any roles." /> : null;

  return (
    <ScrollablePageWrapper>
      <div css={{ padding: theme.spacing.md, display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
            <div
              css={{
                borderRadius: theme.borders.borderRadiusSm,
                backgroundColor: theme.colors.backgroundSecondary,
                padding: theme.spacing.sm,
                display: 'flex',
              }}
            >
              <UserIcon />
            </div>
            <Typography.Title withoutMargins level={2}>
              <FormattedMessage defaultMessage="Account" description="Account page title" />
            </Typography.Title>
          </div>
          {username && (
            <Typography.Text color="secondary">
              Logged in as <strong>{username}</strong>
            </Typography.Text>
          )}
        </div>

        {!username && (
          <Alert
            componentId="account.no_user"
            type="warning"
            message="Not logged in"
            description="Could not determine the current user. Please log in again."
          />
        )}

        {error && (
          <Alert componentId="account.error" type="error" message={error} closable onClose={() => setError(null)} />
        )}
        {successMessage && (
          <Alert
            componentId="account.info"
            type="info"
            message={successMessage}
            closable
            onClose={() => setSuccessMessage(null)}
          />
        )}

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, maxWidth: 600 }}>
          <Typography.Title withoutMargins level={4}>
            Change Password
          </Typography.Title>
          <div>
            <Typography.Text bold>New Password</Typography.Text>
            <Input
              componentId="account.new_password"
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              placeholder="Enter new password"
            />
          </div>
          <div>
            <Typography.Text bold>Confirm Password</Typography.Text>
            <Input
              componentId="account.confirm_password"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="Confirm new password"
            />
          </div>
          <div>
            <Button
              componentId="account.change_password_button"
              type="primary"
              onClick={handleChangePassword}
              loading={updatePassword.isLoading}
              disabled={!username}
            >
              <FormattedMessage defaultMessage="Update Password" description="Button to update password" />
            </Button>
          </div>
        </div>

        {username && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <Typography.Title withoutMargins level={4}>
              My Roles
            </Typography.Title>
            {rolesLoading ? (
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
            ) : (
              <Table
                scrollable
                noMinHeight
                empty={rolesEmptyState}
                css={{
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.general.borderRadiusBase,
                  overflow: 'hidden',
                }}
              >
                <TableRow isHeader>
                  <TableHeader componentId="account.roles.name_header" css={{ flex: 2 }}>
                    Role
                  </TableHeader>
                  <TableHeader componentId="account.roles.workspace_header" css={{ flex: 1 }}>
                    Workspace
                  </TableHeader>
                  <TableHeader componentId="account.roles.admin_header" css={{ flex: 1 }}>
                    Workspace Manager
                  </TableHeader>
                </TableRow>
                {roles.map((role) => (
                  <TableRow key={role.id}>
                    <TableCell css={{ flex: 2 }}>{role.name}</TableCell>
                    <TableCell css={{ flex: 1 }}>{role.workspace}</TableCell>
                    <TableCell css={{ flex: 1 }}>
                      {isWorkspaceAdminRole(role) ? (
                        <Tag componentId="account.role_admin_tag" color="indigo">
                          Manager
                        </Tag>
                      ) : null}
                    </TableCell>
                  </TableRow>
                ))}
              </Table>
            )}
          </div>
        )}

        <div>
          <Button componentId="account.logout_button" onClick={handleLogout}>
            <FormattedMessage defaultMessage="Logout" description="Button to logout" />
          </Button>
        </div>
      </div>
    </ScrollablePageWrapper>
  );
};

export default AccountPage;
