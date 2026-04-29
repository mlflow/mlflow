import { useState } from 'react';
import {
  Alert,
  Button,
  Empty,
  Input,
  Modal,
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
import { performLogout } from '../auth-utils';
import { useCurrentUserQuery, useUpdatePassword, useUserRolesQuery } from '../hooks';
import { isWorkspaceAdminRole } from '../types';

const AccountPage = () => {
  const { theme } = useDesignSystemTheme();
  const queryClient = useQueryClient();
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);

  const updatePassword = useUpdatePassword();
  const { data: currentUserData, isLoading: currentUserLoading } = useCurrentUserQuery();
  const username = currentUserData?.user?.username ?? '';

  const [changePasswordOpen, setChangePasswordOpen] = useState(false);

  const closeChangePassword = () => {
    setChangePasswordOpen(false);
    setCurrentPassword('');
    setNewPassword('');
    setConfirmPassword('');
    setError(null);
  };

  const { data: rolesData, isLoading: rolesLoading, error: rolesError } = useUserRolesQuery(username);
  const roles = rolesData?.roles ?? [];

  const handleChangePassword = async () => {
    setError(null);

    if (!currentPassword) {
      setError('Current password is required');
      return;
    }

    if (!newPassword) {
      setError('Password cannot be empty');
      return;
    }

    if (newPassword !== confirmPassword) {
      setError('New password and confirmation do not match');
      return;
    }

    try {
      // Self-service password changes require current_password — the backend
      // re-asserts the existing password before applying the new one as a
      // defense-in-depth check (see ``update_user_password`` in
      // ``mlflow/server/auth/__init__.py``).
      await updatePassword.mutateAsync({
        username,
        password: newPassword,
        current_password: currentPassword,
      });
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      setChangePasswordOpen(false);
      // The browser keeps sending the OLD HTTP Basic Auth credentials until
      // it's forced through a fresh prompt, so subsequent API calls would
      // 401 even though the password change succeeded. Navigate to /logout
      // immediately — the logout page gives the user a clear "signed out
      // → sign back in" affordance and forces re-auth with the new
      // password, which is also the success signal in lieu of an inline
      // alert that they'd never see.
      window.location.assign(new URL('logout', window.location.href).toString());
    } catch (e: any) {
      setError(e.message || 'Failed to update password');
    }
  };

  const handleLogout = () => performLogout(queryClient);

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

        {!username && !currentUserLoading && (
          <Alert
            componentId="account.no_user"
            type="warning"
            message="Not logged in"
            description="Could not determine the current user. Please log in again."
          />
        )}

        {error && !changePasswordOpen && (
          // The change-password modal renders its own inline Alert for the
          // same ``error`` state, so showing it at the page level too would
          // double up while the modal is open.
          <Alert componentId="account.error" type="error" message={error} closable onClose={() => setError(null)} />
        )}

        <div>
          <Button
            componentId="account.change_password_button"
            type="primary"
            onClick={() => setChangePasswordOpen(true)}
            disabled={!username}
          >
            <FormattedMessage defaultMessage="Change password" description="Button to open the change password modal" />
          </Button>
        </div>
        <Modal
          componentId="account.change_password_modal"
          title="Change password"
          visible={changePasswordOpen}
          onCancel={closeChangePassword}
          onOk={handleChangePassword}
          okText="Update password"
          confirmLoading={updatePassword.isLoading}
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            {error && (
              <Alert
                componentId="account.change_password_modal.error"
                type="error"
                message={error}
                closable
                onClose={() => setError(null)}
              />
            )}
            <div>
              <Typography.Text bold>Current password</Typography.Text>
              <Input
                componentId="account.current_password"
                type="password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                placeholder="Enter current password"
              />
            </div>
            <div>
              <Typography.Text bold>New password</Typography.Text>
              <Input
                componentId="account.new_password"
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                placeholder="Enter new password"
              />
            </div>
            <div>
              <Typography.Text bold>Confirm password</Typography.Text>
              <Input
                componentId="account.confirm_password"
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Confirm new password"
              />
            </div>
          </div>
        </Modal>

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
            ) : rolesError ? (
              <Alert
                componentId="account.roles_error"
                type="error"
                message="Failed to load roles"
                description={(rolesError as Error)?.message || 'An error occurred while fetching your roles.'}
              />
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
                    <TableCell css={{ flex: 2 }}>
                      <div>{role.name}</div>
                      {role.permissions && role.permissions.length > 0 && (
                        <div
                          css={{
                            marginTop: theme.spacing.xs,
                            display: 'flex',
                            flexDirection: 'column',
                            gap: theme.spacing.xs / 2,
                          }}
                        >
                          {role.permissions.map((p) => (
                            <Typography.Text key={p.id} size="sm" color="secondary">
                              <code>
                                {p.resource_type}:{p.resource_pattern}
                              </code>{' '}
                              → <strong>{p.permission}</strong>
                            </Typography.Text>
                          ))}
                        </div>
                      )}
                    </TableCell>
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

        {/* Hide Logout when there's no authenticated user — auth is
            disabled or ``/users/current`` failed. ``/logout`` is only
            registered by the basic-auth app, so the click would 404. */}
        {username && (
          <div>
            <Button componentId="account.logout_button" onClick={handleLogout}>
              <FormattedMessage defaultMessage="Logout" description="Button to logout" />
            </Button>
          </div>
        )}
      </div>
    </ScrollablePageWrapper>
  );
};

export default AccountPage;
