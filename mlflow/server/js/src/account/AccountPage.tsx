import { useMemo, useState } from 'react';
import {
  Alert,
  Avatar,
  Button,
  Empty,
  Input,
  Modal,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Tabs,
  Tag,
  Typography,
  UserIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useWorkspacesEnabled } from '../experiment-tracking/hooks/useServerInfo';
import { useSearchParams } from '../common/utils/RoutingUtils';
import { performLogout } from './auth-utils';
import {
  useCurrentUserQuery,
  useIsBasicAuth,
  useMyPermissionsQuery,
  useUpdatePassword,
  useUserRolesQuery,
} from './hooks';
import { PermissionsSection } from './PermissionsSection';
import { DEFAULT_WORKSPACE_NAME, isWorkspaceAdminRole } from './types';
import type { Role } from './types';

/**
 * One row of the Roles tab on the Account page. Renders the role name,
 * its workspace (when ``workspacesEnabled``), and a "Manager" / "Admin"
 * tag for workspace-admin roles.
 */
const AccountRoleRow = ({ role, workspacesEnabled }: { role: Role; workspacesEnabled: boolean }) => (
  <TableRow>
    <TableCell css={{ flex: 2 }}>{role.name}</TableCell>
    {workspacesEnabled && <TableCell css={{ flex: 1 }}>{role.workspace}</TableCell>}
    <TableCell css={{ flex: 1 }}>
      {isWorkspaceAdminRole(role) ? (
        <Tag componentId="account.role_admin_tag" color="indigo">
          {workspacesEnabled ? (
            <FormattedMessage
              defaultMessage="Manager"
              description="Tag content marking a workspace-admin role (multi-tenant)"
            />
          ) : (
            <FormattedMessage defaultMessage="Admin" description="Tag content marking an admin role (single-tenant)" />
          )}
        </Tag>
      ) : null}
    </TableCell>
  </TableRow>
);

const AccountPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const tabFromUrl = searchParams.get('tab');
  const activeTab = tabFromUrl === 'permissions' ? 'permissions' : 'roles';
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);

  const updatePassword = useUpdatePassword();
  const { data: currentUserData, isLoading: currentUserLoading } = useCurrentUserQuery();
  const username = currentUserData?.user?.username ?? '';
  const { workspacesEnabled } = useWorkspacesEnabled();
  // Change-password drives the same Basic-Auth realm-cache flow as logout,
  // so hide the affordance under custom authorization_function plugins.
  const isBasicAuth = useIsBasicAuth();

  const [changePasswordOpen, setChangePasswordOpen] = useState(false);

  const closeChangePassword = () => {
    setChangePasswordOpen(false);
    setCurrentPassword('');
    setNewPassword('');
    setConfirmPassword('');
    setError(null);
  };

  const { data: rolesData, isLoading: rolesLoading, error: rolesError } = useUserRolesQuery(username);
  const allRoles = useMemo(() => rolesData?.roles ?? [], [rolesData]);

  const { data: directPermsData, isLoading: directPermsLoading, error: directPermsError } = useMyPermissionsQuery();
  const allDirectPermissions = useMemo(() => directPermsData?.permissions ?? [], [directPermsData]);

  // Single-tenant mode hides the Workspace column, so stale non-default
  // rows would look like duplicates. Filter them out; null workspace
  // (deleted resource) stays - could belong to any workspace.
  const roles = useMemo(
    () => (workspacesEnabled ? allRoles : allRoles.filter((r) => r.workspace === DEFAULT_WORKSPACE_NAME)),
    [allRoles, workspacesEnabled],
  );
  const directPermissions = useMemo(
    () =>
      workspacesEnabled
        ? allDirectPermissions
        : allDirectPermissions.filter((p) => p.workspace == null || p.workspace === DEFAULT_WORKSPACE_NAME),
    [allDirectPermissions, workspacesEnabled],
  );

  const handleChangePassword = async () => {
    setError(null);

    if (!currentPassword) {
      setError(
        intl.formatMessage({
          defaultMessage: 'Current password is required',
          description: 'Validation error when the current-password field is empty',
        }),
      );
      return;
    }

    if (!newPassword) {
      setError(
        intl.formatMessage({
          defaultMessage: 'Password cannot be empty',
          description: 'Validation error when the new-password field is empty',
        }),
      );
      return;
    }

    if (newPassword !== confirmPassword) {
      setError(
        intl.formatMessage({
          defaultMessage: 'New password and confirmation do not match',
          description: 'Validation error when new password and confirmation do not match',
        }),
      );
      return;
    }

    try {
      // Backend re-asserts current_password before applying the new one
      // (defense in depth - see ``update_user_password``).
      await updatePassword.mutateAsync({
        username,
        password: newPassword,
        current_password: currentPassword,
      });
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      setChangePasswordOpen(false);
      // Browser still has the old Basic Auth creds cached, so subsequent
      // API calls would 401. Bounce home to trigger a fresh prompt - also
      // serves as the success signal since there's no inline alert.
      performLogout(queryClient);
    } catch (e: unknown) {
      setError(
        e instanceof Error
          ? e.message
          : intl.formatMessage({
              defaultMessage: 'Failed to update password',
              description: 'Generic error when password update fails for an unknown reason',
            }),
      );
    }
  };

  const rolesEmptyState =
    roles.length === 0 ? (
      <Empty
        title={intl.formatMessage({
          defaultMessage: 'No roles',
          description: 'Empty-state title for the roles table on the account page',
        })}
        description={intl.formatMessage({
          defaultMessage: 'You have not been assigned to any roles.',
          description: 'Empty-state description for the roles table on the account page',
        })}
      />
    ) : null;

  return (
    <ScrollablePageWrapper>
      <div css={{ padding: theme.spacing.md, display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
            <Avatar
              type="entity"
              size="lg"
              icon={<UserIcon />}
              label={intl.formatMessage({
                defaultMessage: 'Account',
                description: 'Accessible label for the account-page icon avatar',
              })}
            />
            <Typography.Title withoutMargins level={2}>
              <FormattedMessage defaultMessage="Account" description="Account page title" />
            </Typography.Title>
          </div>
          {username && (
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Logged in as <strong>{username}</strong>"
                description="Subtitle showing the currently authenticated username"
                values={{
                  username,
                  strong: (chunks) => <strong>{chunks}</strong>,
                }}
              />
            </Typography.Text>
          )}
        </div>

        {!username && !currentUserLoading && (
          <Alert
            componentId="account.no_user"
            type="warning"
            message={intl.formatMessage({
              defaultMessage: 'Not logged in',
              description: 'Alert title shown when the current user cannot be determined',
            })}
            description={intl.formatMessage({
              defaultMessage: 'Could not determine the current user. Please log in again.',
              description: 'Alert description prompting the user to log in again',
            })}
          />
        )}

        {error && !changePasswordOpen && (
          // Modal renders its own Alert for ``error`` - don't double up.
          <Alert componentId="account.error" type="error" message={error} closable onClose={() => setError(null)} />
        )}

        {isBasicAuth && (
          <div>
            <Button
              componentId="account.change_password_button"
              type="primary"
              onClick={() => setChangePasswordOpen(true)}
              disabled={!username}
            >
              <FormattedMessage
                defaultMessage="Change password"
                description="Button to open the change password modal"
              />
            </Button>
          </div>
        )}
        <Modal
          componentId="account.change_password_modal"
          title={intl.formatMessage({
            defaultMessage: 'Change password',
            description: 'Title of the change-password modal',
          })}
          visible={changePasswordOpen}
          onCancel={closeChangePassword}
          onOk={handleChangePassword}
          okText={intl.formatMessage({
            defaultMessage: 'Update password',
            description: 'Confirm button label in the change-password modal',
          })}
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
              <Typography.Text bold>
                <FormattedMessage
                  defaultMessage="Current password"
                  description="Label for the current-password field"
                />
              </Typography.Text>
              <Input
                componentId="account.current_password"
                type="password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                placeholder={intl.formatMessage({
                  defaultMessage: 'Enter current password',
                  description: 'Placeholder for the current-password input',
                })}
              />
            </div>
            <div>
              <Typography.Text bold>
                <FormattedMessage defaultMessage="New password" description="Label for the new-password field" />
              </Typography.Text>
              <Input
                componentId="account.new_password"
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                placeholder={intl.formatMessage({
                  defaultMessage: 'Enter new password',
                  description: 'Placeholder for the new-password input',
                })}
              />
            </div>
            <div>
              <Typography.Text bold>
                <FormattedMessage
                  defaultMessage="Confirm password"
                  description="Label for the confirm-password field"
                />
              </Typography.Text>
              <Input
                componentId="account.confirm_password"
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder={intl.formatMessage({
                  defaultMessage: 'Confirm new password',
                  description: 'Placeholder for the confirm-password input',
                })}
              />
            </div>
          </div>
        </Modal>

        {username && (
          <Tabs.Root
            componentId="account.tabs"
            valueHasNoPii
            value={activeTab}
            onValueChange={(value) => {
              const next = new URLSearchParams(searchParams);
              if (value === 'roles') {
                next.delete('tab');
              } else {
                next.set('tab', value);
              }
              setSearchParams(next, { replace: true });
            }}
          >
            <Tabs.List>
              <Tabs.Trigger value="roles">
                <FormattedMessage defaultMessage="Roles" description="Tab trigger for the user's roles" />
              </Tabs.Trigger>
              <Tabs.Trigger value="permissions">
                <FormattedMessage defaultMessage="Permissions" description="Tab trigger for the user's permissions" />
              </Tabs.Trigger>
            </Tabs.List>
            <Tabs.Content value="roles" css={{ paddingTop: theme.spacing.md }}>
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
                  message={intl.formatMessage({
                    defaultMessage: 'Failed to load roles',
                    description: 'Alert title shown when the roles query fails on the account page',
                  })}
                  description={
                    (rolesError as Error)?.message ||
                    intl.formatMessage({
                      defaultMessage: 'An error occurred while fetching your roles.',
                      description: 'Fallback description shown when the roles query has no error message',
                    })
                  }
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
                      <FormattedMessage
                        defaultMessage="Role"
                        description="Roles table column header for the role name"
                      />
                    </TableHeader>
                    {workspacesEnabled && (
                      <TableHeader componentId="account.roles.workspace_header" css={{ flex: 1 }}>
                        <FormattedMessage
                          defaultMessage="Workspace"
                          description="Roles table column header for the workspace"
                        />
                      </TableHeader>
                    )}
                    <TableHeader componentId="account.roles.admin_header" css={{ flex: 1 }}>
                      {workspacesEnabled ? (
                        <FormattedMessage
                          defaultMessage="Workspace Manager"
                          description="Roles table column header for the workspace-admin marker (multi-tenant)"
                        />
                      ) : (
                        <FormattedMessage
                          defaultMessage="Admin"
                          description="Roles table column header for the admin marker (single-tenant)"
                        />
                      )}
                    </TableHeader>
                  </TableRow>
                  {roles.map((role) => (
                    <AccountRoleRow key={role.id} role={role} workspacesEnabled={workspacesEnabled} />
                  ))}
                </Table>
              )}
            </Tabs.Content>
            <Tabs.Content
              value="permissions"
              css={{
                paddingTop: theme.spacing.md,
                display: 'flex',
                flexDirection: 'column',
                gap: theme.spacing.md,
              }}
            >
              <PermissionsSection
                roles={roles}
                directPermissions={directPermissions}
                isLoading={rolesLoading || directPermsLoading}
                rolesError={rolesError}
                directPermsError={directPermsError}
                componentId="account"
                workspacesEnabled={workspacesEnabled}
              />
            </Tabs.Content>
          </Tabs.Root>
        )}
      </div>
    </ScrollablePageWrapper>
  );
};

export default AccountPage;
