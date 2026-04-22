import { useState } from 'react';
import {
  Button,
  Input,
  Typography,
  useDesignSystemTheme,
  Alert,
  Spinner,
  LegacyTable,
  Tag,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ScrollablePageWrapper } from '@mlflow/mlflow/src/common/components/ScrollablePageWrapper';
import { useUpdatePassword, useUserRolesQuery } from '../hooks';
import type { Role } from '../types';
import { isWorkspaceAdminRole } from '../types';

const AccountPage = () => {
  const { theme } = useDesignSystemTheme();
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const updatePassword = useUpdatePassword();

  const username =
    document.cookie
      .split('; ')
      .find((row) => row.startsWith('mlflow_user='))
      ?.substring('mlflow_user='.length) ?? '';

  const { data: rolesData, isLoading: rolesLoading } = useUserRolesQuery(username);

  const handleChangePassword = async () => {
    setError(null);
    setSuccessMessage(null);

    if (newPassword !== confirmPassword) {
      setError('New password and confirmation do not match');
      return;
    }

    if (!newPassword) {
      setError('Password cannot be empty');
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
    document.cookie = 'mlflow_user=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
    window.location.href = '/';
  };

  const roleColumns = [
    {
      title: 'Role',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Workspace',
      dataIndex: 'workspace',
      key: 'workspace',
    },
    {
      title: 'Admin',
      key: 'is_workspace_admin',
      render: (_: unknown, record: Role) =>
        isWorkspaceAdminRole(record) ? (
          <Tag componentId="account.role_admin_tag" color="indigo">
            Admin
          </Tag>
        ) : null,
    },
  ];

  return (
    <ScrollablePageWrapper>
      <div css={{ padding: theme.spacing.lg, maxWidth: 600 }}>
        <Typography.Title level={2} css={{ marginBottom: theme.spacing.lg }}>
          <FormattedMessage defaultMessage="Account" description="Account page title" />
        </Typography.Title>

        {!username && (
          <Alert
            componentId="account.no_user"
            type="warning"
            message="Not logged in"
            description="Could not determine the current user. Please log in again."
            css={{ marginBottom: theme.spacing.md }}
          />
        )}

        {username && (
          <Typography.Text css={{ marginBottom: theme.spacing.lg, display: 'block' }}>
            Logged in as <strong>{username}</strong>
          </Typography.Text>
        )}

        {error && (
          <Alert
            componentId="account.error"
            type="error"
            message={error}
            closable
            onClose={() => setError(null)}
            css={{ marginBottom: theme.spacing.md }}
          />
        )}
        {successMessage && (
          <Alert
            componentId="account.info"
            type="info"
            message={successMessage}
            closable
            onClose={() => setSuccessMessage(null)}
            css={{ marginBottom: theme.spacing.md }}
          />
        )}

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
            <Typography.Title level={4}>Change Password</Typography.Title>
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

          {username && (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
              <Typography.Title level={4}>My Roles</Typography.Title>
              {rolesLoading ? (
                <Spinner size="small" />
              ) : (
                <LegacyTable
                  dataSource={rolesData?.roles ?? []}
                  columns={roleColumns}
                  rowKey="id"
                  pagination={false}
                />
              )}
            </div>
          )}

          <div>
            <Button componentId="account.logout_button" type="tertiary" onClick={handleLogout} danger>
              <FormattedMessage defaultMessage="Logout" description="Button to logout" />
            </Button>
          </div>
        </div>
      </div>
    </ScrollablePageWrapper>
  );
};

export default AccountPage;
