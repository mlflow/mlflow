/**
 * Floating dev toolbar for switching the mock user.
 * Only renders when USE_MOCK_API is true (checked by the parent).
 *
 * Usage in integration tests:
 *   window.__mockAdmin.switchUser('alice');
 *   // then invalidate queries or reload the page
 */
import { useCallback, useState } from 'react';
import { Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { mockState } from './mockApi';

export const DevUserSwitcher = () => {
  const { theme } = useDesignSystemTheme();
  const queryClient = useQueryClient();
  const [, forceUpdate] = useState(0);

  const currentUser = mockState.getCurrentUser();
  const allUsers = mockState.getUsers();

  const handleSwitch = useCallback(
    (username: string) => {
      mockState.switchUser(username);
      // Invalidate all queries so the UI refetches with the new user context
      queryClient.invalidateQueries();
      forceUpdate((n) => n + 1);
    },
    [queryClient],
  );

  return (
    <div
      css={{
        position: 'fixed',
        bottom: theme.spacing.md,
        right: theme.spacing.md,
        zIndex: 9999,
        background: theme.colors.backgroundPrimary,
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: theme.spacing.sm,
        boxShadow: theme.general.shadowLow,
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.xs,
        minWidth: 180,
      }}
    >
      <Typography.Text bold size="sm" css={{ marginBottom: theme.spacing.xs }}>
        Mock User Switcher
      </Typography.Text>
      {allUsers.map((user) => {
        const isActive = user.username === currentUser?.username;
        return (
          <div
            key={user.username}
            role="button"
            tabIndex={0}
            onClick={() => handleSwitch(user.username)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') handleSwitch(user.username);
            }}
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.xs,
              padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
              borderRadius: theme.borders.borderRadiusMd - 2,
              cursor: 'pointer',
              background: isActive ? theme.colors.actionPrimaryBackgroundDefault : 'transparent',
              color: isActive ? theme.colors.actionPrimaryTextDefault : theme.colors.textPrimary,
              '&:hover': {
                background: isActive
                  ? theme.colors.actionPrimaryBackgroundHover
                  : theme.colors.actionDefaultBackgroundHover,
              },
            }}
          >
            <Typography.Text size="sm" css={{ color: 'inherit' }}>
              {user.username}
            </Typography.Text>
            {user.is_admin && (
              <Tag componentId="dev.user_switcher.admin_tag" color="indigo" css={{ marginLeft: 'auto' }}>
                admin
              </Tag>
            )}
          </div>
        );
      })}
    </div>
  );
};
