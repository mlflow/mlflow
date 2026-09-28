import { useDesignSystemTheme } from '@databricks/design-system';

export function ModelTraceExplorerBadge({
  count,
  variant = 'danger',
}: {
  count: number;
  variant?: 'danger' | 'neutral';
}) {
  const { theme } = useDesignSystemTheme();
  const isNeutral = variant === 'neutral';

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: theme.typography.fontSizeBase,
        height: theme.typography.fontSizeBase,
        backgroundColor: isNeutral
          ? theme.colors.backgroundSecondary
          : theme.colors.actionDangerPrimaryBackgroundDefault,
        padding: theme.spacing.xs,
        marginLeft: theme.spacing.xs,
        boxSizing: 'border-box',
      }}
    >
      <span
        css={{ color: isNeutral ? theme.colors.textSecondary : theme.colors.actionPrimaryTextDefault, fontSize: 11 }}
      >
        {count}
      </span>
    </div>
  );
}
