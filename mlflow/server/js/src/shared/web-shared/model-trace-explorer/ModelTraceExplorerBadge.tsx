import { useDesignSystemTheme } from '@databricks/design-system';

export function ModelTraceExplorerBadge({ count }: { count: number }) {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: theme.typography.fontSizeBase,
        height: theme.typography.fontSizeBase,
        backgroundColor: theme.colors.actionDangerPrimaryBackgroundDefault,
        padding: theme.spacing.xs,
        marginLeft: theme.spacing.xs,
        boxSizing: 'border-box',
      }}
    >
      <span css={{ color: theme.colors.actionPrimaryTextDefault, fontSize: 11 }}>{count}</span>
    </div>
  );
}
