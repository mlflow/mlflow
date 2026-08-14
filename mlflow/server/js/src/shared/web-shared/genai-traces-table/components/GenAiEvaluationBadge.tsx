import { useDesignSystemTheme } from '@databricks/design-system';

export const GenAiEvaluationBadge = ({
  backgroundColor,
  icon,
  children,
}:
  | {
      backgroundColor?: string;
      icon?: React.ReactNode;
      children: React.ReactNode;
    }
  | {
      backgroundColor?: string;
      icon: React.ReactNode;
      children?: null;
    }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        padding: '4px 8px',
        gap: theme.spacing.xs,
        borderRadius: theme.borders.borderRadiusMd,
        color: theme.colors.textSecondary,
        backgroundColor: backgroundColor || theme.colors.backgroundSecondary,
        fontSize: theme.typography.fontSizeSm,
      }}
    >
      {icon ? icon : null}
      {children ? <span>{children}</span> : null}
    </div>
  );
};
