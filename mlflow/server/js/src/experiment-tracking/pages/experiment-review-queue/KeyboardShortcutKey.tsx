import { useDesignSystemTheme } from '@databricks/design-system';

export const KeyboardShortcutKey = ({ children }: { children: string }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <kbd
      aria-hidden="true"
      css={{
        minWidth: 20,
        height: 20,
        padding: '0 var(--db-space-1)',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusSm,
        backgroundColor: theme.colors.backgroundSecondary,
        color: theme.colors.textSecondary,
        fontFamily: 'inherit',
        fontSize: 'var(--db-font-size-sm)',
        lineHeight: 1,
      }}
    >
      {children}
    </kbd>
  );
};
