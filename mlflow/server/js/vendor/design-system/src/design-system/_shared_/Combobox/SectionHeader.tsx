import { useDesignSystemTheme } from '../../Hooks';

export const SectionHeader = ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      {...props}
      css={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'flex-start',
        padding: `${theme.spacing.xs}px ${theme.spacing.lg / 2}px`,
        alignSelf: 'stretch',
        fontWeight: 400,
        color: theme.colors.textSecondary,
      }}
    >
      {children}
    </div>
  );
};
