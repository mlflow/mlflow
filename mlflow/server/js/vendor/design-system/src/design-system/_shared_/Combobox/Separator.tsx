import { useDesignSystemTheme } from '../../Hooks';

export const Separator = (props: React.HTMLAttributes<HTMLDivElement>) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      {...props}
      css={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        margin: `${theme.spacing.xs}px ${theme.spacing.lg / 2}px`,
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderBottom: 0,
        alignSelf: 'stretch',
      }}
    />
  );
};
