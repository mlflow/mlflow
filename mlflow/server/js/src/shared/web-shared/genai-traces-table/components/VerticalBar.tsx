import { useDesignSystemTheme } from '@databricks/design-system';

export const VerticalBar = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        width: '1.5px',
        backgroundColor: theme.colors.border,
      }}
    />
  );
};
