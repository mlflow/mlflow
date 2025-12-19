import { useDesignSystemTheme } from '@databricks/design-system';

export const NullCell = ({ isComparing }: { isComparing?: boolean }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <span
      css={{
        height: '20px',
      }}
    >
      {isComparing && <span css={{ fontStyle: 'italic', color: theme.colors.textSecondary }}>null</span>}
    </span>
  );
};
