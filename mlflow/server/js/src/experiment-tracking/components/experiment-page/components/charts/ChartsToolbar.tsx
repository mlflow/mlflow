import { TableFilterLayout, useDesignSystemTheme } from '@databricks/design-system';
import { ChartsDateSelector } from './ChartsDateSelector';
import { ChartsFilters } from './ChartsFilters';

export const ChartsToolbar = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        width: '100%',
        alignItems: 'flex-end',
        gap: theme.spacing.sm,
        paddingBottom: theme.spacing.xs,
        borderBottom: `1px solid ${theme.colors.grey100}`,
      }}
    >
      <TableFilterLayout
        css={{
          marginBottom: 0,
          flex: 1,
        }}
      >
        <ChartsDateSelector />
        <ChartsFilters />
      </TableFilterLayout>
    </div>
  );
};
