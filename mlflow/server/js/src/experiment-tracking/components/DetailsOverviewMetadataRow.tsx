import { useDesignSystemTheme } from '@databricks/design-system';

/**
 * Generic table row component for displaying metadata row in the details overview table (used in runs, logged models etc.)
 */
export const DetailsOverviewMetadataRow = ({ title, value }: { title: React.ReactNode; value: React.ReactNode }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <tr
      css={{
        display: 'flex',
        borderBottom: `1px solid ${theme.colors.borderDecorative}`,
        minHeight: theme.general.heightSm,
      }}
    >
      <th
        css={{
          flex: `0 0 240px`,
          backgroundColor: theme.colors.backgroundSecondary,
          color: theme.colors.textSecondary,
          padding: theme.spacing.sm,
          display: 'flex',
          alignItems: 'flex-start',
        }}
      >
        {title}
      </th>
      <td
        css={{
          flex: 1,
          padding: theme.spacing.sm,
          paddingTop: 0,
          paddingBottom: 0,
          display: 'flex',
          alignItems: 'center',
        }}
      >
        {value}
      </td>
    </tr>
  );
};
