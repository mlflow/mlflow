import { useDesignSystemTheme } from '@databricks/design-system';
import type { ReactNode } from 'react';

/**
 * Generic table component for displaying metadata in the details overview section (used in runs, logged models etc.)
 */
export const DetailsOverviewMetadataTable = ({ children }: { children: ReactNode | ReactNode[] }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <table
      css={{
        display: 'block',
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderBottom: 'none',
        borderRadius: theme.general.borderRadiusBase,
        width: '50%',
        minWidth: 640,
        marginBottom: theme.spacing.lg,
        overflow: 'hidden',
      }}
    >
      <tbody css={{ display: 'block' }}>{children}</tbody>
    </table>
  );
};
