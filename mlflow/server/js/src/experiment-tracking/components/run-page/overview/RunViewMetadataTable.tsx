import { useDesignSystemTheme } from '@databricks/design-system';
import { ReactNode } from 'react';

export const RunViewMetadataTable = ({ children }: { children: ReactNode | ReactNode[] }) => {
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
