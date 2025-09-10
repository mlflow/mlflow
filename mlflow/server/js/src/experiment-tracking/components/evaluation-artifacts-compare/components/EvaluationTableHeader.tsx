import { useDesignSystemTheme } from '@databricks/design-system';
import type { PropsWithChildren, ReactNode } from 'react';

interface Props {
  className?: string;
  groupHeaderContent?: ReactNode;
  isGroupByHeader?: false;
}

export const EvaluationTableHeader = ({ children, className, groupHeaderContent = null }: PropsWithChildren<Props>) => {
  const { theme } = useDesignSystemTheme();

  return (
    // Header cell wrapper element
    <div
      css={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Spacer element serving as a group header */}
      <div
        css={{
          width: '100%',
          flexBasis: 40,
          display: 'flex',
          alignItems: 'center',
          padding: theme.spacing.sm,
          borderBottom: `1px solid ${theme.colors.borderDecorative}`,
        }}
        className="header-group-cell"
      >
        {groupHeaderContent}
      </div>
      {/* Main header cell content */}
      <div
        css={{
          width: '100%',
          flex: 1,
          display: 'flex',
          justifyContent: 'flex-start',
          alignItems: 'flex-start',
          padding: theme.spacing.xs,
          borderRight: `1px solid ${theme.colors.borderDecorative}`,
        }}
        className={className}
      >
        {children}
      </div>
    </div>
  );
};
