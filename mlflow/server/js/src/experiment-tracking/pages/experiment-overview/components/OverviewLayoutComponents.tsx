import React from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';

/**
 * Reusable container for tab content with consistent styling
 */
export const TabContentContainer: React.FC<{ children?: React.ReactNode }> = ({ children }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.lg,
        padding: `${theme.spacing.sm}px 0`,
      }}
    >
      {children}
    </div>
  );
};

/**
 * Reusable grid layout for side-by-side charts
 */
export const ChartGrid: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
        gap: theme.spacing.lg,
      }}
    >
      {children}
    </div>
  );
};
