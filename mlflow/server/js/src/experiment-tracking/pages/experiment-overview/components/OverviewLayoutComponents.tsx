import React from 'react';
import { TitleSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';

/**
 * Props for the StatCard component
 */
export interface StatCardProps {
  icon: React.ReactNode;
  iconColor: string;
  iconBgColor: string;
  value: string | number;
  label: React.ReactNode;
  isLoading?: boolean;
}

/**
 * Reusable stat card component for displaying metrics with an icon
 */
export const StatCard: React.FC<StatCardProps> = ({ icon, iconColor, iconBgColor, value, label, isLoading }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.md,
        padding: theme.spacing.lg,
        backgroundColor: theme.colors.backgroundPrimary,
        borderRadius: theme.borders.borderRadiusMd,
        border: `1px solid ${theme.colors.border}`,
        flex: 1,
        minWidth: 200,
      }}
    >
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: 40,
          height: 40,
          borderRadius: theme.borders.borderRadiusMd,
          backgroundColor: iconBgColor,
          color: iconColor,
          flexShrink: 0,
        }}
      >
        {icon}
      </div>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
        {isLoading ? (
          <TitleSkeleton css={{ width: '60%' }} />
        ) : (
          <Typography.Title level={2} css={{ margin: 0 }}>
            {value}
          </Typography.Title>
        )}
        <Typography.Text color="secondary" size="sm">
          {label}
        </Typography.Text>
      </div>
    </div>
  );
};

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
