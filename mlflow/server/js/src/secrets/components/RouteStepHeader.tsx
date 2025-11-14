import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { ReactNode } from 'react';

interface RouteStepHeaderProps {
  stepNumber: number;
  title: ReactNode;
  actions?: ReactNode;
}

export const RouteStepHeader = ({ stepNumber, title, actions }: RouteStepHeaderProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: theme.spacing.md,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 28,
            height: 28,
            borderRadius: '50%',
            backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
            color: theme.colors.actionPrimaryTextDefault,
            fontSize: 14,
            fontWeight: 600,
          }}
        >
          {stepNumber}
        </div>
        <Typography.Title level={4} css={{ marginBottom: 0, color: theme.colors.textPrimary }}>
          {title}
        </Typography.Title>
      </div>
      {actions && <div>{actions}</div>}
    </div>
  );
};
