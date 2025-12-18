import type { ReactNode } from 'react';
import { useDesignSystemTheme, Typography } from '@databricks/design-system';

export interface LongFormSummaryProps {
  /** Title displayed at the top of the summary panel */
  title: string;
  /** Content to display in the summary */
  children: ReactNode;
}

/**
 * A summary sidebar panel for long forms.
 * Typically used as a sticky sidebar to show form state summary.
 */
export function LongFormSummary({ title, children }: LongFormSummaryProps) {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.general.borderRadiusBase,
        padding: theme.spacing.md,
      }}
    >
      <Typography.Title level={3} css={{ marginBottom: theme.spacing.md }}>
        {title}
      </Typography.Title>
      {children}
    </div>
  );
}
