import type { ReactNode } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';

export interface ExperimentPageMainContentProps {
  breadcrumbs: ReactNode;
  pageActions?: ReactNode;
  children: ReactNode;
}

/**
 * Keeps every revamped experiment route on the same page grid. The asymmetric
 * padding preserves the established datasets-v2 and traces-v4 content position:
 * PageWrapper supplies the remaining horizontal viewport gutter.
 */
export const ExperimentPageMainContent = ({ breadcrumbs, pageActions, children }: ExperimentPageMainContentProps) => {
  const { theme } = useDesignSystemTheme();
  const verticalOffset = theme.spacing.xs + theme.spacing.xs / 4;
  const alignedTopPaddingAndContentGap = theme.spacing.lg - verticalOffset;

  return (
    <main
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minWidth: 0,
        minHeight: 0,
        overflow: 'hidden',
        gap: alignedTopPaddingAndContentGap,
        paddingTop: alignedTopPaddingAndContentGap,
        paddingRight: theme.spacing.sm,
        paddingBottom: alignedTopPaddingAndContentGap,
        paddingLeft: theme.spacing.lg,
        '& [data-testid="PageLoading"] > div': {
          paddingInline: 0,
        },
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
        <div css={{ flex: 1, minWidth: 0 }}>{breadcrumbs}</div>
        {pageActions}
      </div>
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0, minHeight: 0 }}>{children}</div>
    </main>
  );
};
