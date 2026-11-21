/** @jsxImportSource @emotion/react */
import type { ReactNode } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';

export interface ExperimentPageMainContentProps {
  breadcrumbs?: ReactNode;
  pageActions?: ReactNode;
  // Rendered under the breadcrumb row, e.g. the experiment description.
  headerDescription?: ReactNode;
  revamped?: boolean;
  children: ReactNode;
}

/**
 * Keeps routed content mounted when experiment metadata switches layouts.
 * Classic mode uses uniform padding without headers or an extra main landmark.
 * Revamped mode preserves the page grid, with PageWrapper supplying the viewport gutter.
 */
export const ExperimentPageMainContent = ({
  breadcrumbs,
  pageActions,
  headerDescription,
  revamped = true,
  children,
}: ExperimentPageMainContentProps) => {
  const { theme } = useDesignSystemTheme();
  const verticalOffset = theme.spacing.xs + theme.spacing.xs / 4;
  const alignedTopPaddingAndContentGap = theme.spacing.lg - verticalOffset;

  return (
    <div
      role={revamped ? 'main' : undefined}
      css={[
        { display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0, minHeight: 0 },
        revamped
          ? {
              overflow: 'hidden',
              gap: alignedTopPaddingAndContentGap,
              paddingTop: alignedTopPaddingAndContentGap,
              paddingRight: theme.spacing.sm,
              paddingBottom: alignedTopPaddingAndContentGap,
              paddingLeft: theme.spacing.lg,
              '& [data-testid="PageLoading"] > div': {
                paddingInline: 0,
              },
            }
          : { padding: theme.spacing.sm },
      ]}
    >
      {revamped && (breadcrumbs || pageActions) && (
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
          <div css={{ flex: 1, minWidth: 0 }}>{breadcrumbs}</div>
          {pageActions}
        </div>
      )}
      {revamped && headerDescription}
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0, minHeight: 0 }}>{children}</div>
    </div>
  );
};
