import type { ReactNode } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';

export const IssueDetailColumn = ({
  issueKey,
  header,
  testId,
  bodyMaxWidth = 860,
  children,
}: {
  issueKey: string;
  /** Pinned full width above the scrolling body, with a divider under it. */
  header?: ReactNode;
  testId?: string;
  /** The widest the body's content grows, in pixels, before it is centred in the column. */
  bodyMaxWidth?: number;
  children: ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flex: 1,
        flexDirection: 'column',
        minWidth: 0,
        minHeight: 0,
        overflow: 'hidden',
        borderLeft: `1px solid ${theme.colors.border}`,
      }}
      data-testid={testId}
    >
      {header && (
        <div
          key={`${issueKey}-header`}
          css={{
            flexShrink: 0,
            padding: `${theme.spacing.md}px ${theme.spacing.lg}px`,
            borderBottom: `1px solid ${theme.colors.border}`,
          }}
        >
          {header}
        </div>
      )}
      {/* Keep long descriptions scrollable without moving the header. A new issue resets the scroll position. */}
      <div key={issueKey} css={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md + theme.spacing.xs / 2,
            padding: `${theme.spacing.md + theme.spacing.xs}px ${theme.spacing.xl}px`,
            width: '100%',
            maxWidth: bodyMaxWidth,
            marginLeft: 'auto',
            marginRight: 'auto',
          }}
        >
          {children}
        </div>
      </div>
    </div>
  );
};
