import type { ReactNode } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';

export interface LongFormLayoutProps {
  /** Main form content */
  children: ReactNode;
  /** Optional summary sidebar content - hidden on narrow screens */
  sidebar?: ReactNode;
  /** Maximum width of the main form area (default: 900) */
  formMaxWidth?: number;
  /** Minimum width of the sidebar (default: 200) */
  sidebarMinWidth?: number;
  /** Maximum width of the sidebar (default: 360) */
  sidebarMaxWidth?: number;
  /** Breakpoint below which sidebar is hidden (default: 1100) */
  sidebarHideBreakpoint?: number;
}

/**
 * A layout component for long forms with an optional summary sidebar.
 * The sidebar is flexible and hides on narrow screens to avoid clipping the form.
 */
export function LongFormLayout({
  children,
  sidebar,
  formMaxWidth = 900,
  sidebarMinWidth = 200,
  sidebarMaxWidth = 360,
  sidebarHideBreakpoint = 1100,
}: LongFormLayoutProps) {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        flex: 1,
        display: 'flex',
        gap: theme.spacing.lg,
        padding: `0 ${theme.spacing.md}px`,
        overflow: 'auto',
      }}
    >
      {/* Main form column */}
      <div
        css={{
          flex: '1 1 auto',
          maxWidth: formMaxWidth,
          minWidth: 0,
        }}
      >
        {children}
      </div>

      {/* Summary sidebar - flexible width, hidden on narrow screens */}
      {sidebar && (
        <div
          css={{
            flexShrink: 1,
            flexGrow: 0,
            minWidth: sidebarMinWidth,
            maxWidth: sidebarMaxWidth,
            marginLeft: 'auto',
            position: 'sticky',
            top: 0,
            alignSelf: 'flex-start',
            [`@media (max-width: ${sidebarHideBreakpoint}px)`]: {
              display: 'none',
            },
          }}
        >
          {sidebar}
        </div>
      )}
    </div>
  );
}
