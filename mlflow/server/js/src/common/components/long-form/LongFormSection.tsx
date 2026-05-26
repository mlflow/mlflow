import { useState, type ReactNode } from 'react';
import { ChevronDownIcon, ChevronRightIcon, useDesignSystemTheme } from '@databricks/design-system';

export interface LongFormSectionProps {
  /** Section title displayed on the left side */
  title: string;
  /**
   * Optional subtitle displayed under the title (e.g. "Optional",
   * "Required", or a short qualifier). Rendered in muted secondary text.
   */
  subtitle?: string;
  /** Width of the title column in pixels (default: 200) */
  titleWidth?: number;
  /** Content to display on the right side */
  children: ReactNode;
  /** Whether to hide the bottom border divider (default: false) */
  hideDivider?: boolean;
  /** Optional className for custom styling */
  className?: string;
  /**
   * When true, the title becomes a button that toggles the section open
   * and closed. Use for optional sections so dense modals don't surface
   * every field on the user at once.
   */
  collapsible?: boolean;
  /** Initial collapsed state when ``collapsible`` is true (default: false). */
  defaultCollapsed?: boolean;
}

/**
 * A form section with a two-column layout: title/label on the left, content on the right.
 * By default, sections are separated by a horizontal divider at the bottom.
 * Responsive: stacks vertically on narrow screens.
 */
export const LongFormSection = ({
  title,
  subtitle,
  titleWidth = 200,
  children,
  hideDivider = false,
  className,
  collapsible = false,
  defaultCollapsed = false,
}: LongFormSectionProps) => {
  const { theme } = useDesignSystemTheme();
  const [collapsed, setCollapsed] = useState(collapsible && defaultCollapsed);

  const titleBlock = (
    <>
      <div
        css={{
          fontWeight: theme.typography.typographyBoldFontWeight,
          fontSize: theme.typography.fontSizeLg,
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
        }}
      >
        {title}
        {collapsible && (
          <span css={{ display: 'inline-flex', alignItems: 'center', color: theme.colors.textSecondary }}>
            {collapsed ? <ChevronRightIcon /> : <ChevronDownIcon />}
          </span>
        )}
      </div>
      {subtitle && (
        <div
          css={{
            marginTop: theme.spacing.xs,
            fontSize: theme.typography.fontSizeSm,
            color: theme.colors.textSecondary,
          }}
        >
          {subtitle}
        </div>
      )}
    </>
  );

  return (
    <div
      className={className}
      css={{
        display: 'flex',
        gap: 32,
        paddingTop: theme.spacing.lg,
        paddingBottom: theme.spacing.lg,
        borderBottom: hideDivider ? 'none' : `1px solid ${theme.colors.borderDecorative}`,
        '@media (max-width: 1023px)': {
          flexDirection: 'column',
          gap: theme.spacing.md,
        },
      }}
    >
      <div
        css={{
          flexShrink: 0,
          width: titleWidth,
          '@media (max-width: 1023px)': {
            width: '100%',
          },
        }}
      >
        {collapsible ? (
          <button
            type="button"
            onClick={() => setCollapsed((c) => !c)}
            aria-expanded={!collapsed}
            css={{
              background: 'none',
              border: 'none',
              padding: 0,
              cursor: 'pointer',
              font: 'inherit',
              color: 'inherit',
              textAlign: 'left',
              width: '100%',
            }}
          >
            {titleBlock}
          </button>
        ) : (
          titleBlock
        )}
      </div>
      {!collapsed && <div css={{ flexGrow: 1, minWidth: 0 }}>{children}</div>}
    </div>
  );
};
