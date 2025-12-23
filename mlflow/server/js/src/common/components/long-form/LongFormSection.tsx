import type { ReactNode } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';

export interface LongFormSectionProps {
  /** Section title displayed on the left side */
  title: string;
  /** Width of the title column in pixels (default: 200) */
  titleWidth?: number;
  /** Content to display on the right side */
  children: ReactNode;
  /** Whether to hide the bottom border divider (default: false) */
  hideDivider?: boolean;
}

/**
 * A form section with a two-column layout: title/label on the left, content on the right.
 * By default, sections are separated by a horizontal divider at the bottom.
 * Responsive: stacks vertically on narrow screens.
 */
export const LongFormSection = ({ title, titleWidth = 200, children, hideDivider = false }: LongFormSectionProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
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
        <span
          css={{
            fontWeight: theme.typography.typographyBoldFontWeight,
            fontSize: theme.typography.fontSizeLg,
          }}
        >
          {title}
        </span>
      </div>
      <div css={{ flexGrow: 1, minWidth: 0 }}>{children}</div>
    </div>
  );
};
