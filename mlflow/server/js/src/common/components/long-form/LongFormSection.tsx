import { useState, type ReactNode } from 'react';
import { ChevronDownIcon, ChevronRightIcon, useDesignSystemTheme } from '@databricks/design-system';

// React 18.x exposes ``useId`` at runtime, but ``@types/react`` is pinned at
// 17.x in this repo so the named import doesn't type-check. ``import * as
// React from 'react'`` would type-check but is blocked by the
// ``no-restricted-imports`` rule (``Suspense`` must come from
// ``@databricks/web-shared/react``). Pull the symbol via ``require`` so the
// type system sees only the narrow shape we use, and fall back to a
// ``useState``-based counter for the case where ``react-17`` (aliased in
// devDependencies for Enzyme) is the active runtime — keeps the section
// from blowing up at the call site instead of at the import.
// eslint-disable-next-line @typescript-eslint/no-var-requires, @typescript-eslint/no-require-imports
const reactRuntime = require('react') as { useId?: () => string };

let fallbackIdCounter = 0;
const useFallbackId = (): string => useState(() => `long-form-section-fallback-${++fallbackIdCounter}`)[0];

// One stable function for the lifetime of the module — Rules-of-Hooks
// requires the same hook to be called in the same order each render, so the
// React 18 vs. 17 selection happens once at module load.
const useStableId: () => string = reactRuntime.useId ?? useFallbackId;

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
  // Stable id wires the toggle button's ``aria-controls`` to the content
  // region's ``id`` so screen readers announce the disclosure relationship.
  const contentId = useStableId();

  // ``<span>`` (not ``<div>``) so this fragment is valid phrasing content
  // when nested inside the ``<button>`` in the collapsible branch — avoids
  // React's ``validateDOMNesting`` warning ("<div> cannot appear as a
  // descendant of <button>").
  const titleBlock = (
    <>
      <span
        css={{
          display: 'flex',
          fontWeight: theme.typography.typographyBoldFontWeight,
          fontSize: theme.typography.fontSizeLg,
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
      </span>
      {subtitle && (
        <span
          css={{
            display: 'block',
            marginTop: theme.spacing.xs,
            fontSize: theme.typography.fontSizeSm,
            color: theme.colors.textSecondary,
          }}
        >
          {subtitle}
        </span>
      )}
    </>
  );

  // Collapsed sections shrink their vertical padding so the modal actually
  // becomes shorter — otherwise the row keeps its open-state footprint and
  // a "compact" form still scrolls.
  const verticalPadding = collapsed ? theme.spacing.sm : theme.spacing.lg;

  return (
    <div
      className={className}
      css={{
        display: 'flex',
        gap: 32,
        paddingTop: verticalPadding,
        paddingBottom: verticalPadding,
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
            aria-controls={contentId}
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
      {/* ``hidden`` (vs. conditional render) keeps the section mounted so any
          in-progress draft state inside ``children`` (e.g. ``RolePermissionsSection``'s
          staged grant, ``RoleUsersSection``'s search text) survives a collapse / expand
          round-trip. ``inert`` removes the subtree from tab order and the
          accessibility tree — belt-and-suspenders alongside ``hidden`` since
          some CSS resets override ``[hidden] { display: none }``. */}
      <div
        id={contentId}
        hidden={collapsed}
        // React 18 accepts ``inert`` as a string-only DOM attribute; the
        // empty string ``""`` is the canonical "on" value per the HTML spec.
        // ``HTMLAttributes`` didn't add ``inert`` until React 19, so spread
        // the attribute through a narrowly-typed cast.
        // Always pass the key (with ``undefined`` when not collapsed) so
        // React strips the attribute from the DOM on re-render — omitting
        // the key from the spread would leave the previous attribute in
        // place.
        {...({ inert: collapsed ? '' : undefined } as { inert?: '' })}
        css={{ flexGrow: 1, minWidth: 0 }}
      >
        {children}
      </div>
    </div>
  );
};
