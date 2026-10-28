import { useLayoutEffect, useRef, useState, type RefObject } from 'react';
import { Input, type InputRef, SearchIcon, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';
import { TRACES_TOOLBAR_COLLAPSE_QUERY, TRACES_TOOLBAR_CONTAINER_NAME } from './TracesToolbarResponsive';

// Module-local static analytics-id namespace (the no-dynamic-property-value lint rule requires
// static componentId values, so a runtime-injected prefix isn't possible).
const COMPONENT_ID = 'web-shared.traces-table';

export interface TracesTableToolbarProps {
  searchValue: string;
  onSearchChange: (next: string) => void;
  onSearchClear: () => void;
  searchInputRef?: RefObject<InputRef>;
  searchPlaceholder?: string;
  /** Node rendered inside the search input's `suffix` slot (e.g. an AI-filter toggle button). */
  searchSuffix?: React.ReactNode;
  /** Consumer-owned search experience, rendered in place of the standard input when provided. */
  searchControl?: React.ReactNode;
  /**
   * Let a consumer search containing filter pills fill the available toolbar width. The search moves
   * to its own row below the controls when `searchMinimumWidth` doesn't fit beside them.
   */
  expandSearch?: boolean;
  /** Intrinsic width needed for the consumer search, including its applied filter pills. */
  searchMinimumWidth?: number;
  /** Commit the current search (wired to the search `<Input>`'s `onPressEnter`); no-op when omitted. */
  onSearchSubmit?: () => void;
  /** Controls placed before the search box (e.g. a date-range selector). */
  leftControls?: React.ReactNode;
  /** Controls placed after the search box (e.g. filters, columns, a spacer, and trailing actions). */
  rightControls?: React.ReactNode;
}

/**
 * Single-row toolbar shell: `leftControls` slot, a built-in controlled search box (center), a
 * `rightControls` slot. Owns no product controls — the consumer injects everything domain-specific
 * through the two slots. The row has no data dependency, so it paints immediately on first render.
 */
export const TracesTableToolbar: React.FC<TracesTableToolbarProps> = ({
  searchValue,
  onSearchChange,
  onSearchClear,
  searchInputRef,
  searchPlaceholder,
  searchSuffix,
  searchControl,
  expandSearch,
  searchMinimumWidth = 340,
  onSearchSubmit,
  leftControls,
  rightControls,
}: TracesTableToolbarProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const toolbarRef = useRef<HTMLDivElement>(null);
  const leftControlsRef = useRef<HTMLDivElement>(null);
  const rightControlsRef = useRef<HTMLDivElement>(null);
  const [inlineSearchSpace, setInlineSearchSpace] = useState(Infinity);
  useLayoutEffect(() => {
    if (!expandSearch) return;
    const toolbar = toolbarRef.current;
    const left = leftControlsRef.current;
    const right = rightControlsRef.current;
    if (!toolbar || !left || !right) return;
    const measure = () => {
      if (!toolbar.clientWidth) return;
      setInlineSearchSpace(
        Math.max(0, toolbar.clientWidth - left.offsetWidth - right.offsetWidth - 2 * theme.spacing.sm),
      );
    };
    measure();
    const observer = new ResizeObserver(measure);
    [toolbar, left, right].forEach((element) => observer.observe(element));
    return () => observer.disconnect();
  }, [expandSearch, theme.spacing.sm]);
  const searchOnNewRow = expandSearch && searchMinimumWidth > inlineSearchSpace;

  const placeholder =
    searchPlaceholder ??
    intl.formatMessage({
      defaultMessage: 'Search traces by id, input, or output',
      description: 'Placeholder for the search input on the traces table toolbar',
    });
  const searchInput = (
    <Input
      ref={searchInputRef}
      componentId={`${COMPONENT_ID}.search`}
      prefix={<SearchIcon />}
      allowClear
      value={searchValue}
      placeholder={placeholder}
      aria-label={placeholder}
      onChange={(e) => onSearchChange(e.target.value)}
      onClear={onSearchClear}
      onPressEnter={onSearchSubmit}
      suffix={searchSuffix}
      css={{ width: '100%' }}
    />
  );

  if (expandSearch) {
    return (
      <div
        ref={toolbarRef}
        data-search-layout={searchOnNewRow ? 'below-controls' : 'inline'}
        css={{
          display: 'grid',
          gridTemplateColumns: searchOnNewRow ? 'max-content minmax(0, 1fr)' : 'max-content minmax(0, 1fr) max-content',
          alignItems: 'center',
          gap: theme.spacing.sm,
          containerType: 'inline-size',
          containerName: TRACES_TOOLBAR_CONTAINER_NAME,
        }}
      >
        <div
          ref={leftControlsRef}
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            width: 'max-content',
            gridColumn: 1,
            gridRow: 1,
          }}
        >
          {leftControls}
        </div>
        <div css={{ minWidth: 0, gridColumn: searchOnNewRow ? '1 / -1' : 2, gridRow: searchOnNewRow ? 2 : 1 }}>
          {searchControl ?? searchInput}
        </div>
        <div
          ref={rightControlsRef}
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            width: 'max-content',
            justifySelf: 'end',
            gridColumn: searchOnNewRow ? 2 : 3,
            gridRow: 1,
          }}
        >
          {rightControls}
        </div>
      </div>
    );
  }

  return (
    // Query container so the controls collapse on the toolbar's own width, not the viewport's.
    // Wraps to a second row when the contents (e.g. the fixed-width CUSTOM-date RangePicker) can't
    // fit on one line, rather than pushing the trailing controls off-screen (ML-68743/68769).
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: theme.spacing.sm,
        containerType: 'inline-size',
        containerName: TRACES_TOOLBAR_CONTAINER_NAME,
      }}
    >
      {leftControls}
      {/* Caps out so a wide window doesn't stretch the search box across the row; drops its floor
          once labels collapse so the trailing controls fit on one row. */}
      <div css={{ flex: 3, minWidth: 240, maxWidth: 480, [TRACES_TOOLBAR_COLLAPSE_QUERY]: { minWidth: 120 } }}>
        {searchControl ?? searchInput}
      </div>
      {rightControls}
    </div>
  );
};
