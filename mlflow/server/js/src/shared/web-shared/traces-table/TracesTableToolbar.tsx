import { type RefObject } from 'react';
import { Input, type InputRef, SearchIcon, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

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
  onSearchSubmit,
  leftControls,
  rightControls,
}: TracesTableToolbarProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

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

  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
      {leftControls}
      {/* Grows with the toolbar but caps out so a very wide window doesn't stretch the search box
          across the whole row; extra space past the cap flows to the trailing controls / spacer. */}
      <div css={{ flex: 3, minWidth: 240, maxWidth: 480 }}>{searchInput}</div>
      {rightControls}
    </div>
  );
};
