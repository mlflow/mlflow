/* eslint-disable jsx-a11y/no-noninteractive-tabindex -- Keyboard users need to focus the scrollable JSON viewport. */
import { ChevronDownIcon, ChevronRightIcon, useDesignSystemTheme } from '@databricks/design-system';
import { css, type SerializedStyles } from '@emotion/react';
import { useVirtualizer } from '@tanstack/react-virtual';
import React, { useCallback, useMemo, useRef, useState } from 'react';

import { getJsonColors, renderPrimitiveValue } from './ModelTraceExplorerJsonRendering.utils';
import { getJsonRows, type JsonRow } from './ModelTraceExplorerJsonRows.utils';

const ESTIMATED_ROW_HEIGHT = 24;
const VIRTUAL_VIEWPORT_MAX_HEIGHT = '70vh';
const EMPTY_TOGGLED: ReadonlySet<WeakKey> = new Set();

type Theme = ReturnType<typeof useDesignSystemTheme>['theme'];
type JsonColors = ReturnType<typeof getJsonColors>;

interface JsonRowStyles {
  colors: JsonColors;
  indentSize: number;
  line: SerializedStyles;
  lineNumber: SerializedStyles;
  primitive: SerializedStyles;
  header: SerializedStyles;
  chevron: SerializedStyles;
  chevronIcon: SerializedStyles;
  key: SerializedStyles;
  colon: SerializedStyles;
  punctuation: SerializedStyles;
  ellipsis: SerializedStyles;
  preview: SerializedStyles;
  closing: SerializedStyles;
  maxDepth: SerializedStyles;
}

// Built once per theme and shared by every row, instead of serializing styles per line.
const getJsonRowStyles = (theme: Theme): JsonRowStyles => {
  const colors = getJsonColors(theme.isDarkMode, theme.colors.textPrimary, theme.colors.textSecondary);
  const mono = {
    fontFamily: 'monospace',
    fontSize: theme.typography.fontSizeSm,
    lineHeight: theme.typography.lineHeightBase,
  };
  return {
    colors,
    indentSize: theme.spacing.md,
    line: css({ display: 'flex', paddingTop: 2, paddingBottom: 2 }),
    lineNumber: css({
      minWidth: 40,
      paddingRight: theme.spacing.sm,
      textAlign: 'right',
      color: theme.colors.textSecondary,
      userSelect: 'none',
      ...mono,
    }),
    primitive: css({ display: 'flex', alignItems: 'flex-start', ...mono }),
    header: css({
      display: 'flex',
      alignItems: 'center',
      cursor: 'pointer',
      borderRadius: theme.borders.borderRadiusSm,
      marginLeft: -theme.spacing.xs,
      marginRight: -theme.spacing.xs,
      paddingRight: theme.spacing.xs,
      ...mono,
    }),
    chevron: css({
      marginRight: 4,
      marginLeft: theme.spacing.xs,
      display: 'flex',
      alignItems: 'center',
      color: theme.colors.textSecondary,
    }),
    chevronIcon: css({ fontSize: theme.spacing.mid }),
    key: css({ color: colors.key }),
    colon: css({ color: colors.punctuation, marginLeft: 2, marginRight: theme.spacing.xs }),
    punctuation: css({ color: colors.punctuation }),
    ellipsis: css({ color: colors.punctuation, marginLeft: 2 }),
    preview: css({ color: theme.colors.textSecondary, marginLeft: theme.spacing.xs, fontStyle: 'italic', ...mono }),
    closing: css({ color: colors.punctuation, ...mono }),
    maxDepth: css({ color: theme.colors.textSecondary, fontStyle: 'italic', ...mono }),
  };
};

const getCollapsedPreview = (isArray: boolean, entryCount: number): string => {
  if (entryCount === 0) {
    return '';
  }
  if (isArray) {
    return ` // ${entryCount} ${entryCount === 1 ? 'item' : 'items'}`;
  }
  return ` // ${entryCount} ${entryCount === 1 ? 'property' : 'properties'}`;
};

const JsonRowKey = ({ nodeKey, styles }: { nodeKey?: string; styles: JsonRowStyles }) =>
  nodeKey === undefined ? null : (
    <>
      <span css={styles.key}>"{nodeKey}"</span>
      <span css={styles.colon}>:</span>
    </>
  );

const JsonRowView = React.memo(function JsonRowView({
  row,
  styles,
  onToggle,
}: {
  row: JsonRow;
  styles: JsonRowStyles;
  onToggle: (container: WeakKey) => void;
}) {
  const indent = { paddingLeft: row.depth * styles.indentSize };
  const lineNumber = <span css={styles.lineNumber}>{row.line}</span>;

  switch (row.kind) {
    case 'maxDepth':
      return (
        <div css={styles.line}>
          {lineNumber}
          <div css={styles.maxDepth} style={indent}>
            [Max depth reached]
          </div>
        </div>
      );
    case 'primitive':
      return (
        <div css={styles.line}>
          {lineNumber}
          <div css={styles.primitive} style={indent}>
            <JsonRowKey nodeKey={row.nodeKey} styles={styles} />
            {renderPrimitiveValue(row.value, styles.colors)}
            {!row.isLast && <span css={styles.punctuation}>,</span>}
          </div>
        </div>
      );
    case 'close':
      return (
        <div css={styles.line}>
          {lineNumber}
          <div css={styles.closing} style={indent}>
            {row.isArray ? ']' : '}'}
            {!row.isLast && ','}
          </div>
        </div>
      );
    case 'open': {
      const close = row.isArray ? ']' : '}';
      return (
        <div css={styles.line}>
          {lineNumber}
          <div
            css={styles.header}
            style={indent}
            role="button"
            tabIndex={0}
            aria-expanded={!row.collapsed}
            aria-label={row.nodeKey ?? 'JSON container'}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                onToggle(row.container);
              }
            }}
            onClick={() => onToggle(row.container)}
          >
            <span css={styles.chevron}>
              {row.collapsed ? (
                <ChevronRightIcon css={styles.chevronIcon} />
              ) : (
                <ChevronDownIcon css={styles.chevronIcon} />
              )}
            </span>
            <JsonRowKey nodeKey={row.nodeKey} styles={styles} />
            <span css={styles.punctuation}>{row.isArray ? '[' : '{'}</span>
            {row.collapsed ? (
              <>
                <span css={styles.ellipsis}>...</span>
                <span css={styles.punctuation}>{close}</span>
                <span css={styles.preview}>{getCollapsedPreview(row.isArray, row.entryCount)}</span>
              </>
            ) : (
              row.entryCount === 0 && (
                <>
                  <span css={styles.punctuation}>{close}</span>
                  {!row.isLast && <span css={styles.punctuation}>,</span>}
                </>
              )
            )}
          </div>
        </div>
      );
    }
  }
});

/**
 * Draws a JSON object or array like CollapsibleJsonViewer, but from a flat list of lines that covers only expanded
 * containers. `rowLimit` renders just the first lines (for a collapsed preview), and long expansions are
 * virtualized, so the DOM scales with what is on screen rather than with the payload.
 */
export function ModelTraceExplorerJsonRowsViewer({
  value,
  expandDepth,
  rowLimit,
}: {
  value: WeakKey;
  /** Containers at or above this depth start expanded; the root is depth 0. */
  expandDepth: number;
  rowLimit?: number;
}): JSX.Element {
  const { theme } = useDesignSystemTheme();
  const styles = useMemo(() => getJsonRowStyles(theme), [theme]);
  // Containers the user flipped from their default state; dropped when the value changes.
  const [toggleState, setToggleState] = useState<{ value: WeakKey; toggled: ReadonlySet<WeakKey> }>({
    value,
    toggled: EMPTY_TOGGLED,
  });
  const toggled = toggleState.value === value ? toggleState.toggled : EMPTY_TOGGLED;
  const onToggle = useCallback(
    (container: WeakKey) =>
      setToggleState((previous) => {
        const next = new Set(previous.value === value ? previous.toggled : EMPTY_TOGGLED);
        if (!next.delete(container)) {
          next.add(container);
        }
        return { value, toggled: next };
      }),
    [value],
  );
  const { rows } = useMemo(
    () =>
      getJsonRows(
        value,
        (container, depth) => {
          // A toggle flips a container from its default state.
          const isExpandedByDefault = depth <= expandDepth;
          return isExpandedByDefault !== toggled.has(container);
        },
        rowLimit,
      ),
    [value, expandDepth, toggled, rowLimit],
  );

  // Keep the expanded viewport and focused toggles mounted when containers collapse.
  const shouldVirtualize = rowLimit === undefined;
  const scrollRef = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({
    count: shouldVirtualize ? rows.length : 0,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ESTIMATED_ROW_HEIGHT,
    overscan: 20,
  });

  return (
    <div
      css={{
        backgroundColor: theme.colors.backgroundSecondary,
        padding: theme.spacing.sm,
        position: 'relative',
        borderRadius: theme.borders.borderRadiusSm,
      }}
    >
      {shouldVirtualize ? (
        <div
          ref={scrollRef}
          role="region"
          aria-label="JSON"
          tabIndex={0}
          css={{ maxHeight: VIRTUAL_VIEWPORT_MAX_HEIGHT, overflow: 'auto' }}
        >
          <div
            css={{ position: 'relative', paddingRight: theme.spacing.md * 2 }}
            style={{ height: virtualizer.getTotalSize() }}
          >
            {virtualizer.getVirtualItems().map((item) => (
              <div
                key={item.key}
                data-index={item.index}
                ref={virtualizer.measureElement}
                css={{ position: 'absolute', top: 0, left: 0, width: '100%' }}
                style={{ transform: `translateY(${item.start}px)` }}
              >
                <JsonRowView row={rows[item.index]} styles={styles} onToggle={onToggle} />
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div css={{ paddingRight: theme.spacing.md * 2 }}>
          {rows.map((row, index) => (
            <JsonRowView key={index} row={row} styles={styles} onToggle={onToggle} />
          ))}
        </div>
      )}
    </div>
  );
}
