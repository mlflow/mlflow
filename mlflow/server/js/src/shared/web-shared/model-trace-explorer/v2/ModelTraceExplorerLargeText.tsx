import { useLayoutEffect, useMemo, useRef } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { getLargeTextRows, getLargeTextSearchRows } from './largeTextRows';
import { CopyActionButton } from '../../copy/CopyActionButton';
import { ACTIVE_HIGHLIGHT_COLOR, INACTIVE_HIGHLIGHT_COLOR } from '../constants';

const MAX_ROW_HIGHLIGHTS = 20;

export const ModelTraceExplorerLargeText = ({
  text,
  copyText,
  searchFilter = '',
  activeMatchIndex,
}: {
  text: string;
  copyText?: string;
  searchFilter?: string;
  activeMatchIndex?: number;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const scrollElement = useRef<HTMLDivElement>(null);
  const rows = useMemo(() => getLargeTextRows(text), [text]);
  const search = useMemo(() => getLargeTextSearchRows(text, rows, searchFilter), [text, rows, searchFilter]);
  const highlightsLimited = useMemo(
    () => search.matches.some((match) => match && match.lastIndex - match.index >= MAX_ROW_HIGHLIGHTS),
    [search],
  );
  const activeRow = useMemo(
    () =>
      activeMatchIndex === undefined
        ? -1
        : search.matches.findIndex(
            (match) => match && match.index <= activeMatchIndex && match.lastIndex >= activeMatchIndex,
          ),
    [search, activeMatchIndex],
  );
  const selection = useMemo(
    () => ({ revealed: activeMatchIndex === undefined || !search.pattern }),
    [search, activeMatchIndex],
  );
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollElement.current,
    estimateSize: () => 20,
    overscan: 10,
    initialRect: { width: 0, height: 400 },
  });
  const totalSize = virtualizer.getTotalSize();
  useLayoutEffect(() => {
    if (activeRow < 0 || selection.revealed) return;
    // Initial row measurements must reach the spacer DOM before the browser clamps the scroll.
    const frame = requestAnimationFrame(() => {
      if (!selection.revealed) virtualizer.scrollToIndex(activeRow, { align: 'center' });
    });
    return () => cancelAnimationFrame(frame);
  }, [activeRow, selection, totalSize, virtualizer]);
  const revealActiveMatch = useMemo(() => {
    // Virtual rows can remount while the user scrolls without changing the search selection.
    let frame: number | undefined;
    return (node: HTMLSpanElement | null) => {
      if (frame !== undefined) {
        cancelAnimationFrame(frame);
        frame = undefined;
      }
      if (node && search.pattern && activeMatchIndex !== undefined && !selection.revealed) {
        // The parent row measures after this ref attaches and can move the match before the next frame.
        frame = requestAnimationFrame(() => {
          frame = undefined;
          if (node.isConnected) {
            selection.revealed = true;
            node.scrollIntoView({ block: 'nearest', inline: 'nearest' });
          }
        });
      }
    };
  }, [search, activeMatchIndex, selection]);
  const renderRow = (index: number) => {
    const row = rows[index];
    const firstMatch = search.matches[index];
    if (!firstMatch || !search.pattern) return text.slice(row.start, row.end);
    const regex = new RegExp(search.pattern, 'gi');
    // Include crossing matches without searching the rest of the trace for each mounted row.
    const data = text.slice(firstMatch.offset, row.end + search.queryLength - 1);
    const parts: React.ReactNode[] = [];
    let cursor = row.start;
    let matchIndex = firstMatch.index;
    let highlights = 0;
    const activeInRow =
      activeMatchIndex !== undefined &&
      activeMatchIndex >= firstMatch.index &&
      activeMatchIndex <= firstMatch.lastIndex;
    for (let match = regex.exec(data); match && firstMatch.offset + match.index < row.end; match = regex.exec(data)) {
      const offset = firstMatch.offset + match.index;
      const start = Math.max(row.start, offset);
      const end = Math.min(row.end, firstMatch.offset + regex.lastIndex);
      const active = matchIndex === activeMatchIndex;
      if (highlights < MAX_ROW_HIGHLIGHTS || active) {
        parts.push(
          text.slice(cursor, start),
          <span
            key={offset}
            ref={active && offset >= row.start ? revealActiveMatch : undefined}
            css={{ backgroundColor: theme.colors[active ? ACTIVE_HIGHLIGHT_COLOR : INACTIVE_HIGHLIGHT_COLOR] }}
          >
            {text.slice(start, end)}
          </span>,
        );
        cursor = end;
        highlights++;
      }
      matchIndex++;
      if (highlights >= MAX_ROW_HIGHLIGHTS && (!activeInRow || matchIndex > activeMatchIndex)) break;
    }
    parts.push(text.slice(cursor, row.end));
    return parts;
  };

  return (
    <div>
      <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: theme.spacing.sm }}>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Large content is displayed as plain text. Browser find and text selection cover only part of the value. Use Copy for the full text."
            description="Explains large trace text display and how to access the complete value for finding or selecting text"
          />
          {highlightsLimited && (
            <>
              {' '}
              <FormattedMessage
                defaultMessage="Search highlights are limited for dense results. Use Next/Previous to reveal every match."
                description="Explains how to navigate all search results when dense trace text limits visible highlights"
              />
            </>
          )}
        </Typography.Text>
        {copyText !== undefined && (
          <CopyActionButton componentId="shared.model-trace-explorer.copy-large-text" copyText={copyText} />
        )}
      </div>
      <div
        ref={scrollElement}
        role="region"
        // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- Keyboard users must be able to scroll this region.
        tabIndex={0}
        aria-label={intl.formatMessage({
          defaultMessage: 'Trace text',
          description: 'Scrollable large trace text region',
        })}
        css={{ maxHeight: 400, overflow: 'auto', backgroundColor: theme.colors.backgroundSecondary }}
      >
        <div style={{ height: totalSize, position: 'relative' }}>
          {virtualizer.getVirtualItems().map(({ index, key, start }) => (
            <div
              key={key}
              data-index={index}
              ref={virtualizer.measureElement}
              style={{ position: 'absolute', top: 0, width: '100%', transform: `translateY(${start}px)` }}
              css={{
                whiteSpace: 'pre-wrap',
                overflowWrap: 'anywhere',
                minHeight: 20,
                lineHeight: '20px',
                fontFamily: 'monospace',
              }}
            >
              {renderRow(index)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
