import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { getLargeTextRows, getLargeTextSearchRows } from './largeTextRows';
import { CopyActionButton } from '../../copy/CopyActionButton';
import { ACTIVE_HIGHLIGHT_COLOR, INACTIVE_HIGHLIGHT_COLOR } from '../constants';

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
  const activeRow = useMemo(
    () =>
      activeMatchIndex === undefined
        ? -1
        : search.matches.findIndex(
            (match) => match && match.index <= activeMatchIndex && match.lastIndex >= activeMatchIndex,
          ),
    [search, activeMatchIndex],
  );
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollElement.current,
    estimateSize: () => 20,
    overscan: 10,
    initialRect: { width: 0, height: 400 },
  });
  useEffect(() => {
    if (activeRow >= 0) virtualizer.scrollToIndex(activeRow, { align: 'center' });
  }, [activeRow, activeMatchIndex, search, virtualizer]);
  const revealActiveMatch = useCallback((node: HTMLSpanElement | null) => {
    node?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, []);
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
    for (let match = regex.exec(data); match && firstMatch.offset + match.index < row.end; match = regex.exec(data)) {
      const offset = firstMatch.offset + match.index;
      const start = Math.max(row.start, offset);
      const end = Math.min(row.end, firstMatch.offset + regex.lastIndex);
      const active = matchIndex === activeMatchIndex;
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
      matchIndex++;
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
        <div style={{ height: virtualizer.getTotalSize(), position: 'relative' }}>
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
