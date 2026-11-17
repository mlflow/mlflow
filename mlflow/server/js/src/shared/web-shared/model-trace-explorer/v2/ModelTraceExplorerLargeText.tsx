import { useMemo, useRef } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { getLargeTextRows } from './largeTextRows';

export const LARGE_TRACE_TEXT_LIMIT = 100_000;

export const ModelTraceExplorerLargeText = ({ text }: { text: string }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const scrollElement = useRef<HTMLDivElement>(null);
  const rows = useMemo(() => getLargeTextRows(text), [text]);
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollElement.current,
    estimateSize: () => 20,
    overscan: 10,
    initialRect: { width: 0, height: 400 },
  });

  return (
    <div>
      <Typography.Text color="secondary">
        <FormattedMessage
          defaultMessage="Large content is displayed as plain text."
          description="Explains why large trace payloads are rendered without Markdown or syntax highlighting"
        />
      </Typography.Text>
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
              {text.slice(rows[index].start, rows[index].end)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
