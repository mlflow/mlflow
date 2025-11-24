import { useDesignSystemTheme } from '@databricks/design-system';
import type { Cell, Table } from '@tanstack/react-table';
import type { EvaluationDatasetRecord } from '../types';
import { useMemo } from 'react';
import { escapeRegExp } from 'lodash';

const HIGHLIGHT_COLOR = 'yellow200';

const HighlightedText = ({ text, searchFilter, theme }: { text: string; searchFilter: string; theme: any }) => {
  const highlightedParts = useMemo(() => {
    if (!searchFilter.trim()) {
      return [text];
    }

    const regex = new RegExp(`(${escapeRegExp(searchFilter.trim())})`, 'gi');
    const parts = text.split(regex);

    return parts.map((part, index) => {
      const isMatch = part.toLowerCase() === searchFilter.toLowerCase().trim();
      if (isMatch) {
        return (
          <span key={index} css={{ backgroundColor: theme.colors[HIGHLIGHT_COLOR] }}>
            {part}
          </span>
        );
      }
      return part;
    });
  }, [text, searchFilter, theme]);

  return <>{highlightedParts}</>;
};

export const JsonCell = ({
  cell,
  table: { options },
}: {
  cell: Cell<EvaluationDatasetRecord, any>;
  table: Table<EvaluationDatasetRecord>;
}) => {
  const { theme } = useDesignSystemTheme();
  const value = JSON.stringify(cell.getValue(), null, 2);
  const rowSize = (options?.meta as any)?.rowSize;
  const searchFilter = (options?.meta as any)?.searchFilter || '';

  // Calculate number of lines for dynamic row count
  const lineCount = useMemo(() => value.split('\n').length, [value]);
  const rows = useMemo(() => Math.min(Math.max(lineCount + 1, 3), 20), [lineCount]);

  // Max height depends on row size: 'md' shows 10 lines, 'lg' shows 20 lines
  const maxHeightInLines = rowSize === 'md' ? 10 : 20;
  const maxHeight = theme.typography.fontSizeLg * maxHeightInLines;

  return (
    <div css={{ overflow: 'hidden', display: 'flex', alignItems: 'center', gap: theme.spacing.xs, flex: 1 }}>
      {rowSize === 'sm' ? (
        <code
          css={{
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            fontSize: `${theme.typography.fontSizeSm}px !important`,
            margin: '0 !important',
          }}
        >
          {value}
        </code>
      ) : (
        <div
          css={{
            overflow: 'auto',
            fontFamily: 'monospace',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            resize: 'vertical',
            width: '100%',
            border: `1px solid ${theme.colors.border}`,
            padding: theme.spacing.xs,
            backgroundColor: theme.colors.backgroundSecondary,
            fontSize: `${theme.typography.fontSizeSm}px`,
            maxHeight,
          }}
        >
          <HighlightedText text={value} searchFilter={searchFilter} theme={theme} />
        </div>
      )}
    </div>
  );
};
