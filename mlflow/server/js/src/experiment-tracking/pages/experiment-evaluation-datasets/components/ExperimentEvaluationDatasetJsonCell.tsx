import { useDesignSystemTheme } from '@databricks/design-system';
import type { Cell, Table } from '@tanstack/react-table';
import type { EvaluationDatasetRecord } from '../types';
import { useMemo } from 'react';
import { escapeRegExp } from 'lodash';

const HighlightedText = ({ text, searchFilter, theme }: { text: string; searchFilter: string; theme: any }) => {
  const highlightedParts = useMemo(() => {
    if (!searchFilter.trim()) {
      return [text];
    }

    const regex = new RegExp(`(${escapeRegExp(searchFilter)})`, 'gi');
    const parts = text.split(regex);

    // Use blue highlight colors that work well in both light and dark modes
    const highlightColor = theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100;

    return parts.map((part, index) => {
      const isMatch = part.toLowerCase() === searchFilter.toLowerCase();
      if (isMatch) {
        return (
          <span key={index} css={{ backgroundColor: highlightColor }}>
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
  const cellValue = cell.getValue();
  const value = cellValue !== undefined ? JSON.stringify(cellValue, null, 2) : '';
  const rowSize = (options?.meta as any)?.rowSize;
  const searchFilter = (options?.meta as any)?.searchFilter || '';

  // Calculate number of lines for dynamic row count
  const lineCount = useMemo(() => (value ? value.split('\n').length : 1), [value]);
  const rows = useMemo(() => Math.min(Math.max(lineCount + 1, 3), 20), [lineCount]);

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
            maxHeight: theme.typography.fontSizeLg * (rowSize === 'md' ? 10 : 20),
          }}
        >
          <HighlightedText text={value} searchFilter={searchFilter} theme={theme} />
        </div>
      )}
    </div>
  );
};
