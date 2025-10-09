import { useDesignSystemTheme } from '@databricks/design-system';
import { Cell, Table } from '@tanstack/react-table';
import { EvaluationDatasetRecord } from '../types';
import { CodeSnippet } from '@databricks/web-shared/snippet';

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
  // only applies for 'md' and 'lg' row sizes
  const rowHeight = rowSize === 'md' ? theme.typography.fontSizeLg * 5 : theme.typography.fontSizeLg * 10;

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
        <CodeSnippet
          language="json"
          style={{
            padding: theme.spacing.xs,
            border: `1px solid ${theme.colors.border}`,
            height: rowHeight,
            width: '100%',
          }}
        >
          {value}
        </CodeSnippet>
      )}
    </div>
  );
};
