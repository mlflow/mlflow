import React from 'react';
import { useVirtual } from 'react-virtual';

import { Table, TableCell, TableHeader, TableRow, useDesignSystemTheme } from '@databricks/design-system';

import type { ReactMarkdownProps } from './types';

export function useParsedTableComponents({ children }: ReactMarkdownProps<'table'>): {
  header: React.ReactNode | undefined;
  rows: React.ReactElement[];
  isValid: boolean;
} {
  if (!children) {
    return {
      header: undefined,
      rows: [],
      isValid: false,
    };
  }

  const childArray = React.Children.toArray(children);

  const header = childArray[0] ?? undefined;

  // Parse rows, expect all children after the header to be tbody element containing the rows
  const rows: React.ReactElement[] = childArray.slice(1).flatMap((child) => {
    if (React.isValidElement(child)) {
      return React.Children.toArray(child.props.children).filter((c): c is React.ReactElement =>
        React.isValidElement(c),
      );
    }
    return [];
  });

  return {
    header,
    rows,
    isValid: true,
  };
}

const BasicTable = ({ children }: ReactMarkdownProps<'table'>) => {
  return (
    // Tables with many columns were not scrollable but instead become squished rendering the content unreadable
    // Fixed by wrapping the table in a div with overflow auto and setting the table to display inline-flex
    <div data-testid="basic-table" css={{ overflow: 'auto' }}>
      {/* Table has a "height: 100%" style that causes layout issues in some cases */}
      <Table
        scrollable
        css={{
          height: 'auto',
          minHeight: 'initial',
          display: 'inline-flex',
          // Tables in tool call responses were not taking up the full width of the container
          minWidth: '100%',
          zIndex: 0, // Ensure the raised header doesn't cover elements outside the table
        }}
        children={children}
      />
    </div>
  );
};

export const TableRenderer = ({ children, node }: ReactMarkdownProps<'table'>) => {
  const { header, rows, isValid } = useParsedTableComponents({ children, node });

  if (!isValid) {
    // If for some reason the table is not valid, fall back to the basic table
    return <BasicTable children={children} node={node} />;
  }

  return <VirtualizedTable header={header} rows={rows} />;
};

const MAX_TABLE_HEIGHT = 420;

const OVERSCAN = 20;

const VirtualizedTable = ({ header, rows }: { header: React.ReactNode; rows: React.ReactNode[] }) => {
  const { theme } = useDesignSystemTheme();

  const parentRef = React.useRef<HTMLDivElement>(null);

  const rowVirtualizer = useVirtual({
    size: rows.length,
    parentRef,
    overscan: OVERSCAN,
  });

  const { virtualItems, totalSize } = rowVirtualizer;

  return (
    <div
      data-testid="virtualized-table"
      ref={parentRef}
      css={{
        overflow: 'auto',
        maxHeight: MAX_TABLE_HEIGHT,
        border: '1px solid',
        borderColor: theme.colors.border,
        borderRadius: theme.borders.borderRadiusMd,
        marginBottom: theme.spacing.md,
        zIndex: 0, // Ensure the raised header doesn't cover elements outside the table
      }}
    >
      <Table
        css={{
          height: 'auto',
          minHeight: 'initial',
          display: 'inline-flex',
          minWidth: '100%',
        }}
      >
        {header}
        <div
          css={{
            position: 'relative',
            height: `${totalSize}px`,
            width: '100%',
            // Remove bottom border from the last row
            '& > div:last-child [role="cell"]': { borderBottom: 'none' },
          }}
        >
          {virtualItems.map((virtualRow) => {
            const rowIndex = virtualRow.index;
            const rowElement = rows[rowIndex];

            return (
              <div
                ref={virtualRow.measureRef}
                key={rowIndex}
                css={{
                  position: 'absolute',
                  top: `${virtualRow.start}px`,
                  width: '100%',
                }}
              >
                {rowElement}
              </div>
            );
          })}
        </div>
      </Table>
    </div>
  );
};

export const VirtualizedTableRow = ({ children, node }: ReactMarkdownProps<'tr'>) => {
  const isHeader = node?.children.some((child) => child.tagName === 'th');
  const { theme } = useDesignSystemTheme();
  return (
    <TableRow
      style={
        isHeader
          ? { position: 'sticky', top: 0, zIndex: 1, backgroundColor: theme.colors.backgroundPrimary }
          : undefined
      }
      children={children}
      isHeader={isHeader}
    />
  );
};

export const VirtualizedTableCell = ({ children, node }: ReactMarkdownProps<'td' | 'th'>) => {
  const isHeader = node?.tagName === 'th';
  const { theme } = useDesignSystemTheme();

  if (isHeader) {
    return (
      <TableHeader
        data-testid="virtualized-table-header"
        componentId="virtualized-table-header"
        css={{ paddingLeft: theme.spacing.sm, borderColor: theme.colors.border, color: theme.colors.textPrimary }}
        children={children}
      />
    );
  }

  return <TableCell children={children} />;
};
