import classnames from 'classnames';
import type { CSSProperties } from 'react';
import React, { createContext, forwardRef, useContext, useMemo } from 'react';

import { TableContext } from './Table';
import { repeatingElementsStyles, tableClassNames } from './tableStyles';
import { useDesignSystemTheme } from '../Hooks';
import type { HTMLDataAttributes } from '../types';

export const TableRowContext = createContext<{
  isHeader: boolean;
}>({
  isHeader: false,
});

export interface TableRowProps extends HTMLDataAttributes, React.HTMLAttributes<HTMLDivElement> {
  /** Style property */
  style?: CSSProperties;
  /** Class name property */
  className?: string;
  /** Set to true if this row is to be used for a table header row */
  isHeader?: boolean;
  /** @deprecated Vertical alignment of the row's cells. No longer supported (See FEINF-1937) */
  verticalAlignment?: 'top' | 'center';
  /** Child nodes for the table row */
  children?: React.ReactNode | React.ReactNode[];
}

export const TableRow = forwardRef<HTMLDivElement, TableRowProps>(function TableRow(
  { children, className, style, isHeader = false, verticalAlignment, ...rest },
  ref,
) {
  const { size, grid } = useContext(TableContext);
  const { theme } = useDesignSystemTheme();

  // Vertical only be larger if the row is a header AND size is large.
  const shouldUseLargeVerticalPadding = isHeader && size === 'default';

  let rowPadding;

  if (shouldUseLargeVerticalPadding) {
    rowPadding = theme.spacing.sm;
  } else if (size === 'default') {
    rowPadding = 6;
  } else {
    rowPadding = theme.spacing.xs;
  }

  return (
    <TableRowContext.Provider
      value={useMemo(() => {
        return { isHeader };
      }, [isHeader])}
    >
      <div
        {...rest}
        ref={ref}
        role="row"
        style={{
          ...style,
          ['--table-row-vertical-padding' as any]: `${rowPadding}px`,
        }}
        // PE-259 Use more performance className for grid but keep css= for consistency.
        css={!grid ? repeatingElementsStyles.row : undefined}
        className={classnames(className, grid && tableClassNames.row, {
          'table-isHeader': isHeader,
          'table-row-isGrid': grid,
        })}
      >
        {children}
      </div>
    </TableRowContext.Provider>
  );
});
