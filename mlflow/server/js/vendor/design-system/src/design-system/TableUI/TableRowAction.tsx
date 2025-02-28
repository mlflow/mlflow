import { css } from '@emotion/react';
import type { CSSProperties } from 'react';
import React, { forwardRef, useContext } from 'react';

import { TableContext } from './Table';
import { TableRowContext } from './TableRow';
import { useDesignSystemTheme } from '../Hooks';
import type { HTMLDataAttributes } from '../types';

export interface TableRowActionProps extends HTMLDataAttributes, React.HTMLAttributes<HTMLDivElement> {
  /** Style property */
  style?: CSSProperties;
  /** Child nodes for the table row. Should contain a single small-sized button. */
  children?: React.ReactNode;
  /** Class name property */
  className?: string;
}

const TableRowActionStyles = {
  container: css({
    width: 32,
    paddingTop: 'var(--vertical-padding)',
    paddingBottom: 'var(--vertical-padding)',
    display: 'flex',
    alignItems: 'start',
    justifyContent: 'center',
  }),
};

export const TableRowAction = forwardRef<HTMLDivElement, TableRowActionProps>(function TableRowAction(
  { children, style, className, ...rest },
  ref,
) {
  const { size } = useContext(TableContext);
  const { isHeader } = useContext(TableRowContext);
  const { theme } = useDesignSystemTheme();

  return (
    <div
      {...rest}
      ref={ref}
      role={isHeader ? 'columnheader' : 'cell'}
      style={{
        ...style,
        ['--vertical-padding' as any]: size === 'default' ? `${theme.spacing.xs}px` : 0,
      }}
      css={TableRowActionStyles.container}
      className={className}
    >
      {children}
    </div>
  );
});

/** @deprecated Use `TableRowAction` instead */
export const TableRowMenuContainer = TableRowAction;
