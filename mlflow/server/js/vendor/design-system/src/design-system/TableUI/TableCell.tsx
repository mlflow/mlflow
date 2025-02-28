import classnames from 'classnames';
import type { CSSProperties } from 'react';
import React, { forwardRef, useContext } from 'react';

import { TableContext } from './Table';
import { repeatingElementsStyles, tableClassNames } from './tableStyles';
import { Typography } from '../Typography';
import type { TypographyTextProps } from '../Typography';
import type { HTMLDataAttributes } from '../types';

export interface TableCellProps extends HTMLDataAttributes, React.HTMLAttributes<HTMLDivElement> {
  /** @deprecated Use `multiline` prop instead. This prop will be removed soon. */
  ellipsis?: boolean;
  /** Enables multiline wrapping */
  multiline?: boolean;
  /** How to horizontally align the cell contents */
  align?: 'left' | 'center' | 'right';
  /** Class name property */
  className?: string;
  /** Style property */
  style?: CSSProperties;
  /** Child nodes for the table cell */
  children?: React.ReactNode | React.ReactNode[];
  /** If the content of this cell should be wrapped with Typography. Should only be set to false if
   * content is not a text (e.g. images) or you really need to render custom content. */
  wrapContent?: boolean;
}

export const TableCell = forwardRef<HTMLDivElement, TableCellProps>(function (
  { children, className, ellipsis = false, multiline = false, align = 'left', style, wrapContent = true, ...rest },
  ref,
) {
  const { size, grid } = useContext(TableContext);

  let typographySize: TypographyTextProps['size'] = 'md';

  if (size === 'small') {
    typographySize = 'sm';
  }

  const content =
    wrapContent === true ? (
      <Typography.Text
        ellipsis={!multiline}
        size={typographySize}
        title={(!multiline && typeof children === 'string' && children) || undefined}
        // Needed for the button focus outline to be visible for the expand/collapse buttons
        css={{ '&:has(> button)': { overflow: 'visible' } }}
      >
        {children}
      </Typography.Text>
    ) : (
      children
    );

  return (
    <div
      {...rest}
      role="cell"
      style={{ textAlign: align, ...style }}
      ref={ref}
      // PE-259 Use more performance className for grid but keep css= for compatibility.
      css={!grid ? repeatingElementsStyles.cell : undefined}
      className={classnames(grid && tableClassNames.cell, className)}
    >
      {content}
    </div>
  );
});
