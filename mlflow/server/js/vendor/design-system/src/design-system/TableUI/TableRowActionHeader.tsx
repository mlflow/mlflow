import React from 'react';

import { TableRowAction } from './TableRowAction';
import { visuallyHidden } from '../utils';

export const TableRowActionHeader = ({ children }: { children: React.ReactNode }) => {
  return (
    <TableRowAction>
      <span css={visuallyHidden}>{children}</span>
    </TableRowAction>
  );
};
