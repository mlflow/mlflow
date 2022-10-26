import { ICellRendererParams } from '@ag-grid-community/core';
import { Theme } from '@emotion/react';
import React from 'react';
import { Link } from 'react-router-dom';
import Routes from '../../../../../routes';

export interface RunNameCellRendererProps extends ICellRendererParams {
  value: string;
}

export const RunNameCellRenderer = React.memo(
  ({ value, data: { experimentId, runUuid } }: RunNameCellRendererProps) => (
    <Link css={styles.link} to={Routes.getRunPageRoute(experimentId, runUuid)}>
      {value}
    </Link>
  ),
);

const styles = {
  link: (theme: Theme) => ({
    display: 'inline-block',
    minWidth: theme.typography.fontSizeBase,
    minHeight: theme.typography.fontSizeBase,
  }),
};
