import { ICellRendererParams } from '@ag-grid-community/core';
import { Button, MinusBoxIcon, PlusSquareIcon } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React from 'react';
import { Link } from 'react-router-dom';
import Routes from '../../../../../routes';
import { RunRowType } from '../../../utils/experimentPage.row-types';

export interface RunNameCellRendererProps extends ICellRendererParams {
  data: RunRowType;
  onExpand: (runUuid: string, childrenIds?: string[]) => void;
}

export const RunNameCellRenderer = React.memo(
  ({
    onExpand,
    data: { runName, experimentId, runUuid, runDateAndNestInfo },
  }: RunNameCellRendererProps) => {
    const { hasExpander, expanderOpen, childrenIds, level } = runDateAndNestInfo || {};

    const renderingAsParent = !isNaN(level) && hasExpander;

    return (
      <div css={styles.cellWrapper}>
        <div css={styles.expanderWrapper}>
          <div css={styles.nestLevel(level)}>
            {renderingAsParent && (
              <Button
                css={styles.expanderButton}
                size='small'
                onClick={() => {
                  onExpand(runUuid, childrenIds);
                }}
                key={'Expander-' + runUuid}
                type='link'
                icon={expanderOpen ? <MinusBoxIcon /> : <PlusSquareIcon />}
              />
            )}
          </div>
        </div>
        <Link to={Routes.getRunPageRoute(experimentId, runUuid)}>{runName}</Link>
      </div>
    );
  },
);

const styles = {
  link: (theme: Theme) => ({
    display: 'inline-block',
    minWidth: theme.typography.fontSizeBase,
    minHeight: theme.typography.fontSizeBase,
  }),
  cellWrapper: {
    display: 'flex',
  },
  expanderButton: {
    svg: {
      width: 12,
      height: 12,
    },
  },
  expanderWrapper: {
    display: 'none',
    '.ag-grid-expanders-visible &': {
      display: 'block',
    },
  },

  nestLevel: (level: number) => (theme: Theme) => ({
    display: 'flex',
    justifyContent: 'flex-end',
    width: (level + 1) * theme.spacing.lg,
  }),
};
