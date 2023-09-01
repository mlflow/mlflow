import { ICellRendererParams } from '@ag-grid-community/core';
import { Button, MinusBoxIcon, PlusSquareIcon } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React from 'react';
import { Link } from '../../../../../../common/utils/RoutingUtils';
import Routes from '../../../../../routes';
import { RunRowType } from '../../../utils/experimentPage.row-types';

export interface RunNameCellRendererProps extends ICellRendererParams {
  data: RunRowType;
  onExpand: (runUuid: string, childrenIds?: string[]) => void;
}

export const RunNameCellRenderer = React.memo(
  ({
    onExpand,
    data: { runName, experimentId, runUuid, runDateAndNestInfo, color },
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
        <Link to={Routes.getRunPageRoute(experimentId, runUuid)} css={styles.runLink}>
          <div
            css={styles.colorPill}
            data-testid='experiment-view-table-run-color'
            style={{ backgroundColor: color }}
          />
          <span css={styles.runName}>{runName}</span>
        </Link>
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
  runLink: {
    overflow: 'hidden',
    display: 'flex',
    gap: 8,
    alignItems: 'center',
  },
  runName: {
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  expanderWrapper: {
    display: 'none',
    '.ag-grid-expanders-visible &': {
      display: 'block',
    },
  },
  colorPill: {
    width: 12,
    height: 12,
    borderRadius: 6,
    flexShrink: 0,
    // Straighten it up on retina-like screens
    '@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi)': {
      marginBottom: 1,
    },
  },
  nestLevel: (level: number) => (theme: Theme) => ({
    display: 'flex',
    justifyContent: 'flex-end',
    width: (level + 1) * theme.spacing.lg,
    height: theme.spacing.lg,
  }),
};
