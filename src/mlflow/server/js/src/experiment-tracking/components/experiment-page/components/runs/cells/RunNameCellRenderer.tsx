import { ICellRendererParams } from '@ag-grid-community/core';
import { Button, MinusBoxIcon, PlusSquareIcon, useDesignSystemTheme } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React from 'react';
import { Link } from '../../../../../../common/utils/RoutingUtils';
import Routes from '../../../../../routes';
import { RunRowType } from '../../../utils/experimentPage.row-types';
import { GroupParentCellRenderer } from './GroupParentCellRenderer';
import invariant from 'invariant';
import { RunColorPill } from '../../RunColorPill';
import { shouldUseNewRunRowsVisibilityModel } from '../../../../../../common/utils/FeatureUtils';

export interface RunNameCellRendererProps extends ICellRendererParams {
  data: RunRowType;
  isComparingRuns?: boolean;
  onExpand: (runUuid: string, childrenIds?: string[]) => void;
}

export const RunNameCellRenderer = React.memo((props: RunNameCellRendererProps) => {
  const { theme } = useDesignSystemTheme();

  // If we're rendering a group row, use relevant component
  if (props.data.groupParentInfo) {
    return <GroupParentCellRenderer {...props} />;
  }
  const { onExpand, data } = props;
  const { runName, experimentId, runUuid, runDateAndNestInfo, color, hidden } = data;

  // If we are not rendering a group, assert existence of necessary fields
  invariant(experimentId, 'experimentId should be set for run rows');
  invariant(runUuid, 'runUuid should be set for run rows');
  invariant(runDateAndNestInfo, 'runDateAndNestInfo should be set for run rows');

  const { hasExpander, expanderOpen, childrenIds, level, belongsToGroup } = runDateAndNestInfo;

  const renderingAsParent = !isNaN(level) && hasExpander;

  return (
    <div css={styles.cellWrapper}>
      <div css={styles.expanderWrapper}>
        <div
          css={styles.nestLevel(theme)}
          style={{
            width: (level + 1) * theme.spacing.lg,
          }}
        >
          {renderingAsParent && (
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_runnamecellrenderer.tsx_46"
              css={styles.expanderButton}
              size="small"
              onClick={() => {
                onExpand(runUuid, childrenIds);
              }}
              key={'Expander-' + runUuid}
              type="link"
              icon={expanderOpen ? <MinusBoxIcon /> : <PlusSquareIcon />}
            />
          )}
        </div>
      </div>
      <Link to={Routes.getRunPageRoute(experimentId, runUuid)} css={styles.runLink}>
        <RunColorPill
          color={!belongsToGroup ? color : 'transparent'}
          hidden={shouldUseNewRunRowsVisibilityModel() && !belongsToGroup && props.isComparingRuns && hidden}
          data-testid="experiment-view-table-run-color"
        />
        <span css={styles.runName}>{runName}</span>
      </Link>
    </div>
  );
});

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
  nestLevel: (theme: Theme) => ({
    display: 'flex',
    justifyContent: 'flex-end',
    height: theme.spacing.lg,
  }),
};
