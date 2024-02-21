import type { ICellRendererParams } from '@ag-grid-community/core';
import { Button } from '@databricks/design-system';
import { uniqueId } from 'lodash';
import React from 'react';
import { FormattedMessage } from 'react-intl';

export const createLoadMoreRow = () => ({
  runUuid: '',
  rowUuid: uniqueId('load_more'),
  isLoadMoreRow: true,
});

/**
 * A cell renderer for special type of full width rows housing "Load more"
 * button displayed at the bottom of the grid
 */
export const LoadMoreRowRenderer = React.memo(
  ({ loadMoreRunsFunc }: ICellRendererParams & { loadMoreRunsFunc: () => void }) => (
    <div css={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 32 }}>
      <Button
        componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_loadmorerowrenderer.tsx_20"
        type="primary"
        onClick={loadMoreRunsFunc}
        size="small"
      >
        <FormattedMessage defaultMessage="Load more" description="Load more button text to load more experiment runs" />
      </Button>
    </div>
  ),
);
