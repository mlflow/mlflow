import type { ICellRendererParams } from '@ag-grid-community/core';
import { Button } from '@databricks/design-system';
import { uniqueId } from 'lodash';
import React from 'react';
import { FormattedMessage } from 'react-intl';

export const createLoadMoreRow = () => ({
  runUuid: uniqueId('load_more'),
  isLoadMoreRow: true,
});

/**
 * A cell renderer for special type of full width rows housing "Load more"
 * button displayed at the bottom of the grid
 */
export const LoadMoreRowRenderer = React.memo(
  ({ loadMoreRunsFunc }: ICellRendererParams & { loadMoreRunsFunc: () => void }) => (
    <div css={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 32 }}>
      <Button type='primary' onClick={loadMoreRunsFunc} size='small'>
        <FormattedMessage
          defaultMessage='Load more'
          description='Load more button text to load more experiment runs'
        />
      </Button>
    </div>
  ),
);
