import React from 'react';
import { Pagination } from '@databricks/design-system';

type Props = {
  currentPage: number;
  isLastPage: boolean;
  onClickNext: () => void;
  onClickPrev: () => void;
  maxResultOptions?: string[];
  handleSetMaxResult?: ({ key }: { key: number }) => void;
  getSelectedPerPageSelection: () => number;
  removeBottomSpacing?: boolean;
};

export class SimplePagination extends React.Component<Props> {
  render() {
    const { currentPage, isLastPage, onClickNext, onClickPrev, maxResultOptions, removeBottomSpacing } = this.props;
    const numEntries = this.props.getSelectedPerPageSelection();
    let total;

    // This is necessary because for tables using this component, we do not know the total records.
    // Thus this is a proxy to determine whether or not the "next" button should be disabled ==>
    // if it's the last page, we set total = max number of entries, else we add 1 page more than
    // the current page.
    if (isLastPage) {
      total = currentPage * numEntries;
    } else {
      total = (currentPage + 1) * numEntries;
    }

    return (
      <div
        className="pagination-section"
        css={[classNames.wrapper, classNames.paginationOverride, removeBottomSpacing && classNames.removeBottomSpacing]}
        data-testid="pagination-section"
      >
        <Pagination
          currentPageIndex={currentPage}
          numTotal={total}
          onChange={(nextPage) => {
            // Fire callbacks only if new page differs from the previous one
            if (nextPage !== currentPage) {
              nextPage > currentPage ? onClickNext() : onClickPrev();
            }
          }}
          dangerouslySetAntdProps={{
            showQuickJumper: false,
            showSizeChanger: true,
            pageSize: numEntries,
            pageSizeOptions: maxResultOptions,
            onShowSizeChange: (current, size) => this.props.handleSetMaxResult?.({ key: size }),
          }}
          pageSize={1}
          componentId="codegen_mlflow_app_src_common_components_SimplePagination.tsx_54"
        />
      </div>
    );
  }
}

const classNames = {
  wrapper: {
    textAlign: 'right',
    paddingBottom: 30,
  },
  removeBottomSpacing: {
    paddingBottom: 0,
  },
  paginationOverride: {
    // Hide extra page buttons
    '.du-bois-light-pagination-item:not(.du-bois-light-pagination-item-active)': {
      display: 'none',
    },
    // Hide jump buttons
    '.du-bois-light-pagination-jump-prev': {
      display: 'none',
    },
  },
};
