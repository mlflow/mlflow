import React from 'react';
import PropTypes from 'prop-types';
import { Pagination } from '@databricks/design-system';

export class SimplePagination extends React.Component {
  static propTypes = {
    currentPage: PropTypes.number.isRequired,
    isLastPage: PropTypes.bool.isRequired,
    onClickNext: PropTypes.func.isRequired,
    onClickPrev: PropTypes.func.isRequired,
    maxResultOptions: PropTypes.array,
    handleSetMaxResult: PropTypes.func,
    getSelectedPerPageSelection: PropTypes.func.isRequired,
  };

  render() {
    const { currentPage, isLastPage, onClickNext, onClickPrev, maxResultOptions } = this.props;
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
      <div className='pagination-section' css={[classNames.wrapper, classNames.paginationOverride]}>
        <Pagination
          currentPageIndex={currentPage}
          numTotal={total}
          onChange={(nextPage) => (nextPage > currentPage ? onClickNext() : onClickPrev())}
          dangerouslySetAntdProps={{
            showQuickJumper: false,
            showSizeChanger: true,
            pageSize: numEntries,
            pageSizeOptions: maxResultOptions,
            onShowSizeChange: (current, size) => this.props.handleSetMaxResult({ key: size }),
          }}
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
