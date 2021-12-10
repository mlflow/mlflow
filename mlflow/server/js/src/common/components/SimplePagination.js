import React from 'react';
import PropTypes from 'prop-types';
import { Menu, Pagination } from 'antd';
import { css } from 'emotion';

export class SimplePagination extends React.Component {
  static propTypes = {
    currentPage: PropTypes.number.isRequired,
    isLastPage: PropTypes.bool.isRequired,
    onClickNext: PropTypes.func.isRequired,
    onClickPrev: PropTypes.func.isRequired,
    loading: PropTypes.bool,
    maxResultOptions: PropTypes.array,
    handleSetMaxResult: PropTypes.func,
    getSelectedPerPageSelection: PropTypes.func,
  };

  constructDropdown() {
    return (
      <Menu
        className={`pagination-dropdown ${classNames.paginationDropdownMenuWrapper}`}
        onClick={this.props.handleSetMaxResult}
      >
        {this.props.maxResultOptions.map((num_models) => (
          <Menu.Item key={num_models.toString()} title={num_models.toString()}>
            {num_models}
          </Menu.Item>
        ))}
      </Menu>
    );
  }

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
      <div className={`pagination-section ${classNames.wrapper}`}>
        <Pagination
          className={classNames.paginationOverride}
          current={currentPage}
          total={total}
          onChange={(nextPage) => (nextPage > currentPage ? onClickNext() : onClickPrev())}
          showSizeChanger
          pageSize={numEntries}
          pageSizeOptions={maxResultOptions}
          onShowSizeChange={(current, size) => this.props.handleSetMaxResult({ key: size })}
        />
      </div>
    );
  }
}

const classNames = {
  wrapper: css({
    textAlign: 'right',
    paddingBottom: 30,
  }),
  paginationDropdownMenuWrapper: css({
    '.ant-dropdown-menu-item': {
      textAlign: 'center',
    },
  }),
  paginationOverride: css({
    '.ant-pagination-item:not(.ant-pagination-item-active)': {
      display: 'none',
    },
  }),
};
