/**
 * Custom header component
 * https://www.ag-grid.com/javascript-grid-header-rendering/#react-header-rendering
 */
import React from 'react';
import PropTypes from 'prop-types';

export class RunsTableCustomHeader extends React.Component {
  /**
   All `IHeaderParams` fields are available from ag-grid as well, add to propTypes when needed
   export interface IHeaderParams {
      column: Column;
      displayName: string;
      enableSorting: boolean;
      enableMenu: boolean;
      showColumnMenu: (source: HTMLElement) => void;
      progressSort: (multiSort?: boolean) => void;
      setSort: (sort: string, multiSort?: boolean) => void;
      columnApi: ColumnApi;
      api: GridApi;
      context: any;
      template: string;
    }
   */
  static propTypes = {
    /* props from `IHeaderParams` */
    enableSorting: PropTypes.bool,
    displayName: PropTypes.string,
    /* custom props from `headerComponentParams` */
    canonicalSortKey: PropTypes.string,
    // Note(Zangr) using object in headerComponentParams will cause ag-grid error, like making
    // `labelStyle` an object will cause rendering error.
    // TODO(Zangr) investigate workaround to allow passing in object as custom props
    style: PropTypes.string,
    orderByAsc: PropTypes.bool,
    orderByKey: PropTypes.string,
    onSortBy: PropTypes.func,
    computedStylesOnSortKey: PropTypes.func,
  };

  static defaultProps = {
    onSortBy: () => {},
  };

  handleSortBy() {
    const { canonicalSortKey, orderByAsc, orderByKey } = this.props;
    let newOrderByAsc = !orderByAsc;

    // If the new sortKey is not equal to the previous sortKey, reset the orderByAsc
    if (canonicalSortKey !== orderByKey) {
      newOrderByAsc = false;
    }

    this.props.onSortBy(canonicalSortKey, newOrderByAsc);
  }

  render() {
    const {
      enableSorting,
      canonicalSortKey,
      displayName,
      style = '{}',
      computedStylesOnSortKey = () => {},
      orderByKey,
      orderByAsc,
    } = this.props;

    return (
      <div
        role='columnheader'
        style={{
          ...styles.headerLabelWrapper,
          ...JSON.parse(style),
          ...computedStylesOnSortKey(canonicalSortKey),
        }}
        onClick={enableSorting ? () => this.handleSortBy() : undefined}
      >
        {enableSorting && canonicalSortKey === orderByKey ? (
          <SortByIcon orderByAsc={orderByAsc} />
        ) : null}
        <span data-test-id={`sort-header-${displayName}`}>{displayName}</span>
      </div>
    );
  }
}

export function SortByIcon({ orderByAsc }) {
  return (
    <span style={styles.headerSortIcon}>
      <i className={`fa fa-long-arrow-alt-${orderByAsc ? 'up' : 'down'}`} />
    </span>
  );
}
SortByIcon.propTypes = { orderByAsc: PropTypes.bool };

const styles = {
  headerLabelWrapper: {
    height: '100%',
    width: '100%',
    display: 'flex',
    alignItems: 'center',
    padding: '0 12px',
  },
  headerSortIcon: {
    marginRight: 4,
  },
};
