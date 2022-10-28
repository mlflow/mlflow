import React from 'react';

export interface ColumnHeaderCellProps {
  enableSorting: boolean;
  displayName: string;
  canonicalSortKey: string;
  orderByAsc: boolean;
  orderByKey: string;
  onSortBy: (sortKey: string, newOrder: boolean) => void;
  getClassName: (sortKey: string) => string;
}

export class ColumnHeaderCell extends React.Component<ColumnHeaderCellProps> {
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
      getClassName = () => '',
      orderByKey,
      orderByAsc,
    } = this.props;

    return (
      // eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/interactive-supports-focus
      <div
        role='columnheader'
        css={styles.headerLabelWrapper}
        className={getClassName(canonicalSortKey)}
        onClick={enableSorting ? () => this.handleSortBy() : undefined}
      >
        {enableSorting && canonicalSortKey === orderByKey ? (
          <span style={styles.headerSortIcon}>
            <i className={`fa fa-long-arrow-${orderByAsc ? 'up' : 'down'}`} />
          </span>
        ) : null}
        <span data-test-id={`sort-header-${displayName}`}>{displayName}</span>
      </div>
    );
  }
}

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
