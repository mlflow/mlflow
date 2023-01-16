import { SortAscendingIcon, SortDescendingIcon } from '@databricks/design-system';
import { Theme } from '@emotion/react';
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
        css={styles.headerLabelWrapper(enableSorting)}
        className={getClassName(canonicalSortKey)}
        onClick={enableSorting ? () => this.handleSortBy() : undefined}
      >
        <span data-test-id={`sort-header-${displayName}`}>{displayName}</span>
        {enableSorting && canonicalSortKey === orderByKey ? (
          orderByAsc ? (
            <SortAscendingIcon />
          ) : (
            <SortDescendingIcon />
          )
        ) : null}
      </div>
    );
  }
}

const styles = {
  headerLabelWrapper: (sortable: boolean) => (theme: Theme) => ({
    height: '100%',
    width: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0 12px',
    svg: {
      color: theme.colors.textSecondary,
    },
    '&:hover': {
      color: sortable ? theme.colors.actionTertiaryTextHover : 'unset',
      svg: {
        color: theme.colors.actionTertiaryTextHover,
      },
    },
  }),

  headerSortIcon: {
    marginRight: 4,
  },
};
