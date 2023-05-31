/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

/**
 * Custom header component
 * https://www.ag-grid.com/javascript-grid-header-rendering/#react-header-rendering
 */
import React from 'react';

type OwnRunsTableCustomHeaderProps = {
  enableSorting?: boolean;
  displayName?: string;
  canonicalSortKey?: string;
  style?: string;
  orderByAsc?: boolean;
  orderByKey?: string;
  onSortBy?: (...args: any[]) => any;
  computedStylesOnSortKey?: (...args: any[]) => any;
};

type RunsTableCustomHeaderProps = OwnRunsTableCustomHeaderProps &
  typeof RunsTableCustomHeader.defaultProps;

export class RunsTableCustomHeader extends React.Component<RunsTableCustomHeaderProps> {
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
        <span data-test-id={`sort-header-${displayName}`} title={canonicalSortKey}>
          {displayName}
        </span>
      </div>
    );
  }
}

type SortByIconProps = {
  orderByAsc?: boolean;
};

export function SortByIcon({ orderByAsc }: SortByIconProps) {
  return (
    <span style={styles.headerSortIcon}>
      <i className={`fa fa-long-arrow-${orderByAsc ? 'up' : 'down'}`} />
    </span>
  );
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
