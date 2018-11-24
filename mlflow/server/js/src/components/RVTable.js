import React, { PropTypes, Component } from 'react';
import Immutable from 'immutable';

import { Table, Column, AutoSizer } from 'react-virtualized';

export function defaultCellRenderer(_ref) {
  const cellData = _ref.cellData;

  if (cellData === null) {
    return '';
  }
  return String(cellData);
}

/**
 * Common usage of react-virtualized Table.
 * https://github.com/bvaughn/react-virtualized
 *
 * Provide rows with tableData. columns[].dataKey and columns[].cellRenderer determine cell data.
 */
export default class RVTable extends Component {
  static propTypes = {
    // Immututable.List of Records of row data. Total row count is determined by size of tableData.
    // Unloaded rows are undefined.
    tableData: PropTypes.instanceOf(Immutable.List),

    // Props for rendering Table

    // Array of objects containing props for Columns
    // https://github.com/bvaughn/react-virtualized/blob/master/docs/Column.md
    columns: PropTypes.array.isRequired,

    noRowsMessage: PropTypes.string,
    notLoadedMessage: PropTypes.string,
    onSort: PropTypes.func,
    onRowClick: PropTypes.func,
    rowMapper: PropTypes.func,
    wrapCellRenderer: PropTypes.func,
    headerHeight: PropTypes.number,
    rowHeight: PropTypes.number,
    isLoading: PropTypes.bool,
    isLoaded: PropTypes.bool,
    error: PropTypes.string,

    // Extra props are passed to Table
    // https://github.com/bvaughn/react-virtualized/blob/master/docs/Table.md
  };

  static defaultProps = {
    rowMapper: (row) => row,
    noRowsMessage: 'No Data',
    headerHeight: 26,
    rowHeight: 30,
    isLoading: false,
    isLoaded: false,
  };

  state = {
    lastClickedIndex: undefined,
  }

  getCellRenderer(cellRenderer) {
    const { wrapCellRenderer } = this.props;
    const finalCellRenderer = cellRenderer || defaultCellRenderer;
    if (wrapCellRenderer) {
      return wrapCellRenderer(finalCellRenderer);
    }
    return finalCellRenderer;
  }

  headerRenderer({ label }) {
    return label;
  }

  getRowClassName = ({ index }) => (this.state.lastClickedIndex === index ? 'last-clicked' : '')

  handleRowClick = (params) => {
    this.setState({ lastClickedIndex: params.index });
    this.props.onRowClick(params);
  }

  renderNoRows = () => {
    const { isLoading, isLoaded, noRowsMessage, error, notLoadedMessage } = this.props;
    if (isLoading) {
      // Empty string here to override the default
      return '';
    }
    if (isLoaded) {
      // if (error) {
      //   return <div className='no-rows-error'>
      //     <i className={`fa fa-fw fa-${IconsForType.error}`}/> {error}
      //   </div>;
      // }
      return <div className='no-rows'>{noRowsMessage}</div>;
    }
    if (notLoadedMessage) {
      return <div className='not-loaded'>{notLoadedMessage}</div>;
    }
    return '';
  }

  render() {
    const {
      tableData, rowMapper, columns, onSort, noRowsMessage, onRowClick, headerHeight, rowHeight,
      isLoading, error,
      ...tableProps
    } = this.props;

    const rowCount = (tableData && !error) ? tableData.size : 0;
    const rowGetter = ({ index }) => rowMapper(tableData.get(index));

    return <AutoSizer>
      {({ height, width }) =>
        <Table
          // Fixed sizes are for integration tests.
          // TODO move width/height || to AutoSizer defaultWidth Prop after React 16
          width={(width || 800) - 2 /* -2 for parent border.  */}
          height={(height || 600) - 2 /* -2 for parent border. */}
          headerHeight={headerHeight}
          rowGetter={rowGetter}
          rowCount={rowCount}
          rowHeight={rowHeight}
          noRowsRenderer={this.renderNoRows}
          sort={onSort}
          gridClassName={onRowClick ? 'row-clickable' : ''}
          rowClassName={this.getRowClassName}
          {...tableProps}
          onRowClick={onRowClick ? this.handleRowClick : undefined}
        >
          {columns.map((column) =>
            <Column
              key={column.dataKey}
              headerRenderer={this.headerRenderer}
              {...column}
              cellRenderer={this.getCellRenderer(column.cellRenderer)}
            />)
          }
        </Table>
      }
    </AutoSizer>;
  }
}