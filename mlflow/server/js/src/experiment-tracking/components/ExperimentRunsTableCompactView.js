import React from 'react';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import ExperimentViewUtil from './ExperimentViewUtil';
import { RunInfo } from '../sdk/MlflowMessages';
import classNames from 'classnames';
import { Dropdown, MenuItem } from 'react-bootstrap';
import ExperimentRunsSortToggle from './ExperimentRunsSortToggle';
import BaggedCell from './BaggedCell';
import { CellMeasurer, CellMeasurerCache, AutoSizer, Column, Table } from 'react-virtualized';
import _ from 'lodash';
import { LoadMoreBar } from './LoadMoreBar';

import 'react-virtualized/styles.css';
import { ColumnTypes } from '../constants';

export const NUM_RUN_METADATA_COLS = 10;
const TABLE_HEADER_HEIGHT = 40;
const UNBAGGED_COL_WIDTH = 125;
const BAGGED_COL_WIDTH = 250;
const BORDER_STYLE = '1px solid #e2e2e2';
const LOAD_MORE_ROW_HEIGHT = 37;

const styles = {
  sortArrow: {
    marginLeft: '2px',
  },
  sortContainer: {
    minHeight: '18px',
  },
  sortToggle: {
    cursor: 'pointer',
  },
  sortKeyName: {
    display: 'inline-block',
  },
  metricParamCellContent: {
    display: 'inline-block',
    maxWidth: 120,
  },
  metricParamNameContainer: {
    verticalAlign: 'middle',
    display: 'inline-block',
    overflow: 'hidden',
  },
  unbaggedMetricParamColHeader: {
    verticalAlign: 'middle',
    maxWidth: UNBAGGED_COL_WIDTH,
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    padding: '8px 0px 8px 8px',
    height: '100%',
  },
  columnStyle: {
    display: 'flex',
    alignItems: 'flex-start',
  },
  baggedCellContainer: {
    whiteSpace: 'normal',
  },
};

/**
 * Compact table view for displaying runs associated with an experiment. Renders metrics/params in
 * a single table cell per run (as opposed to one cell per metric/param).
 */
export class ExperimentRunsTableCompactView extends React.Component {
  constructor(props) {
    super(props);
    this.getRow = this.getRow.bind(this);
    this.tableRef = React.createRef();
    this.state = {
      expanding: false,
      isAtScrollBottom: false,
    };
  }

  static propTypes = {
    runInfos: PropTypes.arrayOf(PropTypes.instanceOf(RunInfo)).isRequired,
    modelVersionsByRunUuid: PropTypes.object.isRequired,
    // List of list of params in all the visible runs
    paramsList: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.object)).isRequired,
    // List of list of metrics in all the visible runs
    metricsList: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.object)).isRequired,
    // List of tags dictionary in all the visible runs.
    tagsList: PropTypes.arrayOf(PropTypes.object).isRequired,
    // Function which takes one parameter (runId)
    onCheckbox: PropTypes.func.isRequired,
    onCheckAll: PropTypes.func.isRequired,
    onExpand: PropTypes.func.isRequired,
    isAllChecked: PropTypes.bool.isRequired,
    onSortBy: PropTypes.func.isRequired,
    orderByKey: PropTypes.string,
    orderByAsc: PropTypes.bool.isRequired,
    runsSelected: PropTypes.object.isRequired,
    runsExpanded: PropTypes.object.isRequired,
    paramKeyList: PropTypes.arrayOf(PropTypes.string).isRequired,
    metricKeyList: PropTypes.arrayOf(PropTypes.string).isRequired,
    metricRanges: PropTypes.object.isRequired,
    // Handler for adding a metric or parameter to the set of bagged columns. All bagged metrics
    // are displayed in a single column, while each unbagged metric has its own column. Similar
    // logic applies for params.
    onAddBagged: PropTypes.func.isRequired,
    // Handler for removing a metric or parameter from the set of bagged columns.
    onRemoveBagged: PropTypes.func.isRequired,
    // Array of keys corresponding to unbagged params
    unbaggedParams: PropTypes.arrayOf(PropTypes.string).isRequired,
    // Array of keys corresponding to unbagged metrics
    unbaggedMetrics: PropTypes.arrayOf(PropTypes.string).isRequired,

    numRunsFromLatestSearch: PropTypes.number,
    handleLoadMoreRuns: PropTypes.func.isRequired,
    loadingMore: PropTypes.bool.isRequired,
    categorizedUncheckedKeys: PropTypes.object.isRequired,
    nestChildren: PropTypes.bool,
  };

  /** Returns a row of table content (i.e. a non-header row) corresponding to a single run. */
  getRow({ idx, isParent, hasExpander, expanderOpen, childrenIds }) {
    const {
      runInfos,
      modelVersionsByRunUuid,
      paramsList,
      metricsList,
      onCheckbox,
      orderByKey,
      orderByAsc,
      runsSelected,
      tagsList,
      onSortBy,
      onExpand,
      paramKeyList,
      metricKeyList,
      metricRanges,
      unbaggedMetrics,
      unbaggedParams,
      onRemoveBagged,
      categorizedUncheckedKeys,
    } = this.props;
    const paramsMap = ExperimentViewUtil.toParamsMap(paramsList[idx]);
    const metricsMap = ExperimentViewUtil.toMetricsMap(metricsList[idx]);
    const runInfo = runInfos[idx];
    const modelVersionInfo = modelVersionsByRunUuid[runInfo.run_uuid];
    const selected = runsSelected[runInfo.run_uuid] === true;
    const rowContents = [
      ExperimentViewUtil.getCheckboxForRow(selected, () => onCheckbox(runInfo.run_uuid), 'div'),
      ExperimentViewUtil.getExpander(
        hasExpander,
        expanderOpen,
        () => {
          onExpand(runInfo.run_uuid, childrenIds);
          this.setState({ expanding: true });
        },
        runInfo.run_uuid,
        'div',
      ),
    ];
    const excludedTagsSet = new Set(categorizedUncheckedKeys[ColumnTypes.TAGS]);
    ExperimentViewUtil.getRunInfoCellsForRow(
      runInfo,
      _.pickBy(tagsList[idx], (t) => !excludedTagsSet.has(t.key)),
      isParent,
      'div',
      this.handleCellToggle,
      categorizedUncheckedKeys[ColumnTypes.ATTRIBUTES],
    ).forEach((col) => rowContents.push(col));
    rowContents.push(
      ExperimentViewUtil.getLinkedModelCell(modelVersionInfo, this.handleCellToggle),
    );
    const unbaggedParamSet = new Set(unbaggedParams);
    const unbaggedMetricSet = new Set(unbaggedMetrics);
    const baggedParams = paramKeyList.filter(
      (paramKey) => !unbaggedParamSet.has(paramKey) && paramsMap[paramKey] !== undefined,
    );
    const baggedMetrics = metricKeyList.filter(
      (metricKey) => !unbaggedMetricSet.has(metricKey) && metricsMap[metricKey] !== undefined,
    );

    // Add params (unbagged, then bagged)
    unbaggedParams.forEach((paramKey) => {
      rowContents.push(ExperimentViewUtil.getUnbaggedParamCell(paramKey, paramsMap, 'div'));
    });
    // Add bagged params
    const paramsCellContents = baggedParams.map((paramKey) => {
      const keyname = 'param-' + paramKey;
      const sortIcon = ExperimentViewUtil.getSortIcon(
        orderByKey,
        orderByAsc,
        ExperimentViewUtil.makeCanonicalKey('params', paramKey),
      );
      return (
        <BaggedCell
          key={keyname}
          sortIcon={sortIcon}
          keyName={paramKey}
          value={paramsMap[paramKey].getValue()}
          onSortBy={onSortBy}
          isMetric={false}
          isParam
          onRemoveBagged={onRemoveBagged}
        />
      );
    });
    if (this.shouldShowBaggedColumn(true)) {
      rowContents.push(
        <div
          key={'params-container-cell-' + runInfo.run_uuid}
          className='metric-param-container-cell'
        >
          {paramsCellContents}
        </div>,
      );
    }

    // Add metrics (unbagged, then bagged)
    unbaggedMetrics.forEach((metricKey) => {
      rowContents.push(
        ExperimentViewUtil.getUnbaggedMetricCell(metricKey, metricsMap, metricRanges, 'div'),
      );
    });

    // Add bagged metrics
    const metricsCellContents = baggedMetrics.map((metricKey) => {
      const keyname = 'metric-' + metricKey;
      const sortIcon = ExperimentViewUtil.getSortIcon(
        orderByKey,
        orderByAsc,
        ExperimentViewUtil.makeCanonicalKey('metrics', metricKey),
      );
      return (
        <BaggedCell
          key={keyname}
          keyName={metricKey}
          value={metricsMap[metricKey].getValue().toString()}
          sortIcon={sortIcon}
          onSortBy={onSortBy}
          isMetric
          isParam={false}
          onRemoveBagged={onRemoveBagged}
        />
      );
    });
    if (this.shouldShowBaggedColumn(false)) {
      rowContents.push(
        <div
          key={'metrics-container-cell-' + runInfo.run_uuid}
          className='metric-param-container-cell'
        >
          {metricsCellContents}
        </div>,
      );
    }
    return {
      key: runInfo.run_uuid,
      contents: rowContents,
      isChild: !isParent,
    };
  }

  /**
   * Returns true if our table should contain a column for displaying bagged params (if isParam is
   * truthy) or bagged metrics.
   */
  shouldShowBaggedColumn(isParam) {
    const { metricKeyList, paramKeyList, unbaggedMetrics, unbaggedParams } = this.props;
    if (isParam) {
      return unbaggedParams.length !== paramKeyList.length || paramKeyList.length === 0;
    }
    return unbaggedMetrics.length !== metricKeyList.length || metricKeyList.length === 0;
  }

  /**
   * Returns an array of header-row cells (DOM elements) corresponding to metric / parameter
   * columns.
   */
  getMetricParamHeaderCells() {
    const {
      onSortBy,
      orderByKey,
      orderByAsc,
      paramKeyList,
      metricKeyList,
      unbaggedMetrics,
      unbaggedParams,
      onAddBagged,
    } = this.props;
    const columns = [];
    const getHeaderCell = (isParam, key, i) => {
      const keyType = isParam ? 'params' : 'metrics';
      const canonicalKey = ExperimentViewUtil.makeCanonicalKey(keyType, key);
      const sortIcon = ExperimentViewUtil.getSortIcon(orderByKey, orderByAsc, canonicalKey);
      const className = classNames('bottom-row', { 'left-border': i === 0 });
      const elemKey = (isParam ? 'param-' : 'metric-') + key;
      const keyContainerWidth = sortIcon ? 'calc(100% - 20px)' : '100%';
      return (
        <div key={elemKey} className={className} style={styles.unbaggedMetricParamColHeader}>
          <Dropdown id='dropdown-custom-1' style={{ width: '100%' }}>
            <ExperimentRunsSortToggle bsRole='toggle' className='metric-param-sort-toggle'>
              <span
                style={{
                  maxWidth: keyContainerWidth,
                  ...styles.metricParamNameContainer,
                }}
              >
                {key}
              </span>
              <span style={ExperimentViewUtil.styles.sortIconContainer}>{sortIcon}</span>
            </ExperimentRunsSortToggle>
            <Dropdown.Menu className='mlflow-menu'>
              <MenuItem className='mlflow-menu-item' onClick={() => onSortBy(canonicalKey, true)}>
                Sort ascending
              </MenuItem>
              <MenuItem className='mlflow-menu-item' onClick={() => onSortBy(canonicalKey, false)}>
                Sort descending
              </MenuItem>
              <MenuItem className='mlflow-menu-item' onClick={() => onAddBagged(isParam, key)}>
                Collapse column
              </MenuItem>
            </Dropdown.Menu>
          </Dropdown>
        </div>
      );
    };

    const paramClassName = classNames('bottom-row', { 'left-border': unbaggedParams.length === 0 });
    const metricClassName = classNames('bottom-row', {
      'left-border': unbaggedMetrics.length === 0,
    });
    unbaggedParams.forEach((paramKey, i) => {
      columns.push(getHeaderCell(true, paramKey, i));
    });

    if (this.shouldShowBaggedColumn(true)) {
      columns.push(
        <div key='meta-bagged-params left-border' className={paramClassName}>
          {paramKeyList.length !== 0 ? '' : '(n/a)'}
        </div>,
      );
    }
    unbaggedMetrics.forEach((metricKey, i) => {
      columns.push(getHeaderCell(false, metricKey, i));
    });
    if (this.shouldShowBaggedColumn(false)) {
      columns.push(
        <div key='meta-bagged-metrics left-border' className={metricClassName}>
          {metricKeyList.length !== 0 ? '' : '(n/a)'}
        </div>,
      );
    }
    return columns;
  }

  handleCellToggle = () => {
    this._cache.clearAll();
    this.forceUpdate();
  };

  _cache = new CellMeasurerCache({
    fixedWidth: true,
    minHeight: 32,
  });

  _lastRenderedWidth = -1;
  _lastOrderByKey = this.props.orderByKey;
  _lastOrderByAsc = this.props.orderByAsc;
  _lastRunsExpanded = this.props.runsExpanded;
  _lastUnbaggedMetrics = this.props.unbaggedMetrics;
  _lastUnbaggedParams = this.props.unbaggedParams;

  render() {
    const {
      runInfos,
      onCheckAll,
      isAllChecked,
      onSortBy,
      orderByKey,
      orderByAsc,
      tagsList,
      runsExpanded,
      unbaggedMetrics,
      unbaggedParams,
      numRunsFromLatestSearch,
      loadingMore,
      handleLoadMoreRuns,
      categorizedUncheckedKeys,
      nestChildren,
    } = this.props;
    const rows = ExperimentViewUtil.getRowRenderMetadata({
      runInfos,
      tagsList,
      runsExpanded,
      nestChildren,
    });

    const headerCells = [
      ExperimentViewUtil.getSelectAllCheckbox(onCheckAll, isAllChecked, 'div'),
      // placeholder for expander header cell,
      ExperimentViewUtil.getExpanderHeader('div'),
    ];

    ExperimentViewUtil.getRunMetadataHeaderCells(
      onSortBy,
      orderByKey,
      orderByAsc,
      'div',
      categorizedUncheckedKeys[ColumnTypes.ATTRIBUTES],
    ).forEach((headerCell) => headerCells.push(headerCell));

    this.getMetricParamHeaderCells().forEach((cell) => headerCells.push(cell));
    const showLoadMore = this.state.isAtScrollBottom || this.props.loadingMore;
    return (
      <div id='autosizer-container' className='runs-table-flex-container'>
        <AutoSizer>
          {({ width, height }) => {
            if (this._lastRenderedWidth !== width) {
              this._lastRenderedWidth = width;
              this._cache.clearAll();
            }
            if (this._lastOrderByKey !== orderByKey) {
              this._lastOrderByKey = orderByKey;
              this._cache.clearAll();
            }
            if (this._lastOrderByAsc !== orderByAsc) {
              this._lastOrderByAsc = orderByAsc;
              this._cache.clearAll();
            }
            if (this._lastUnbaggedMetrics !== unbaggedMetrics) {
              this._lastUnbaggedMetrics = unbaggedMetrics;
              this._cache.clearAll();
            }
            if (this._lastUnbaggedParams !== unbaggedParams) {
              this._lastUnbaggedParams = unbaggedParams;
              this._cache.clearAll();
            }
            const runMetadataColWidths = [
              30, // checkbox column width
              20, // expander column width
              35, // 'Status' column width
              180, // 'Date' column width
              120, // 'user' column width
              120, // 'Run Name' column width
              100, // 'Source' column width
              80, // 'Version' column width
              250, // 'Tags' column width,
              190, // 'Linked Models' column width
            ];
            const showBaggedParams = this.shouldShowBaggedColumn(true);
            const showBaggedMetrics = this.shouldShowBaggedColumn(false);
            const runMetadataWidth = runMetadataColWidths.reduce((a, b) => a + b);
            const tableMinWidth =
              BAGGED_COL_WIDTH * (showBaggedParams + showBaggedMetrics) +
              runMetadataWidth +
              UNBAGGED_COL_WIDTH * (unbaggedMetrics.length + unbaggedParams.length);
            const tableWidth = Math.max(width, tableMinWidth);
            // If we aren't showing bagged metrics or params (bagged metrics & params are the
            // only cols that use the CellMeasurer component), set the row height statically
            const cellMeasurerProps = {};
            if (showBaggedMetrics || showBaggedParams) {
              cellMeasurerProps.rowHeight = this._cache.rowHeight;
              cellMeasurerProps.deferredMeasurementCache = this._cache;
            } else {
              cellMeasurerProps.rowHeight = 32;
            }
            const numVisibleMetaColumns =
              NUM_RUN_METADATA_COLS - categorizedUncheckedKeys[ColumnTypes.ATTRIBUTES].length;
            return [
              <Table
                key='table'
                ref={this.tableRef}
                onScroll={this.handleScroll}
                {...cellMeasurerProps}
                width={tableWidth}
                height={Math.max(height - TABLE_HEADER_HEIGHT, 200) + 20}
                headerHeight={TABLE_HEADER_HEIGHT}
                overscanRowCount={2}
                rowCount={rows.length}
                gridStyle={{
                  borderLeft: BORDER_STYLE,
                  borderBottom: BORDER_STYLE,
                  borderRight: BORDER_STYLE,
                }}
                rowGetter={({ index }) => this.getRow(rows[index])}
                rowStyle={({ index }) => {
                  const base = {
                    alignItems: 'stretch',
                    borderBottom: BORDER_STYLE,
                    overflow: 'visible',
                  };
                  if (index === -1) {
                    return {
                      ...base,
                      backgroundColor: '#fafafa',
                      borderTop: BORDER_STYLE,
                      borderLeft: BORDER_STYLE,
                      borderRight: BORDER_STYLE,
                    };
                  }
                  return base;
                }}
              >
                {[...Array(numVisibleMetaColumns).keys()].map((colIdx) => {
                  return (
                    <Column
                      dataKey={'column-' + colIdx}
                      key={'column-' + colIdx}
                      width={runMetadataColWidths[colIdx]}
                      headerRenderer={() => headerCells[colIdx]}
                      style={{
                        ...styles.columnStyle,
                        // show left boarder for run tags column
                        ...(colIdx === numVisibleMetaColumns - 1
                          ? { borderLeft: BORDER_STYLE }
                          : undefined),
                      }}
                      cellRenderer={({ rowIndex, rowData, parent, dataKey }) => (
                        <CellMeasurer
                          cache={this._cache}
                          columnIndex={colIdx}
                          key={dataKey}
                          parent={parent}
                          rowIndex={rowIndex}
                        >
                          {rowData.contents[colIdx]}
                        </CellMeasurer>
                      )}
                    />
                  );
                })}
                {unbaggedParams.map((unbaggedParam, idx) => {
                  return (
                    <Column
                      key={'param-' + unbaggedParam}
                      dataKey={'param-' + unbaggedParam}
                      width={UNBAGGED_COL_WIDTH}
                      headerRenderer={() => headerCells[numVisibleMetaColumns + idx]}
                      style={styles.columnStyle}
                      cellRenderer={({ rowData }) => rowData.contents[numVisibleMetaColumns + idx]}
                    />
                  );
                })}
                {showBaggedParams && (
                  <Column
                    width={BAGGED_COL_WIDTH}
                    flexShrink={0}
                    label='Parameters'
                    dataKey='params'
                    headerRenderer={() => {
                      return (
                        <div
                          style={{
                            ...styles.unbaggedMetricParamColHeader,
                            leftBorder: BORDER_STYLE,
                          }}
                        >
                          Parameters
                        </div>
                      );
                    }}
                    style={{ ...styles.columnStyle, borderLeft: BORDER_STYLE }}
                    cellRenderer={({ rowIndex, rowData, parent, dataKey }) => {
                      const colIdx = numVisibleMetaColumns + unbaggedParams.length;
                      // Add extra padding for load more
                      const paddingOpt =
                        rowIndex === rows.length - 1
                          ? { paddingBottom: LOAD_MORE_ROW_HEIGHT * 2 }
                          : {};
                      return (
                        <CellMeasurer
                          cache={this._cache}
                          columnIndex={colIdx}
                          key={dataKey}
                          parent={parent}
                          rowIndex={rowIndex}
                        >
                          <div style={{ ...styles.baggedCellContainer, ...paddingOpt }}>
                            {rowData.contents[colIdx]}
                          </div>
                        </CellMeasurer>
                      );
                    }}
                  />
                )}
                {unbaggedMetrics.map((unbaggedMetric, idx) => {
                  const colIdx =
                    numVisibleMetaColumns + showBaggedParams + unbaggedParams.length + idx;
                  return (
                    <Column
                      key={'metric-' + unbaggedMetric}
                      label='Version'
                      dataKey={'metric-' + unbaggedMetric}
                      width={UNBAGGED_COL_WIDTH}
                      headerRenderer={() => headerCells[colIdx]}
                      style={styles.columnStyle}
                      cellRenderer={({ rowData }) => rowData.contents[colIdx]}
                    />
                  );
                })}
                {showBaggedMetrics && (
                  <Column
                    width={BAGGED_COL_WIDTH}
                    flexShrink={0}
                    label='Metrics'
                    dataKey='metrics'
                    headerRenderer={() => {
                      return (
                        <div
                          style={{
                            ...styles.unbaggedMetricParamColHeader,
                            leftBorder: BORDER_STYLE,
                          }}
                        >
                          Metrics
                        </div>
                      );
                    }}
                    style={{ ...styles.columnStyle, borderLeft: BORDER_STYLE }}
                    cellRenderer={({ rowIndex, rowData, parent, dataKey }) => {
                      const colIdx =
                        numVisibleMetaColumns +
                        showBaggedParams +
                        unbaggedParams.length +
                        unbaggedMetrics.length;
                      return (
                        <CellMeasurer
                          cache={this._cache}
                          columnIndex={colIdx}
                          key={dataKey}
                          parent={parent}
                          rowIndex={rowIndex}
                        >
                          <div style={{ ...styles.baggedCellContainer }}>
                            {rowData.contents[colIdx]}
                          </div>
                        </CellMeasurer>
                      );
                    }}
                  />
                )}
              </Table>,
              showLoadMore ? (
                <LoadMoreBar
                  key='load-more-row'
                  loadingMore={loadingMore}
                  onLoadMore={handleLoadMoreRuns}
                  disableButton={ExperimentViewUtil.disableLoadMoreButton({
                    numRunsFromLatestSearch: numRunsFromLatestSearch,
                  })}
                  nestChildren={nestChildren}
                  style={{
                    position: 'absolute',
                    bottom: 20,
                    border: BORDER_STYLE,
                    width: tableWidth,
                    height: LOAD_MORE_ROW_HEIGHT,
                  }}
                />
              ) : null,
            ];
          }}
        </AutoSizer>
      </div>
    );
  }

  componentDidUpdate(prevProps) {
    this.maybeHandleScroll();
    this.maybeHandleLoadMoreFinish(prevProps);
  }

  maybeHandleLoadMoreFinish(prevProps) {
    const loadMoreJustFinished = prevProps.loadingMore === false && this.props.loadingMore === true;
    if (loadMoreJustFinished) {
      this.setState({ isAtScrollBottom: false });
    }
  }

  maybeHandleScroll() {
    if (this.state.expanding) {
      this.handleScroll();
      this.setState({ expanding: false });
    }
  }

  handleScroll = _.debounce(() => {
    // Getting clientHeight, scrollHeight and scrollTop from the Grid instance directly here because
    // corresponding inputs provided by onScroll are wrong at mounting phase and upon toggling
    if (!this.tableRef.current) return;
    const grid = this.tableRef.current.Grid;
    const { clientHeight, scrollHeight, scrollTop } = {
      clientHeight: grid.props.height,
      scrollHeight: grid.getTotalRowsHeight(),
      scrollTop: grid.state.scrollTop,
    };
    const isRunsListShort = scrollHeight < clientHeight;
    const isAtScrollBottom = isRunsListShort || clientHeight + scrollTop >= scrollHeight;
    this.setState({ isAtScrollBottom });
  }, 100);
}

const mapStateToProps = (state, ownProps) => {
  const { metricsList } = ownProps;
  return { metricRanges: ExperimentViewUtil.computeMetricRanges(metricsList) };
};

export default connect(mapStateToProps)(ExperimentRunsTableCompactView);
