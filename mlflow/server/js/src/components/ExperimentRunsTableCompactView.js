import React, { Component } from 'react';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import ExperimentViewUtil from "./ExperimentViewUtil";
import { RunInfo } from '../sdk/MlflowMessages';
import classNames from 'classnames';
import { Dropdown, MenuItem } from 'react-bootstrap';
import ExperimentRunsSortToggle from './ExperimentRunsSortToggle';
import Utils from '../utils/Utils';
import BaggedCell from "./BaggedCell";

import { CellMeasurer, CellMeasurerCache, Grid, AutoSizer } from 'react-virtualized';


import ReactDOM from 'react-dom';
import { Column, Table } from 'react-virtualized';
import 'react-virtualized/styles.css'; // only needs to be imported once

// Table data as an array of objects
const list = [...Array(10000).keys()].map((i) => {
  return {
    name: "Person " + i, description:  'Software engineer',
  };
});


const styles = {
  sortArrow: {
    marginLeft: "2px",
  },
  sortContainer: {
    minHeight: "18px",
  },
  sortToggle: {
    cursor: "pointer",
  },
  sortKeyName: {
    display: "inline-block"
  },
  metricParamCellContent: {
    display: "inline-block",
    maxWidth: 120,
  },
  metricParamNameContainer: {
    verticalAlign: "middle",
    display: "inline-block",
  },
  metricParamHeaderContainer: {
    verticalAlign: "middle",
    // display: "inline-block",
    maxWidth: 120,
  }
};

/**
 * Compact table view for displaying runs associated with an experiment. Renders metrics/params in
 * a single table cell per run (as opposed to one cell per metric/param).
 */
class ExperimentRunsTableCompactView extends Component {
  constructor(props) {
    super(props);
    this.getRow = this.getRow.bind(this);
    this.onHover = this.onHover.bind(this);
  }

  static propTypes = {
    runInfos: PropTypes.arrayOf(RunInfo).isRequired,
    // List of list of params in all the visible runs
    paramsList: PropTypes.arrayOf(Array).isRequired,
    // List of list of metrics in all the visible runs
    metricsList: PropTypes.arrayOf(Array).isRequired,
    // List of tags dictionary in all the visible runs.
    tagsList: PropTypes.arrayOf(Object).isRequired,
    // Function which takes one parameter (runId)
    onCheckbox: PropTypes.func.isRequired,
    onCheckAll: PropTypes.func.isRequired,
    onExpand: PropTypes.func.isRequired,
    isAllChecked: PropTypes.bool.isRequired,
    onSortBy: PropTypes.func.isRequired,
    sortState: PropTypes.object.isRequired,
    runsSelected: PropTypes.object.isRequired,
    runsExpanded: PropTypes.object.isRequired,
    setSortByHandler: PropTypes.func.isRequired,
    paramKeyList: PropTypes.arrayOf(String).isRequired,
    metricKeyList: PropTypes.arrayOf(String).isRequired,
    metricRanges: PropTypes.object.isRequired,
    // Handler for adding a metric or parameter to the set of bagged columns. All bagged metrics
    // are displayed in a single column, while each unbagged metric has its own column. Similar
    // logic applies for params.
    onAddBagged: PropTypes.func.isRequired,
    // Handler for removing a metric or parameter from the set of bagged columns.
    onRemoveBagged: PropTypes.func.isRequired,
    // Array of keys corresponding to unbagged params
    unbaggedParams: PropTypes.arrayOf(String).isRequired,
    // Array of keys corresponding to unbagged metrics
    unbaggedMetrics: PropTypes.arrayOf(String).isRequired,
  };

  state = {
    hoverState: {isMetric: false, isParam: false, key: ""},
  };

  onHover({isParam, isMetric, key}) {
    this.setState({ hoverState: {isParam, isMetric, key} });
  }

  /** Returns a row of table content (i.e. a non-header row) corresponding to a single run. */
  getRow({ idx, isParent, hasExpander, expanderOpen, childrenIds }) {
    const {
      runInfos,
      paramsList,
      metricsList,
      onCheckbox,
      sortState,
      runsSelected,
      tagsList,
      setSortByHandler,
      onExpand,
      paramKeyList,
      metricKeyList,
      metricRanges,
      unbaggedMetrics,
      unbaggedParams,
      onRemoveBagged,
    } = this.props;
    const paramsMap =ExperimentViewUtil.toParamsMap(paramsList[idx]);
    const metricsMap = ExperimentViewUtil.toMetricsMap(metricsList[idx]);
    const tagsMap = tagsList[idx];
    const runInfo = runInfos[idx];
    const hoverState = this.state.hoverState;
    const selected = runsSelected[runInfo.run_uuid] === true;
    const rowContents = [
      ExperimentViewUtil.getCheckboxForRow(selected, () => onCheckbox(runInfo.run_uuid), "div"),
      ExperimentViewUtil.getExpander(
        hasExpander, expanderOpen, () => onExpand(runInfo.run_uuid, childrenIds), runInfo.run_uuid, "div")
    ];
    ExperimentViewUtil.getRunInfoCellsForRow(runInfo, tagsMap, isParent, "div")
      .forEach((col) => rowContents.push(col));

    const unbaggedParamSet = new Set(unbaggedParams);
    const unbaggedMetricSet = new Set(unbaggedMetrics);
    const baggedParams = paramKeyList.filter((paramKey) =>
      !unbaggedParamSet.has(paramKey) && paramsMap[paramKey] !== undefined);
    const baggedMetrics = metricKeyList.filter((metricKey) =>
      !unbaggedMetricSet.has(metricKey) && metricsMap[metricKey] !== undefined);

    // Add params (unbagged, then bagged)
    unbaggedParams.forEach((paramKey) => {
      rowContents.push(ExperimentViewUtil.getUnbaggedParamCell(paramKey, paramsMap, "div"));
    });
    // Add bagged params
    const paramsCellContents = baggedParams.map((paramKey) => {
      const isHovered = hoverState.isParam && hoverState.key === paramKey;
      const keyname = "param-" + paramKey;
      const sortIcon = ExperimentViewUtil.getSortIcon(sortState, false, true, paramKey);
      return (<BaggedCell
        key={keyname}
        // keyName={paramKey + "a".repeat(1000)} value={"b".repeat(1000)} onHover={this.onHover}
        sortIcon={sortIcon}
        keyName={paramKey} value={paramsMap[paramKey].getValue()} onHover={this.onHover}
        setSortByHandler={setSortByHandler} isMetric={false} isParam={true} isHovered={isHovered}
        onRemoveBagged={onRemoveBagged}/>);
    });
    if (this.shouldShowBaggedColumn(true)) {
      rowContents.push(
        <div key={"params-container-cell-" + runInfo.run_uuid} className="left-border">
          <div>{paramsCellContents}</div>
        </div>);
    }

    // Add metrics (unbagged, then bagged)
    unbaggedMetrics.forEach((metricKey) => {
      rowContents.push(
        ExperimentViewUtil.getUnbaggedMetricCell(metricKey, metricsMap, metricRanges, "div"));
    });

    // Add bagged metrics
    const metricsCellContents = baggedMetrics.map((metricKey) => {
    // const metricsCellContents = [...Array(100).keys()].map((metricKey) => {
      const keyname = "metric-" + metricKey;
      const isHovered = hoverState.isMetric && hoverState.key === metricKey;
      const sortIcon = ExperimentViewUtil.getSortIcon(sortState, true, false, metricKey);
      return (
        <BaggedCell key={keyname}
                    keyName={metricKey} value={metricsMap[metricKey].getValue().toString()} onHover={this.onHover}
                    // keyName={metricKey + "a".repeat(1000)} value={"b".repeat(1000)} onHover={this.onHover}
                    sortIcon={sortIcon}
                    setSortByHandler={setSortByHandler} isMetric={true} isParam={false} isHovered={isHovered}
                    onRemoveBagged={onRemoveBagged}/>
      );
    });
    if (this.shouldShowBaggedColumn(false)) {
      rowContents.push(
        <div key={"metrics-container-cell-" + runInfo.run_uuid} className="metric-param-container-cell left-border">
          {metricsCellContents}
        </div>
      );
    }

    const sortValue = ExperimentViewUtil.computeSortValue(
      sortState, metricsMap, paramsMap, runInfo, tagsMap);
    return {
      key: runInfo.run_uuid,
      sortValue,
      contents: rowContents,
      isChild: !isParent,
    };
  }

  getSortInfo(isMetric, isParam) {
    const { sortState, onSortBy } = this.props;
    const sortIcon = sortState.ascending ?
      <i className="fas fa-caret-up" style={styles.sortArrow}/> :
      <i className="fas fa-caret-down" style={styles.sortArrow}/>;
    if (sortState.isMetric === isMetric && sortState.isParam === isParam) {
      return (
        <span
          style={styles.sortToggle}
          onClick={() => onSortBy(isMetric, isParam, sortState.key)}
        >
        <span style={styles.sortKeyName} className="run-table-container">
          (sort: {sortState.key}
        </span>
          {sortIcon}
          <span>)</span>
      </span>);
    }
    return undefined;
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
      setSortByHandler,
      sortState,
      paramKeyList,
      metricKeyList,
      unbaggedMetrics,
      unbaggedParams,
      onAddBagged,
    } = this.props;
    const columns = [];
    const getHeaderCell = (isParam, key, i) => {
      const isMetric = !isParam;
      const sortIcon = ExperimentViewUtil.getSortIcon(sortState, isMetric, isParam, key);
      const className = classNames("bottom-row", { "left-border": i === 0 });
      const elemKey = (isParam ? "param-" : "metric-") + key;
      const keyContainerWidth = sortIcon ? "calc(100% - 20px)" : "100%";
      return (
        <div
          key={elemKey}
          className={className}
        >
          <span
            style={styles.metricParamHeaderContainer}
            className="run-table-container"
          >
            <Dropdown style={{width: "100%"}}>
              <ExperimentRunsSortToggle
                bsRole="toggle"
                className="metric-param-sort-toggle"
              >
                <span style={{maxWidth: keyContainerWidth, overflow: "hidden", display: "inline-block", verticalAlign: "middle"}}>{key}</span>
                <span style={ExperimentViewUtil.styles.sortIconContainer}>{sortIcon}</span>
              </ExperimentRunsSortToggle>
              <Dropdown.Menu className="mlflow-menu">
                <MenuItem
                  className="mlflow-menu-item"
                  onClick={() => setSortByHandler(!isParam, isParam, key, true)}
                >
                  Sort ascending
                </MenuItem>
                <MenuItem
                  className="mlflow-menu-item"
                  onClick={() => setSortByHandler(!isParam, isParam, key, false)}
                >
                  Sort descending
                </MenuItem>
                <MenuItem
                  className="mlflow-menu-item"
                  onClick={() => onAddBagged(isParam, key)}
                >
                  Collapse column
                </MenuItem>
              </Dropdown.Menu>
            </Dropdown>
          </span>
        </div>);
    };

    const paramClassName = classNames("bottom-row", {"left-border": unbaggedParams.length === 0});
    const metricClassName = classNames("bottom-row", {"left-border": unbaggedMetrics.length === 0});
    unbaggedParams.forEach((paramKey, i) => {
      columns.push(getHeaderCell(true, paramKey, i));
    });

    if (this.shouldShowBaggedColumn(true)) {
      columns.push(<div key="meta-bagged-params left-border" className={paramClassName}>
        {paramKeyList.length !== 0 ? "" : "(n/a)"}
      </div>);
    }
    unbaggedMetrics.forEach((metricKey, i) => {
      columns.push(getHeaderCell(false, metricKey, i));
    });
    if (this.shouldShowBaggedColumn(false)) {
      columns.push(<div key="meta-bagged-metrics left-border" className={metricClassName}>
        {metricKeyList.length !== 0 ? "" : "(n/a)"}
      </div>);
    }
    return columns;
  }

  _cache = new CellMeasurerCache({
    fixedWidth: true,
    minHeight: 25,
  });

  _lastRenderedWidth = this.props.width;
  _lastSortState = this.props.sortState;
  _lastRunsExpanded = this.props.runsExpanded;


  render() {
    const {
      runInfos,
      onCheckAll,
      isAllChecked,
      onSortBy,
      sortState,
      tagsList,
      runsExpanded,
      unbaggedMetrics,
      unbaggedParams,
    } = this.props;
    const rows = ExperimentViewUtil.getRows({
      runInfos,
      sortState,
      tagsList,
      runsExpanded,
      getRow: this.getRow });

    const headerCells = [
      ExperimentViewUtil.getSelectAllCheckbox(onCheckAll, isAllChecked, "div"),
      // placeholder for expander header cell,
      ExperimentViewUtil.getExpanderHeader("div"),
    ];
    ExperimentViewUtil.getRunMetadataHeaderCells(onSortBy, sortState, "div")
      .forEach((headerCell) => headerCells.push(headerCell));
    this.getMetricParamHeaderCells().forEach((cell) => headerCells.push(cell));

    // // Thought: Need to use this to render the header row, since you have two of them.
    // EDIT: react-virtualized only shows one row though.
    // const headerRowRenderer = ({
    //                              className,
    //                              columns,
    //                              style
    //                            }) => {
    //   // const topRowContents = [...Array(7).keys()].map(() => <div/>);
    //   // topRowContents.concat([<div>Parameters</div>, <div>Metrics</div>]);
    //   // const topRow = <div role="row" className={className}>
    //   //   {topRowContents}
    //   // </div>;
    //   const bottomRow = <div role="row" className={className}>{headerCells}</div>;
    //   // return [topRow, bottomRow];
    //   return bottomRow;
    // };
    return (
          <AutoSizer>
            {({width, height}) => {
              if (this._lastRenderedWidth !== width) {
                this._lastRenderedWidth = width;
                console.log("Clearing all!");
                this._cache.clearAll();
              }
              if (this._lastSortState !== sortState) {
                this._lastSortState = sortState;
                console.log("Clearing all because sort state changed!");
                this._cache.clearAll();
              }
              if (this._lastRunsExpanded !== runsExpanded) {
                this._lastRunsExpanded = runsExpanded;
                console.log("Clearing all because runs expanded changed!");
                this._cache.clearAll();
              }
              return (<Table
                width={width + unbaggedMetrics.length * 120 + unbaggedParams.length * 120}
                deferredMeasurementCache={this._cache}
                // height={height}
                height={500}
                headerHeight={32}
                overscanRowCount={2}
                // onRowsRendered={({ overscanStartIndex, overscanStopIndex, startIndex, stopIndex }) => {
                //   console.log("overscanStartIndex: " + overscanStartIndex);
                //   console.log("overscanStopIndex: " + overscanStopIndex);
                //   console.log("startIndex: " + startIndex);
                //   console.log("stopIndex: " + stopIndex);
                // }}
                rowHeight={this._cache.rowHeight}
                rowCount={rows.length}
                // overscanIndicesGetter={({
                //                           direction,          // One of "horizontal" or "vertical"
                //                           cellCount,          // Number of rows or columns in the current axis
                //                           scrollDirection,    // 1 (forwards) or -1 (backwards)
                //                           overscanCellsCount, // Maximum number of cells to over-render in either direction
                //                           startIndex,         // Begin of range of visible cells
                //                           stopIndex           // End of range of visible cells
                //                         }) => {
                //   const startIdx = Math.max(0, startIndex - overscanCellsCount);
                //   const endIdx = Math.min(stopIndex + overscanCellsCount, cellCount - 1);
                //   return {overscanStartIndex: startIdx, overscanStopIndex: endIdx};
                // }}
                rowGetter={({index}) => rows[index]}
                rowStyle={({index}) => {
                  // console.log("Row style for row " + index);
                  const borderStyle = "1px solid #e2e2e2";
                  const base = {alignItems: "stretch", borderBottom: borderStyle, overflow: "visible"};
                  if (index === - 1) {
                    return {...base, borderTop: borderStyle};
                  }
                  return base;
                }}
              >
                <Column
                  label='Checkbox'
                  dataKey='checkbox'
                  width={30}
                  headerRenderer={() => {
                    return headerCells[0]
                  }}
                  style={{display: "flex", alignItems: "flex-start", overflow: "visible", borderLeft: "1px gray"}}
                  cellRenderer={({rowIndex}) => {
                    return rows[rowIndex].contents[0];
                  }}
                />
                <Column
                  label='Expander'
                  dataKey='expander'
                  width={30}
                  headerRenderer={() => {
                  return headerCells[1]
                  }}
                  style={{display: "flex", alignItems: "flex-start", overflow: "visible", borderLeft: "1px gray"}}
                  cellRenderer={({rowIndex}) => {
                    return rows[rowIndex].contents[1];
                  }}
                />
                <Column
                  label='Date'
                  dataKey='date'
                  width={150}
                  headerRenderer={() => {
                    return headerCells[2]
                  }}
                  style={{display: "flex", alignItems: "flex-start", overflow: "visible", borderLeft: "1px gray"}}
                  flexShrink={0}
                  cellRenderer={({cellData, rowIndex, parent, dataKey}) => {
                    return rows[rowIndex].contents[1 + 1];
                  }}
                />
                <Column
                  label='User'
                  dataKey='user'
                  width={120}
                  headerRenderer={() => {
                    return headerCells[3]
                  }}
                  style={{display: "flex", alignItems: "flex-start", overflow: "visible", borderLeft: "1px gray"}}
                  cellRenderer={({rowIndex}) => {
                    return rows[rowIndex].contents[2 + 1];
                  }}
                />
                <Column
                  label='Run Name'
                  dataKey='name'
                  width={120}
                  headerRenderer={() => {
                    return headerCells[4]
                  }}
                  style={{display: "flex", alignItems: "flex-start", overflow: "visible", borderLeft: "1px gray"}}
                  cellRenderer={({rowIndex}) => {
                    return rows[rowIndex].contents[3 + 1];
                  }}
                />
                <Column
                  label='Source'
                  dataKey='source'
                  width={120}
                  headerRenderer={() => {
                    return headerCells[5]
                  }}
                  style={{display: "flex", alignItems: "flex-start", overflow: "visible", borderLeft: "1px gray"}}
                  cellRenderer={({rowIndex}) => {
                    return rows[rowIndex].contents[4 + 1];
                  }}
                />
                <Column
                  label='Version'
                  dataKey='version'
                  width={120}
                  headerRenderer={() => {
                    return headerCells[6]
                  }}
                  style={{display: "flex", alignItems: "flex-start", overflow: "visible", borderLeft: "1px gray"}}
                  cellRenderer={({rowIndex}) => {
                    return rows[rowIndex].contents[5 + 1];
                  }}
                />
                {unbaggedParams.map((unbaggedParam, idx) => {
                  return <Column
                    key={"param-" + unbaggedParam}
                    label={"param-" + unbaggedParam}
                    dataKey={"param-" + unbaggedParam}
                    width={120}
                    headerRenderer={() => {
                      // return <div>{unbaggedParam}</div>
                      return headerCells[7 + idx]
                    }}
                  style={{display: "flex", alignItems: "flex-start", overflow: "visible", borderLeft: "1px gray"}}
                    cellRenderer={({rowIndex}) => {
                      return rows[rowIndex].contents[7 + idx];
                    }}
                  />
                })}
                <Column
                  width={300}
                  label='Parameters'
                  dataKey='params'
                  headerRenderer={() => {
                    //return headerCells[7 + unbaggedParams.length]
                    return <div>Parameters</div>;
                  }}
                  style={{display: "flex", alignItems: "flex-start", overflow: "visible", borderLeft: "1px gray"}}
                  cellRenderer={({cellData, rowIndex, parent, dataKey}) => {
                    return (<CellMeasurer
                      cache={this._cache}
                      columnIndex={0}
                      key={dataKey}
                      parent={parent}
                      rowIndex={rowIndex}>
                      <div
                        style={{
                          whiteSpace: 'normal',
                        }}>
                        {rows[rowIndex].contents[7 + unbaggedParams.length]}
                      </div>
                    </CellMeasurer>);
                  }}
                />
                {unbaggedMetrics.map((unbaggedMetric, idx) => {
                  return <Column
                    key={"metric-" + unbaggedMetric}
                    label='Version'
                    dataKey={"metric-" + unbaggedMetric}
                    width={120}
                    headerRenderer={() => {
                      // return <div>{unbaggedMetric}</div>
                      return headerCells[8 + unbaggedParams.length + idx];
                    }}
                  style={{display: "flex", alignItems: "flex-start", overflow: "visible", borderLeft: "1px gray"}}
                    cellRenderer={({rowIndex}) => {
                      return rows[rowIndex].contents[8 + unbaggedParams.length + idx];
                    }}
                  />
                })}
                <Column
                  width={300}
                  label='Metrics'
                  dataKey='metrics'
                  headerRenderer={() => {
                    // return headerCells[8 + unbaggedParams.length + unbaggedMetrics.length]
                    return <div>Metrics</div>
                  }}
                  style={{display: "flex", alignItems: "flex-start", overflow: "visible", borderLeft: "1px gray"}}
                  cellRenderer={({cellData, rowIndex, parent, dataKey}) => {
                    return (<CellMeasurer
                      cache={this._cache}
                      columnIndex={1}
                      key={dataKey}
                      parent={parent}
                      rowIndex={rowIndex}>
                      <div
                        style={{
                          whiteSpace: 'normal',
                        }}>
                        {rows[rowIndex].contents[8 + unbaggedParams.length + unbaggedMetrics.length]}
                      </div>
                    </CellMeasurer>);
                  }}
                />
              </Table>);
            }}
          </AutoSizer>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { metricsList } = ownProps;
  return {metricRanges: ExperimentViewUtil.computeMetricRanges(metricsList)};
};

export default connect(mapStateToProps)(ExperimentRunsTableCompactView);
