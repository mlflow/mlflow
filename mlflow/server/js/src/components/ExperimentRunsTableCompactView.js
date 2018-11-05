import React, { Component } from 'react';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import ExperimentViewUtil from "./ExperimentViewUtil";
import { RunInfo } from '../sdk/MlflowMessages';
import classNames from 'classnames';
import { Table, Dropdown, MenuItem } from 'react-bootstrap';
import ExperimentRunsSortToggle from './ExperimentRunsSortToggle';
import Utils from '../utils/Utils';

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
  },
};

/**
 * Compact table view for displaying runs associated with an experiment. Renders metrics/params in
 * a single table cell per run (as opposed to one cell per metric/param).
 */
class ExperimentRunsTableCompactView extends Component {
  constructor(props) {
    super(props);
    this.onHover = this.onHover.bind(this);
    this.getRow = this.getRow.bind(this);
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
    const hoverState = this.state.hoverState;
    const runInfo = runInfos[idx];
    const paramsMap = ExperimentViewUtil.toParamsMap(paramsList[idx]);
    const metricsMap = ExperimentViewUtil.toMetricsMap(metricsList[idx]);
    const selected = runsSelected[runInfo.run_uuid] === true;
    const rowContents = [
      ExperimentViewUtil.getCheckboxForRow(selected, () => onCheckbox(runInfo.run_uuid)),
      ExperimentViewUtil.getExpander(
        hasExpander, expanderOpen, () => onExpand(runInfo.run_uuid, childrenIds), runInfo.run_uuid),
    ];
    ExperimentViewUtil.getRunInfoCellsForRow(runInfo, tagsList[idx], isParent)
      .forEach((col) => rowContents.push(col));

    const unbaggedParamSet = new Set(unbaggedParams);
    const unbaggedMetricSet = new Set(unbaggedMetrics);
    const baggedParams = paramKeyList.filter((paramKey) =>
      !unbaggedParamSet.has(paramKey) && paramsMap[paramKey] !== undefined);
    const baggedMetrics = metricKeyList.filter((metricKey) =>
      !unbaggedMetricSet.has(metricKey) && metricsMap[metricKey] !== undefined);

    // Add params (unbagged, then bagged)
    unbaggedParams.forEach((paramKey) => {
      rowContents.push(ExperimentViewUtil.getUnbaggedParamCell(paramKey, paramsMap));
    });
    // Add bagged params
    const paramsCellContents = baggedParams.map((paramKey) => {
      const cellClass = classNames("metric-param-content",
        { highlighted: hoverState.isParam && hoverState.key === paramKey });
      const keyname = "param-" + paramKey;
      const sortIcon = ExperimentViewUtil.getSortIcon(sortState, false, true, paramKey);
      return (
        <div
          key={keyname}
          className="metric-param-cell"
        >
          <span
            className={cellClass}
            onMouseEnter={() => this.onHover({isParam: true, isMetric: false, key: paramKey})}
            onMouseLeave={() => this.onHover({isParam: false, isMetric: false, key: ""})}
          >
            <Dropdown id="dropdown-custom-1">
              <ExperimentRunsSortToggle
                bsRole="toggle"
                className="metric-param-sort-toggle"
              >
                <span
                  className="run-table-container underline-on-hover"
                  style={styles.metricParamCellContent}
                >
                  <span style={{marginRight: sortIcon ? 2 : 0}}>
                    {sortIcon}
                  </span>
                  <span className="metric-param-name" title={paramKey}>
                    {paramKey}
                  </span>
                  <span>
                    :
                  </span>
                </span>
              </ExperimentRunsSortToggle>
              <span
                className="metric-param-value run-table-container"
                style={styles.metricParamCellContent}
                title={paramsMap[paramKey].getValue()}
              >
                  {paramsMap[paramKey].getValue()}
              </span>
              <Dropdown.Menu className="mlflow-menu">
                <MenuItem
                  className="mlflow-menu-item"
                  onClick={() => setSortByHandler(false, true, paramKey, true)}
                >
                  Sort ascending
                </MenuItem>
                <MenuItem
                  className="mlflow-menu-item"
                  onClick={() => setSortByHandler(false, true, paramKey, false)}
                >
                  Sort descending
                </MenuItem>
                <MenuItem
                  className="mlflow-menu-item"
                  onClick={() => onRemoveBagged(true, paramKey)}
                >
                  Display in own column
                </MenuItem>
              </Dropdown.Menu>
            </Dropdown>
          </span>
        </div>
      );
    });
    if (this.shouldShowBaggedColumn(true)) {
      rowContents.push(
        <td key="params-container-cell" className="left-border">
          <div>{paramsCellContents}</div>
        </td>);
    }

    // Add metrics (unbagged, then bagged)
    unbaggedMetrics.forEach((metricKey) => {
      rowContents.push(
        ExperimentViewUtil.getUnbaggedMetricCell(metricKey, metricsMap, metricRanges));
    });

    // Add bagged metrics
    const metricsCellContents = baggedMetrics.map((metricKey) => {
      const keyname = "metric-" + metricKey;
      const cellClass = classNames("metric-param-content",
        { highlighted: hoverState.isMetric && hoverState.key === metricKey });
      const sortIcon = ExperimentViewUtil.getSortIcon(sortState, true, false, metricKey);
      const metric = metricsMap[metricKey].getValue();
      return (
        <span
          key={keyname}
          className={"metric-param-cell"}
          onMouseEnter={() => this.onHover({isParam: false, isMetric: true, key: metricKey})}
          onMouseLeave={() => this.onHover({isParam: false, isMetric: false, key: ""})}
        >
          <span className={cellClass}>
            <Dropdown id="dropdown-custom-1">
              <ExperimentRunsSortToggle
                bsRole="toggle"
                className={"metric-param-sort-toggle"}
              >
                <span
                  className="run-table-container underline-on-hover"
                  style={styles.metricParamCellContent}
                >
                  <span style={{marginRight: sortIcon ? 2 : 0}}>
                    {sortIcon}
                  </span>
                  <span className="metric-param-name" title={metricKey}>
                    {metricKey}
                  </span>
                  <span>
                    :
                  </span>
                </span>
              </ExperimentRunsSortToggle>
              <span
                className="metric-param-value run-table-container"
                style={styles.metricParamCellContent}
              >
                {Utils.formatMetric(metric)}
              </span>
              <Dropdown.Menu className="mlflow-menu">
                <MenuItem
                  className="mlflow-menu-item"
                  onClick={() => setSortByHandler(true, false, metricKey, true)}
                >
                  Sort ascending
                </MenuItem>
                <MenuItem
                  className="mlflow-menu-item"
                  onClick={() => setSortByHandler(true, false, metricKey, false)}
                >
                  Sort descending
                </MenuItem>
                <MenuItem
                  className="mlflow-menu-item"
                  onClick={() => onRemoveBagged(false, metricKey)}
                >
                  Display in own column
                </MenuItem>
              </Dropdown.Menu>
            </Dropdown>
          </span>
        </span>
      );
    });
    if (this.shouldShowBaggedColumn(false)) {
      rowContents.push(
        <td key="metrics-container-cell" className="metric-param-container-cell left-border">
          <div>
            {metricsCellContents}
          </div>
        </td>
      );
    }
    const sortValue = ExperimentViewUtil.computeSortValue(
      sortState, metricsMap, paramsMap, runInfo, tagsList[idx]);
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
      return (
        <th
          key={elemKey} className={className}
        >
          <span
            style={styles.metricParamNameContainer}
            className="run-table-container"
          >
            <Dropdown id="dropdown-custom-1">
              <ExperimentRunsSortToggle
                bsRole="toggle"
                className="metric-param-sort-toggle"
              >
                {key}
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
        </th>);
    };

    const paramClassName = classNames("bottom-row", {"left-border": unbaggedParams.length === 0});
    const metricClassName = classNames("bottom-row", {"left-border": unbaggedMetrics.length === 0});
    unbaggedParams.forEach((paramKey, i) => {
      columns.push(getHeaderCell(true, paramKey, i));
    });

    if (this.shouldShowBaggedColumn(true)) {
      columns.push(<th key="meta-bagged-params left-border" className={paramClassName}>
        {paramKeyList.length !== 0 ? "" : "(n/a)"}
      </th>);
    }
    unbaggedMetrics.forEach((metricKey, i) => {
      columns.push(getHeaderCell(false, metricKey, i));
    });
    if (this.shouldShowBaggedColumn(false)) {
      columns.push(<th key="meta-bagged-metrics left-border" className={metricClassName}>
        {metricKeyList.length !== 0 ? "" : "(n/a)"}
      </th>);
    }
    return columns;
  }

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
      ExperimentViewUtil.getSelectAllCheckbox(onCheckAll, isAllChecked),
      // placeholder for expander header cell,
      ExperimentViewUtil.getExpanderHeader(),
    ];
    ExperimentViewUtil.getRunMetadataHeaderCells(onSortBy, sortState)
      .forEach((headerCell) => headerCells.push(headerCell));
    this.getMetricParamHeaderCells().forEach((cell) => headerCells.push(cell));
    return (
      <Table hover>
        <colgroup span="9"/>
        <colgroup span={unbaggedMetrics.length}/>
        <colgroup span={unbaggedParams.length}/>
        <tbody>
        <tr>
          <th className="top-row" scope="colgroup" colSpan="7"/>
          <th
            className="top-row left-border"
            scope="colgroup"

            colSpan={unbaggedParams.length + this.shouldShowBaggedColumn(true)}
          >
            Parameters
          </th>
          <th className="top-row left-border" scope="colgroup"
            colSpan={unbaggedMetrics.length + this.shouldShowBaggedColumn(false)}
          >
            Metrics
          </th>
        </tr>
        <tr>
          {headerCells}
        </tr>
        {ExperimentViewUtil.renderRows(rows)}
        </tbody>
      </Table>);
  }
}

const mapStateToProps = (state, ownProps) => {
  const { metricsList } = ownProps;
  return {metricRanges: ExperimentViewUtil.computeMetricRanges(metricsList)};
};

export default connect(mapStateToProps)(ExperimentRunsTableCompactView);
