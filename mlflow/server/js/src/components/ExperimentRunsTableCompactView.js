import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Table from 'react-bootstrap/es/Table';
import ExperimentViewUtil from "./ExperimentViewUtil";
import { RunInfo } from '../sdk/MlflowMessages';
import classNames from 'classnames';
import { Dropdown, MenuItem } from 'react-bootstrap';
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
    paramKeyList: PropTypes.arrayOf(PropTypes.string).isRequired,
    metricKeyList: PropTypes.arrayOf(PropTypes.string).isRequired,
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
  };

  state = {
    hoverState: {isMetric: false, isParam: false, key: ""},
  };

  onHover({isParam, isMetric, key}) {
    this.setState({ hoverState: {isParam, isMetric, key} });
  }

  getRow({ idx, isParent, hasExpander, expanderOpen, childrenIds }) {
    const {
      runInfos,
      paramsList,
      metricsList,
      paramKeyList,
      metricKeyList,
      onCheckbox,
      sortState,
      runsSelected,
      tagsList,
      setSortByHandler,
      onExpand,
    } = this.props;
    const hoverState = this.state.hoverState;
    const runInfo = runInfos[idx];
    const paramsMap = ExperimentViewUtil.toParamsMap(paramsList[idx]);
    const metricsMap = ExperimentViewUtil.toMetricsMap(metricsList[idx]);
    const selected = runsSelected[runInfo.run_uuid] === true;
    const rowContents = [
      ExperimentViewUtil.getCheckboxForRow(selected, () => onCheckbox(runInfo.run_uuid)),
      ExperimentViewUtil.getExpander(
        hasExpander, expanderOpen, () => onExpand(runInfo.run_uuid, childrenIds)),
    ];
    ExperimentViewUtil.getRunInfoCellsForRow(runInfo, tagsList[idx], isParent)
      .forEach((col) => rowContents.push(col));
    const filteredParamKeys = paramKeyList.filter((paramKey) => paramsMap[paramKey] !== undefined);
    const paramsCellContents = filteredParamKeys.map((paramKey) => {
      const cellClass = classNames("metric-param-content",
        { highlighted: hoverState.isParam && hoverState.key === paramKey });
      const keyname = "param-" + paramKey;
      const sortIcon = ExperimentViewUtil.getSortIcon(sortState, true, false, paramKey);
      return (
        <div
          key={keyname}
          className="metric-param-cell"
          onMouseEnter={() => this.onHover({isParam: true, isMetric: false, key: paramKey})}
          onMouseLeave={() => this.onHover({isParam: false, isMetric: false, key: ""})}
        >
          <span className={cellClass}>
            <Dropdown id="dropdown-custom-1">
              <ExperimentRunsSortToggle
                bsRole="toggle"
                className="metric-param-sort-toggle"
              >
                <span
                  className="run-table-container"
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
              </Dropdown.Menu>
            </Dropdown>
          </span>
        </div>
      );
    });
    rowContents.push(
      <td key="params-container-cell" className="left-border"><div>{paramsCellContents}</div></td>);
    const filteredMetricKeys = metricKeyList.filter((key) => metricsMap[key] !== undefined);
    const metricsCellContents = filteredMetricKeys.map((metricKey) => {
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
                  className="run-table-container"
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
              </Dropdown.Menu>
            </Dropdown>
          </span>
        </span>
      );
    });
    rowContents.push(
      <td key="metrics-container-cell" className="left-border metric-param-container-cell">
        <div>
        {metricsCellContents}
        </div>
      </td>
    );

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

  render() {
    const {
      runInfos,
      onCheckAll,
      isAllChecked,
      onSortBy,
      sortState,
      tagsList,
      runsExpanded,
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
    return (
      <Table hover>
        <colgroup span="9"/>
        <colgroup span="1"/>
        <colgroup span="1"/>
        <tbody>
        <tr>
          {headerCells}
          <th
            className="top-row left-border"
            scope="colgroup"
            colSpan="1"
          >
            <div>Parameters</div>
            <div style={styles.sortContainer} className="unselectable">
              {this.getSortInfo(false, true)}
            </div>
          </th>
          <th
            className="top-row left-border"
            scope="colgroup"
            colSpan="1"
          >
            <div>Metrics</div>
            <div style={styles.sortContainer} className="unselectable">
              {this.getSortInfo(true, false)}
            </div>
          </th>
        </tr>
        {ExperimentViewUtil.renderRows(rows)}
        </tbody>
      </Table>);
  }
}

export default ExperimentRunsTableCompactView;
