import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Table from 'react-bootstrap/es/Table';
import ExperimentViewUtil from './ExperimentViewUtil';
import { RunInfo } from '../sdk/MlflowMessages';
import Utils from '../utils/Utils';
import { Dropdown, MenuItem } from 'react-bootstrap';
import ExperimentRunsSortToggle from './ExperimentRunsSortToggle';

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
};

/**
 * Compact table view for displaying runs associated with an experiment. Renders metrics/params in
 * a single table cell per run (as opposed to one cell per metric/param).
 */
class ExperimentRunsTableCompactView extends Component {
  constructor(props) {
    super(props);
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

  getRow({ idx, isParent, hasExpander, expanderOpen, isHidden, childrenIds }) {
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
    const runInfo = runInfos[idx];
    const paramsMap = ExperimentViewUtil.toParamsMap(paramsList[idx]);
    const metricsMap = ExperimentViewUtil.toMetricsMap(metricsList[idx]);
    const metricRanges = ExperimentViewUtil.computeMetricRanges(metricsList);
    const selected = runsSelected[runInfo.run_uuid] === true;

    const rowContents = [
      ExperimentViewUtil.getCheckboxForRow(selected, () => onCheckbox(runInfo.run_uuid), isHidden),
      ExperimentViewUtil.getExpander(
         hasExpander, expanderOpen, () => onExpand(runInfo.run_uuid, childrenIds), isHidden),
    ];
    ExperimentViewUtil.getRunInfoCellsForRow(runInfo, tagsList[idx], isParent, isHidden)
      .forEach((col) => rowContents.push(col));
    const filteredParamKeys = paramKeyList.filter((paramKey) => paramsMap[paramKey] !== undefined);
    const paramsCellContents = filteredParamKeys.map((paramKey) => {
      const keyName = "param-" + paramKey;
      const cellClass = ExperimentViewUtil.isSortedBy(sortState, false, true, paramKey) ?
        "highlighted" : "";
      return (
        <div key={keyName} className="metric-param-cell">
          <Dropdown id="dropdown-custom-1">
            <ExperimentRunsSortToggle
              bsRole="toggle"
              className={"metric-param-sort-toggle " + cellClass}
            >
            <span className="run-table-container" style={{display: "inline-block"}}>
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
              style={{display: "inline-block"}}
              title={paramsMap[paramKey].getValue()}
            >
              {paramsMap[paramKey].getValue()}
          </span>
            <Dropdown.Menu className="mlflow-menu">
              <MenuItem
                className="mlflow-menu-item sort-run-menu-item"
                onClick={() => setSortByHandler(false, true, paramKey, true)}
              >
                Sort ascending ({paramKey})
              </MenuItem>
              <MenuItem
                className="mlflow-menu-item sort-run-menu-item"
                onClick={() => setSortByHandler(false, true, paramKey, false)}
              >
                Sort descending ({paramKey})
              </MenuItem>
            </Dropdown.Menu>
          </Dropdown>
        </div>
      );
    });
    rowContents.push(
      <td key="params-container-cell" className="left-border">{paramsCellContents}</td>);
    const filteredMetricKeys = metricKeyList.filter((key) => metricsMap[key] !== undefined);
    const metricsCellContents = filteredMetricKeys.map((metricKey) => {
      const keyName = "metric-" + metricKey;
      const cellClass = ExperimentViewUtil.isSortedBy(sortState, true, false, metricKey) ?
        "highlighted" : "";
      const metric = metricsMap[metricKey].getValue();
      const range = metricRanges[metricKey];
      let fraction = 1.0;
      if (range.max > range.min) {
        fraction = (metric - range.min) / (range.max - range.min);
      }
      const percent = (fraction * 100) + "%";
      return (
        <div key={keyName} className="metric-param-cell">
          <Dropdown id="dropdown-custom-1">
            <ExperimentRunsSortToggle
              bsRole="toggle"
              className={"metric-param-sort-toggle " + cellClass}
            >
              <span className="run-table-container" style={{display: "inline-block"}}>
                <span className="metric-param-name" title={metricKey}>
                  {metricKey}
                </span>
                <span>
                  :
                </span>
              </span>
            </ExperimentRunsSortToggle>
            <span className="metric-filler-bg metric-param-value">
              <span className="metric-filler-fg" style={{width: percent}}/>
              <span className="metric-text">
                {Utils.formatMetric(metric)}
              </span>
            </span>
            <Dropdown.Menu className="mlflow-menu">
              <MenuItem
                className="mlflow-menu-item sort-run-menu-item"
                onClick={() => setSortByHandler(true, false, metricKey, true)}
              >
                Sort ascending ({metricKey})
              </MenuItem>
              <MenuItem
                className="mlflow-menu-item sort-run-menu-item"
                onClick={() => setSortByHandler(true, false, metricKey, false)}
              >
                Sort descending ({metricKey})
              </MenuItem>
            </Dropdown.Menu>
          </Dropdown>
        </div>
      );
    });
    rowContents.push(
      <td key="metrics-container-cell" className="left-border">{metricsCellContents}</td>);

    const sortValue = ExperimentViewUtil.computeSortValue(
      sortState, metricsMap, paramsMap, runInfo, tagsList[idx]);

    return {
      key: runInfo.run_uuid,
      sortValue,
      contents: rowContents,
      isChild: !isParent,
      isHidden,
    };
  }

  getSortInfo({ isMetric, isParam }) {
    const { sortState, onSortBy } = this.props;
    const sortIcon = sortState.ascending ?
      <i className="fas fa-arrow-up" style={styles.sortArrow}/> :
      <i className="fas fa-arrow-down" style={styles.sortArrow}/>;
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
    return "";
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
      <colgroup span="8"/>
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
              {this.getSortInfo({ isMetric: false, isParam: true })}
            </div>
          </th>
          <th className="top-row left-border" scope="colgroup"
              colSpan="1">
            <div>Metrics</div>
            <div style={styles.sortContainer} className="unselectable">
              {this.getSortInfo({ isMetric: true, isParam: false })}
            </div>
          </th>
      </tr>
      {ExperimentViewUtil.renderRows(rows)}
      </tbody>
      </Table>);
  }
}

export default ExperimentRunsTableCompactView;
