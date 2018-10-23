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
  };

  state = {
    hoverState: {isMetric: false, isParam: false, key: ""},
    unbaggedMetrics: [],
    unbaggedParams: [],
  };

  onHover({isParam, isMetric, key}) {
    this.setState({ hoverState: {isParam, isMetric, key} });
  }

  // Mark a column as "bagged" by removing it from the unbagged array
  addBagged(isParam, colName) {
    const unbagged = isParam ? this.state.unbaggedParams : this.state.unbaggedMetrics;
    const idx = unbagged.indexOf(colName);
    const newUnbagged = idx >= 0 ?
      unbagged.slice(0, idx).concat(unbagged.slice(idx + 1, unbagged.length)) : unbagged;
    const stateKey = isParam ? "unbaggedParams" : "unbaggedMetrics";
    this.setState({[stateKey]: newUnbagged});
  }

  // Split out a column (add it to array of unbagged cols)
  removeBagged(isParam, colName) {
    const unbagged = isParam ? this.state.unbaggedParams : this.state.unbaggedMetrics;
    const stateKey = isParam ? "unbaggedParams" : "unbaggedMetrics";
    this.setState({[stateKey]: unbagged.concat([colName])});
  }

  // Builds a single row of table content (not header)
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
    } = this.props;
    const {
      unbaggedParams,
      unbaggedMetrics,
    } = this.state;
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
    const baggedParams = paramKeyList.filter((elem) => !unbaggedParamSet.has(elem));
    const baggedMetrics = metricKeyList.filter((elem) => !unbaggedMetricSet.has(elem));

    // Add params (unbagged, then bagged)
    unbaggedParams.forEach((paramKey, i) => {
      const className = (i === 0 ? "left-border" : "") + " run-table-container";
      const keyName = "param-" + paramKey;
      if (paramsMap[paramKey]) {
        rowContents.push(<td className={className} key={keyName}>
          <div>
            {paramsMap[paramKey].getValue()}
          </div>
        </td>);
      } else {
        rowContents.push(<td className={className} key={keyName}/>);
      }
    });
    // Add bagged params
    const filteredParamKeys = baggedParams.filter((paramKey) => paramsMap[paramKey] !== undefined);
    const paramsCellContents = filteredParamKeys.map((paramKey) => {
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
                  onClick={() => this.removeBagged(true, paramKey)}
                >
                  Display in own column
                </MenuItem>
              </Dropdown.Menu>
            </Dropdown>
          </span>
        </div>
      );
    });
    const baggedParamsClassname =
      classNames({"left-border": unbaggedParams.length !== paramKeyList.length});
    rowContents.push(
      <td key="params-container-cell" className={baggedParamsClassname}>
        <div>{paramsCellContents}</div>
      </td>);

    // Add metrics (unbagged, then bagged)
    unbaggedMetrics.forEach((metricKey, i) => {
      const className = (i === 0 ? "left-border" : "") + " run-table-container";
      const keyName = "metric-" + metricKey;
      if (metricsMap[metricKey]) {
        const metric = metricsMap[metricKey].getValue();
        const range = metricRanges[metricKey];
        let fraction = 1.0;
        if (range.max > range.min) {
          fraction = (metric - range.min) / (range.max - range.min);
        }
        const percent = (fraction * 100) + "%";
        rowContents.push(
          <td className={className} key={keyName}>
            {/* We need the extra div because metric-filler-bg is inline-block */}
            <div>
              <div className="metric-filler-bg">
                <div className="metric-filler-fg" style={{width: percent}}/>
                <div className="metric-text">
                  {Utils.formatMetric(metric)}
                </div>
              </div>
            </div>
          </td>
        );
      } else {
        rowContents.push(<td className={className} key={keyName}/>);
      }
    });

    // Add bagged metrics
    const filteredMetricKeys = baggedMetrics.filter((key) => metricsMap[key] !== undefined);
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
                  onClick={() => this.removeBagged(false, metricKey)}
                >
                  Display in own column
                </MenuItem>
              </Dropdown.Menu>
            </Dropdown>
          </span>
        </span>
      );
    });
    const baggedMetricsClassname = classNames("metric-param-container-cell",
      {"left-border": unbaggedMetrics.length !== metricKeyList.length});
    rowContents.push(
      <td key="metrics-container-cell" className={baggedMetricsClassname}>
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

  getMetricParamHeaderCells() {
    const {
      setSortByHandler,
      sortState,
      paramKeyList,
      metricKeyList,
    } = this.props;
    const {
      unbaggedParams,
      unbaggedMetrics,
    } = this.state;
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
                  onClick={() => this.addBagged(isParam, key)}
                >
                  Collapse column
                </MenuItem>
              </Dropdown.Menu>
            </Dropdown>
          </span>
        </th>);
    };

    const paramClassName = classNames("bottom-row",
      { "left-border": unbaggedParams.length === 0 });
    const metricClassName = classNames("bottom-row",
      { "left-border": unbaggedMetrics.length === 0 });
    unbaggedParams.forEach((paramKey, i) => {
      columns.push(getHeaderCell(true, paramKey, i));
    });

    const hasBaggedParams = unbaggedParams.length !== paramKeyList.length;
    columns.push(<th key="meta-bagged-params left-border" className={paramClassName}>
      {hasBaggedParams ? this.getBaggedHeaderDropdown(true) : ""}
    </th>);
    unbaggedMetrics.forEach((metricKey, i) => {
      columns.push(getHeaderCell(false, metricKey, i));
    });
    const hasBaggedMetrics = unbaggedMetrics.length !== metricKeyList.length;
    columns.push(<th key="meta-bagged-metrics left-border" className={metricClassName}>
      {hasBaggedMetrics ? this.getBaggedHeaderDropdown(false) : ""}
    </th>);
    return columns;
  }

  getBaggedHeaderDropdown(isParam) {
    // TODO rename toggle component to something appropriate, like EmptyToggle or something
    const stateKeyToUpdate = isParam ? "unbaggedParams" : "unbaggedMetrics";
    const keyList = isParam ? this.props.paramKeyList : this.props.metricKeyList;
    return <Dropdown id={stateKeyToUpdate + "-header"}>
      <ExperimentRunsSortToggle
        bsRole="toggle"
        className="metric-param-sort-toggle"
      >
        <i className="fas fa-cog" style={{cursor: "pointer", marginLeft: 10, fontSize: '1.5em'}}/>
      </ExperimentRunsSortToggle>
      <Dropdown.Menu className="mlflow-menu">
        <MenuItem
          className="mlflow-menu-item"
          onClick={() => this.setState({[stateKeyToUpdate]: keyList.slice(0, keyList.length)})}
        >
          Display all in columns
        </MenuItem>
        <MenuItem
          className="mlflow-menu-item"
          onClick={() => this.setState({[stateKeyToUpdate]: []})}
        >
          Collapse all columns
        </MenuItem>
      </Dropdown.Menu>
    </Dropdown>;
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
      paramKeyList,
      metricKeyList,
    } = this.props;
    const {
      unbaggedMetrics,
      unbaggedParams,
    } = this.state;
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
    const noBaggedParams = unbaggedParams.length === paramKeyList.length;
    const noBaggedMetrics = unbaggedMetrics.length === metricKeyList.length;
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

            colSpan={unbaggedParams.length + 1}
          >
            Parameters
            { noBaggedParams ? this.getBaggedHeaderDropdown(true) : ""}
          </th>
          <th className="top-row left-border" scope="colgroup"
            colSpan={unbaggedMetrics.length + 1}
          >
            Metrics
            { noBaggedMetrics ? this.getBaggedHeaderDropdown(false) : ""}
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
