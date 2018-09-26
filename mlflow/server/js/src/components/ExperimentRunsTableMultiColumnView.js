import React, { Component } from 'react';
import classNames from 'classnames';
import PropTypes from 'prop-types';
import Table from 'react-bootstrap/es/Table';
import ExperimentViewUtil from './ExperimentViewUtil';
import { RunInfo } from '../sdk/MlflowMessages';
import Utils from '../utils/Utils';

/**
 * Table view for displaying runs associated with an experiment. Renders each metric and param
 * value associated with a run in its own column.
 */
class ExperimentRunsTableMultiColumnView extends Component {
  static propTypes = {
    runInfos: PropTypes.arrayOf(RunInfo).isRequired,
    // List of list of params in all the visible runs
    paramsList: PropTypes.arrayOf(Array).isRequired,
    // List of list of metrics in all the visible runs
    metricsList: PropTypes.arrayOf(Array).isRequired,
    paramKeyList: PropTypes.arrayOf(PropTypes.string),
    metricKeyList: PropTypes.arrayOf(PropTypes.string),
    // List of tags dictionary in all the visible runs.
    tagsList: PropTypes.arrayOf(Object).isRequired,
    // Function which takes one parameter (runId)
    onCheckbox: PropTypes.func.isRequired,
    onCheckAll: PropTypes.func.isRequired,
    isAllChecked: PropTypes.bool.isRequired,
    onSortBy: PropTypes.func.isRequired,
    sortState: PropTypes.object.isRequired,
    runsSelected: PropTypes.object.isRequired,
  };

  getRows() {
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
    } = this.props;
    const metricRanges = ExperimentViewUtil.computeMetricRanges(metricsList);
    const rows = [...Array(runInfos.length).keys()].map((idx) => {
      const runInfo = runInfos[idx];
      const paramsMap = ExperimentViewUtil.toParamsMap(paramsList[idx]);
      const metricsMap = ExperimentViewUtil.toMetricsMap(metricsList[idx]);
      const numParams = paramKeyList.length;
      const numMetrics = metricKeyList.length;
      const selected = runsSelected[runInfo.run_uuid] === true;
      const rowContents = [ExperimentViewUtil.getCheckboxForRow(selected, onCheckbox)];
      ExperimentViewUtil.getRunInfoCellsForRow(runInfo, tagsList[idx]).forEach((col) =>
        rowContents.push(col));
      paramKeyList.forEach((paramKey, i) => {
        const className = (i === 0 ? "left-border" : "") + " run-table-container";
        const keyName = "param-" + paramKey;
        if (paramsMap[paramKey]) {
          rowContents.push(<td className={className} key={keyName}>
            {paramsMap[paramKey].getValue()}
          </td>);
        } else {
          rowContents.push(<td className={className} key={keyName}/>);
        }
      });
      if (numParams === 0) {
        rowContents.push(<td className="left-border" key={"meta-param-empty"}/>);
      }

      metricKeyList.forEach((metricKey, i) => {
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
              <div className="metric-filler-bg">
                <div className="metric-filler-fg" style={{width: percent}}/>
                <div className="metric-text">
                  {Utils.formatMetric(metric)}
                </div>
              </div>
            </td>
          );
        } else {
          rowContents.push(<td className={className} key={keyName}/>);
        }
      });
      if (numMetrics === 0) {
        rowContents.push(<td className="left-border" key="meta-metric-empty" />);
      }

      const sortValue = ExperimentViewUtil.computeSortValue(
        sortState, metricsMap, paramsMap, runInfo, tagsList[idx]);
      return {
        key: runInfo.run_uuid,
        sortValue: sortValue,
        contents: rowContents,
      };
    });
    ExperimentViewUtil.sortRows(rows, sortState);
    return rows;
  }

  getMetricParamHeaderCells() {
    const {
      paramKeyList,
      metricKeyList,
      onSortBy,
      sortState
    } = this.props;
    const numParams = paramKeyList.length;
    const numMetrics = metricKeyList.length;
    const columns = [];
    paramKeyList.forEach((paramKey, i) => {
      const className = "bottom-row "
        + "run-table-container "
        + (i === 0 ? "left-border " : "")
        + ExperimentViewUtil.sortedClassName(sortState, false, true, paramKey);
      columns.push(
        <th
          key={'param-' + paramKey}
          className={className}
          onClick={() => onSortBy(false, true, paramKey)}
        >
          {paramKey}
        </th>
      );
    });
    if (numParams === 0) {
      columns.push(<th key="meta-param-empty" className="bottom-row left-border">(n/a)</th>);
    }

    let firstMetric = true;
    metricKeyList.forEach((metricKey) => {
      const className = classNames(
        "bottom-row",
        "run-table-container",
        {"left-border": firstMetric},
         ExperimentViewUtil.sortedClassName(sortState, true, false, metricKey));
      firstMetric = false;
      columns.push(<th key={'metric-' + metricKey} className={className}
                       onClick={() => onSortBy(true, false, metricKey)}>{metricKey}</th>);
    });
    if (numMetrics === 0) {
      columns.push(<th key="meta-metric-empty" className="bottom-row left-border">(n/a)</th>);
    }
    return columns;
  };

  render() {
    const {
      paramKeyList,
      metricKeyList,
      onCheckAll,
      isAllChecked,
      onSortBy,
      sortState } = this.props;
    const rows = this.getRows();
    const columns = [ExperimentViewUtil.getSelectAllCheckbox(onCheckAll, isAllChecked)];
    ExperimentViewUtil.getRunMetadataHeaderCells(onSortBy, sortState).forEach((cell) =>
      columns.push(cell));
    this.getMetricParamHeaderCells().forEach((cell) => columns.push(cell));
    return (
      <Table hover>
        <colgroup span="7"/>
        <colgroup span={paramKeyList.length}/>
        <colgroup span={metricKeyList.length}/>
        <tbody>
        <tr>
          <th className="top-row" scope="colgroup" colSpan="5"/>
          <th className="top-row left-border" scope="colgroup" colSpan={paramKeyList.length}>
            Parameters
          </th>
          <th className="top-row left-border" scope="colgroup" colSpan={metricKeyList.length}>
            Metrics
          </th>
        </tr>
        <tr>
          {columns}
        </tr>
        {rows.map(row => <tr key={row.key}>{row.contents}</tr>)}
        </tbody>
      </Table>
    );
  }
}

export default ExperimentRunsTableMultiColumnView;
