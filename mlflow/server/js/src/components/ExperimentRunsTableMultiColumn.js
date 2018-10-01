import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Table from 'react-bootstrap/es/Table';
import ExperimentViewUtil from './ExperimentViewUtil';

/**
 * Table view for displaying runs associated with an experiment. Renders each metric and param
 * value associated with a run in its own column.
 */
class ExperimentRunsTableMultiColumn extends Component {
  static propTypes = {
    rows: PropTypes.arrayOf(PropTypes.object),
    paramKeyList: PropTypes.arrayOf(PropTypes.string),
    metricKeyList: PropTypes.arrayOf(PropTypes.string),
    onCheckAll: PropTypes.func.isRequired,
    isAllChecked: PropTypes.func.isRequired,
    onSortBy: PropTypes.func.isRequired,
    sortState: PropTypes.object.isRequired,
  };

  render() {
    const { paramKeyList, metricKeyList, rows, onCheckAll, isAllChecked, onSortBy,
      sortState } = this.props;
    const columns = [ExperimentViewUtil.getSelectAllCheckbox(onCheckAll, isAllChecked())];
    ExperimentViewUtil.getRunMetadataHeaderCells(onSortBy, sortState)
      .forEach((cell) => columns.push(cell));
    this.getMetricParamHeaderCells(paramKeyList, metricKeyList, onSortBy, sortState)
      .forEach((cell) => columns.push(cell));
    return (<Table hover>
      <colgroup span="7"/>
      <colgroup span={paramKeyList.length}/>
      <colgroup span={metricKeyList.length}/>
      <tbody>
      <tr>
        <th className="top-row" scope="colgroup" colSpan="5"></th>
        <th className="top-row left-border" scope="colgroup"
          colSpan={paramKeyList.length}>Parameters
        </th>
        <th className="top-row left-border" scope="colgroup"
          colSpan={metricKeyList.length}>Metrics
        </th>
      </tr>
      <tr>
        {columns}
      </tr>
      {rows.map(row => <tr key={row.key}>{row.contents}</tr>)}
      </tbody>
    </Table>);
  }

  getMetricParamHeaderCells(
    paramKeyList,
    metricKeyList,
    onSortBy,
    sortState) {
    const numParams = paramKeyList.length;
    const numMetrics = metricKeyList.length;
    const columns = [];
    paramKeyList.forEach((paramKey, i) => {
      const sortIcon = ExperimentViewUtil.getSortIcon(sortState, false, true, paramKey);
      const className = "bottom-row "
        + "run-table-container "
        + "sortable "
        + (i === 0 ? "left-border " : "");
      columns.push(
        <th
          key={'param-' + paramKey} className={className}
          onClick={() => onSortBy(false, true, paramKey)}
        >
          {paramKey}
          <span style={{marginLeft: 2}}>{sortIcon}</span>
        </th>);
    });
    if (numParams === 0) {
      columns.push(<th key="meta-param-empty" className="bottom-row left-border">(n/a)</th>);
    }

    let firstMetric = true;
    metricKeyList.forEach((metricKey) => {
      const className = "bottom-row "
        + "run-table-container "
        + "sortable "
        + (firstMetric ? "left-border " : "");
      firstMetric = false;
      const sortIcon = ExperimentViewUtil.getSortIcon(sortState, true, false, metricKey);
      columns.push(
        <th
          key={'metric-' + metricKey} className={className}
          onClick={() => onSortBy(true, false, metricKey)}
        >
          {metricKey}
          <span style={{marginLeft: 2}}>{sortIcon}</span>
        </th>);
    });
    if (numMetrics === 0) {
      columns.push(<th key="meta-metric-empty" className="bottom-row left-border">(n/a)</th>);
    }
    return columns;
  }
}
export default ExperimentRunsTableMultiColumn;
