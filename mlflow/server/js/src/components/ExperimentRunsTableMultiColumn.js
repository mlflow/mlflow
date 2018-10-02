import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Table from 'react-bootstrap/es/Table';
import ExperimentViewUtil from './ExperimentViewUtil';
import classNames from 'classnames';

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
        <th
          className="top-row left-border"
          scope="colgroup"
          colSpan={paramKeyList.length}
        >
          Parameters
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

    const getHeaderCell = (isParam, key, i) => {
      const isMetric = !isParam;
      const sortIcon = ExperimentViewUtil.getSortIcon(sortState, isMetric, isParam, key);
      const className = classNames("bottom-row", "sortable", { "left-border": i === 0 });
      const elemKey = (isParam ? "param-" : "metric-") + key;
      return (
        <th
          key={elemKey} className={className}
          onClick={() => onSortBy(isMetric, isParam, key)}
        >
          <span
            style={styles.metricParamNameContainer}
            className="run-table-container"
          >
            {key}
          </span>
          <span style={styles.sortIconContainer}>{sortIcon}</span>
        </th>);
    };

    paramKeyList.forEach((paramKey, i) => {
      columns.push(getHeaderCell(true, paramKey, i));
    });
    if (numParams === 0) {
      columns.push(<th key="meta-param-empty" className="bottom-row left-border">(n/a)</th>);
    }

    metricKeyList.forEach((metricKey, i) => {
      columns.push(getHeaderCell(false, metricKey, i));
    });
    if (numMetrics === 0) {
      columns.push(<th key="meta-metric-empty" className="bottom-row left-border">(n/a)</th>);
    }
    return columns;
  }
}

const styles = {
  sortIconContainer: {
    marginLeft: 2,
    minWidth: 12.5,
    display: 'inline-block',
  },
  metricParamNameContainer: {
    verticalAlign: "middle",
    display: "inline-block",
  },
};

export default ExperimentRunsTableMultiColumn;
