import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Table from 'react-bootstrap/es/Table';
import ExperimentViewUtil from "./ExperimentViewUtil";

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
class ExperimentRunsTableCompact extends Component {
  static propTypes = {
    rows: PropTypes.arrayOf(PropTypes.object),
    onCheckAll: PropTypes.func.isRequired,
    isAllChecked: PropTypes.func.isRequired,
    onSortBy: PropTypes.func.isRequired,
    sortState: PropTypes.object.isRequired,
  };

  getSortInfo(isMetric, isParam) {
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
    const { rows, onCheckAll, isAllChecked, onSortBy, sortState } = this.props;
    const headerCells = [ExperimentViewUtil.getSelectAllCheckbox(onCheckAll, isAllChecked())];
    ExperimentViewUtil.getRunMetadataHeaderCells(onSortBy, sortState)
      .forEach((headerCell) => headerCells.push(headerCell));
    return (
      <Table hover>
      <colgroup span="7"/>
      <colgroup span="1"/>
      <colgroup span="1"/>
      <tbody>
      <tr>
          {headerCells}
          <th className="top-row left-border" scope="colgroup"
              colSpan="1">
            <div>Parameters</div>
            <div style={styles.sortContainer} className="unselectable">
              {this.getSortInfo(false, true)}
            </div>
          </th>
          <th className="top-row left-border" scope="colgroup"
              colSpan="1">
            <div>Metrics</div>
            <div style={styles.sortContainer} className="unselectable">
              {this.getSortInfo(true, false)}
            </div>
          </th>
      </tr>
      {rows.map(row => <tr key={row.key}>{row.contents}</tr>)}
      </tbody>
      </Table>);
  }
}

export default ExperimentRunsTableCompact;
