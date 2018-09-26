import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Table from 'react-bootstrap/es/Table';
import ExperimentViewUtil from "./ExperimentViewUtil";

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

  render() {
    const { rows, onCheckAll, isAllChecked, onSortBy, sortState } = this.props;
    const headerCells = [ExperimentViewUtil.getSelectAllCheckbox(onCheckAll, isAllChecked())];
    ExperimentViewUtil.getRunMetadataHeaderCells(onSortBy, sortState)
      .forEach((headerCell) => headerCells.push(headerCell));
    const sortIcon = sortState.ascending ?
      <i className="fas fa-arrow-up"/> : <i className="fas fa-arrow-down"/>;
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
            <div style={{minHeight: 18}}>
              {
                sortState.isParam ?
                  <span style={{cursor: "pointer"}}
                        onClick={() => onSortBy(false, true, sortState.key)}>
                    <span> (sort: {sortState.key}</span>
                    {sortIcon}
                    <span>)</span>
                  </span> :
                  ""
              }
            </div>
          </th>
          <th className="top-row left-border" scope="colgroup"
              colSpan="1">Metrics
            <div style={{minHeight: 18}}>
              {
                sortState.isMetric ?
                  <span style={{cursor: "pointer"}}
                        onClick={() => onSortBy(true, false, sortState.key)}>
                    <span>(sort: {sortState.key} </span>
                    {sortIcon}
                    <span>)</span>
                  </span> :
                  ""
              }
            </div>
          </th>
      </tr>
      {rows.map(row => <tr key={row.key}>{row.contents}</tr>)}
      </tbody>
      </Table>);
  }
}

export default ExperimentRunsTableCompact;

