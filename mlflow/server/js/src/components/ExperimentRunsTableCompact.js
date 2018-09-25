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
    const runMetadataHeaderCells = ExperimentViewUtil.sharedColumnHeaders(
      onSortBy, onCheckAll, isAllChecked(), sortState);
    return (
      <Table hover>
      <colgroup span="7"/>
      <colgroup span="1"/>
      <colgroup span="1"/>
      <tbody>
      <tr>
          {runMetadataHeaderCells}
          <th className="top-row left-border" scope="colgroup"
              colSpan="1">Parameters
          </th>
          <th className="top-row left-border" scope="colgroup"
              colSpan="1">Metrics
          </th>
          <tr>

          </tr>
      </tr>
      {rows.map(row => <tr key={row.key}>{row.contents}</tr>)}
      </tbody>
      </Table>);
  }
}

export default ExperimentRunsTableCompact;

