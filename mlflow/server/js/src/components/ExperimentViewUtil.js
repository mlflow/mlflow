import React from 'react';
import Utils from "../utils/Utils";
import { Link } from 'react-router-dom';
import Routes from '../Routes';

export default class ExperimentViewUtil {
  /** Returns checkbox cell for a row. */
  static getCheckboxForRow(selected, checkboxHandler) {
    return <td key="meta-check">
      <input type="checkbox" checked={selected} onClick={checkboxHandler}/>
    </td>;
  }

  /**
   * Returns table cells describing run metadata (i.e. not params/metrics) comprising part of
   * the display row for a run.
   */
  static getRunInfoCellsForRow(runInfo, tags) {
    const user = Utils.formatUser(runInfo.user_id);
    const sourceType = Utils.renderSource(runInfo, tags);
    return [
      <td key="meta-link" className="run-table-container">
        <Link to={Routes.getRunPageRoute(runInfo.experiment_id, runInfo.run_uuid)}>
          {runInfo.start_time ? Utils.formatTimestamp(runInfo.start_time) : '(unknown)'}
        </Link>
      </td>,
      <td key="meta-user" className="run-table-container" title={user}>{user}</td>,
      <td className="run-table-container" key="meta-source" title={sourceType}>
        {Utils.renderSourceTypeIcon(runInfo.source_type)}
        {sourceType}
      </td>,
      <td className="run-table-container" key="meta-version">{Utils.renderVersion(runInfo)}</td>,
    ];
  }

  /** Returns checkbox element for selecting all runs */
  static getSelectAllCheckbox(onCheckAll, isAllCheckedBool) {
    return <th key="meta-check" className="bottom-row">
      <input type="checkbox" onChange={onCheckAll} checked={isAllCheckedBool} />
    </th>;
  }

  /**
   * Returns header-row table cells for columns containing run metadata.
   */
  static getRunMetadataHeaderCells(onSortBy, sortState) {
    const getHeaderCell = (key, text) => {
      const sortedClassName = ExperimentViewUtil.sortedClassName(sortState, false, false, key)
        + " run-table-container";
      return <th key={"meta-" + key}
                 className={"bottom-row " + sortedClassName}
                 onClick={() => onSortBy(false, false, key)}>{text}</th>;
    };
    return [
      getHeaderCell("start_time", <span>{"Date"}</span>),
      getHeaderCell("user_id", <span>{"User"}</span>),
      getHeaderCell("source", <span>{"Source"}</span>),
      getHeaderCell("source_version", <span>{"Version"}</span>),
    ];
  }

  static isSortedBy(sortState, isMetric, isParam, key) {
    return (sortState.isMetric === isMetric && sortState.isParam === isParam
      && sortState.key === key);
  }

  /** Returns a classname for a sortable column (a run metadata column, or a metric/param column) */
  static sortedClassName = (sortState, isMetric, isParam, key) => {
    if (!ExperimentViewUtil.isSortedBy(sortState, isMetric, isParam, key)) {
      return "sortable";
    }
    return "sortable sorted " + (sortState.ascending ? "asc" : "desc");
  };
}
