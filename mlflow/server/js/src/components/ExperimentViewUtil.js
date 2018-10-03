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

  static styles = {
    sortIconStyle: {
      verticalAlign: "middle",
      fontSize: 20,
    },
    headerCellText: {
      verticalAlign: "middle"
    },
    sortIconContainer: {
      marginLeft: 2,
      minWidth: 12.5,
      display: 'inline-block',
    },
  };

  /**
   * Returns table cells describing run metadata (i.e. not params/metrics) comprising part of
   * the display row for a run.
   */
  static getRunInfoCellsForRow(runInfo, tags) {
    const user = Utils.formatUser(runInfo.user_id);
    const sourceType = Utils.renderSource(runInfo, tags);
    return [
      <td key="meta-link" className="run-table-container" style={{whiteSpace: "inherit"}}>
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

  /**
   * Returns an icon for sorting the metric or param column with the specified key. The icon
   * is visible if we're currently sorting by the corresponding column. Otherwise, the icon is
   * invisible but takes up space.
   */
  static getSortIcon(sortState, isMetric, isParam, key) {
    if (ExperimentViewUtil.isSortedBy(sortState, isMetric, isParam, key)) {
      return (
        <span>
          <i
            className={sortState.ascending ? "fas fa-caret-up" : "fas fa-caret-down"}
            style={ExperimentViewUtil.styles.sortIconStyle}
          />
        </span>);
    }
    return undefined;
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
      const sortIcon = ExperimentViewUtil.getSortIcon(sortState, false, false, key);
      return (
        <th
          key={"meta-" + key}
          className="bottom-row sortable run-table-container"
          onClick={() => onSortBy(false, false, key)}
        >
          <span style={ExperimentViewUtil.styles.headerCellText}>{text}</span>
          <span style={ExperimentViewUtil.styles.sortIconContainer}>{sortIcon}</span>
        </th>);
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
}
