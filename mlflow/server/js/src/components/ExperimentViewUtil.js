import React from 'react';
import Utils from "../utils/Utils";
import { Link } from 'react-router-dom';
import Routes from '../Routes';

export default class ExperimentViewUtil {
  /**
   * Returns table cells describing run metadata (i.e. not params/metrics) comprising part of
   *
   */
  static getRunInfoColumnsForRow(runInfo, tags, selected, onCheckbox) {
    return [
      <td key="meta-check"><input type="checkbox" checked={selected}
                                  onClick={() => onCheckbox(runInfo.run_uuid)}/></td>,
      <td key="meta-link">
        <Link to={Routes.getRunPageRoute(runInfo.experiment_id, runInfo.run_uuid)}>
          {runInfo.start_time ? Utils.formatTimestamp(runInfo.start_time) : '(unknown)'}
        </Link>
      </td>,
      <td key="meta-user">{Utils.formatUser(runInfo.user_id)}</td>,
      <td key="meta-source" style={{
        "white-space": "nowrap",
        "max-width": "250px",
        "overflow": "hidden",
        "text-overflow": "ellipsis",
      }}>
        {Utils.renderSourceTypeIcon(runInfo.source_type)}
        {Utils.renderSource(runInfo, tags)}
      </td>,
      <td key="meta-version">{Utils.renderVersion(runInfo)}</td>,
    ];
  }

  /**
   * Returns shared headers for
   * @param onSortBy
   * @param onCheckall
   * @param isAllCheckedBool
   * @param sortState
   * @returns {*[]}
   */
  static sharedColumnHeaders(onSortBy, onCheckall, isAllCheckedBool, sortState) {
    const getHeaderCell = (key, text) => {
      const sortedClassName = ExperimentViewUtil.sortedClassName(sortState, false, false, key);
      return <th key={"meta-" + key}
                 className={"bottom-row " + sortedClassName}
                 onClick={() => onSortBy(false, false, key)}>{text}</th>;
    };
    return [
      <th key="meta-check" className="bottom-row">
        <input type="checkbox" onChange={onCheckAll} checked={isAllCheckedBool} />
      </th>,
      getHeaderCell("start_time", <span>{"Date"}</span>),
      getHeaderCell("user_id", <span>{"User"}</span>),
      getHeaderCell("source", <span>{"Source"}</span>),
      getHeaderCell("source_version", <span>{"Version"}</span>),
    ];
  }

  static sortedClassName = (sortState, isMetric, isParam, key) => {
    if (sortState.isMetric !== isMetric
      || sortState.isParam !== isParam
      || sortState.key !== key) {
      return "sortable";
    }
    return "sortable sorted " + (sortState.ascending ? "asc" : "desc");
  };


  // static getSortValue(sort, metricsMap, paramsMap, tagsMap, runInfo) {
  //   if (sort.isMetric || sort.isParam) {
  //     const sortValue = (sort.isMetric ? metricsMap : paramsMap)[sort.key];
  //     return sortValue === undefined ? undefined : sortValue.value;
  //   } else if (sort.key === 'user_id') {
  //     return Utils.formatUser(runInfo.user_id);
  //   } else if (sort.key === 'source') {
  //     return Utils.formatSource(runInfo, tagsMap);
  //   }
  //   return runInfo[sort.key];
  // }
  //
  // static sortRows() {
  //
  // }
}
