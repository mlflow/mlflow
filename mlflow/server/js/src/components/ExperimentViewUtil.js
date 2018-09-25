import React from 'react';
import Utils from "../utils/Utils";
import { Link } from 'react-router-dom';
import Routes from '../Routes';

export default class ExperimentViewUtil {
  static runInfoToSharedColumns(runInfo, tags, selected, onCheckbox) {
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

  static sortedClassName = (sortState, isMetric, isParam, key) => {
    if (sortState.isMetric !== isMetric
      || sortState.isParam !== isParam
      || sortState.key !== key) {
      return "sortable";
    }
    return "sortable sorted " + (sortState.ascending ? "asc" : "desc");
  };

  static getHeaderCell = (key, text, sortable, onSortBy, sortState) => {
    let onClick = () => {};
    if (sortable) {
      onClick = () => onSortBy(false, false, key);
    }
    const sortedClassName = ExperimentViewUtil.sortedClassName(sortState, false, false, key);
    return <th key={"meta-" + key}
               className={"bottom-row " + sortedClassName}
               onClick={onClick}>{text}</th>;
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
