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

  static computeMetricRanges(metricsByRun) {
    const ret = {};
    metricsByRun.forEach(metrics => {
      metrics.forEach(metric => {
        if (!ret.hasOwnProperty(metric.key)) {
          ret[metric.key] = {min: Math.min(metric.value, metric.value * 0.7), max: metric.value};
        } else {
          if (metric.value < ret[metric.key].min) {
            ret[metric.key].min = Math.min(metric.value, metric.value * 0.7);
          }
          if (metric.value > ret[metric.key].max) {
            ret[metric.key].max = metric.value;
          }
        }
      });
    });
    return ret;
  }

  /**
   * Turn a list of metrics to a map of metric key to metric.
   */
  static toMetricsMap(metrics) {
    const ret = {};
    metrics.forEach((metric) => {
      ret[metric.key] = metric;
    });
    return ret;
  }

  /**
   * Turn a list of metrics to a map of metric key to metric.
   */
  static toParamsMap(params) {
    const ret = {};
    params.forEach((param) => {
      ret[param.key] = param;
    });
    return ret;
  }

  /**
   * Mutates and sorts the rows by the sortValue member.
   */
  static sortRows(rows, sortState) {
    rows.sort((a, b) => {
      if (a.sortValue === undefined) {
        return 1;
      } else if (b.sortValue === undefined) {
        return -1;
      } else if (!sortState.ascending) {
        // eslint-disable-next-line no-param-reassign
        [a, b] = [b, a];
      }
      let x = a.sortValue;
      let y = b.sortValue;
      // Casting to number if possible
      if (!isNaN(+x)) {
        x = +x;
      }
      if (!isNaN(+y)) {
        y = +y;
      }
      return x < y ? -1 : (x > y ? 1 : 0);
    });
  }

  /**
   * Computes the sortValue for this row
   */
  static computeSortValue(sortState, metricsMap, paramsMap, runInfo, tags) {
    if (sortState.isMetric || sortState.isParam) {
      const sortValue = (sortState.isMetric ? metricsMap : paramsMap)[sortState.key];
      return (sortValue === undefined ? undefined : sortValue.value);
    } else if (sortState.key === 'user_id') {
      return Utils.formatUser(runInfo.user_id);
    } else if (sortState.key === 'source') {
      return Utils.formatSource(runInfo, tags);
    } else {
      return runInfo[sortState.key];
    }
  }
}
