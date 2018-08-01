import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import './ExperimentView.css';
import { getApis, getExperiment, getParams, getRunInfos } from '../reducers/Reducers';
import 'react-virtualized/styles.css';
import { Link, withRouter } from 'react-router-dom';
import Routes from '../Routes';
import { Button } from 'react-bootstrap';
import Table from 'react-bootstrap/es/Table';
import { Experiment, RunInfo } from '../sdk/MlflowMessages';
import { SearchUtils } from '../utils/SearchUtils';
import { saveAs } from 'file-saver';
import { getLatestMetrics } from '../reducers/MetricReducer';
import KeyFilter from '../utils/KeyFilter';
import Utils from '../utils/Utils';

class ExperimentView extends Component {
  constructor(props) {
    super(props);
    this.onCheckbox = this.onCheckbox.bind(this);
    this.onCompare = this.onCompare.bind(this);
    this.onDownloadCsv = this.onDownloadCsv.bind(this);
    this.onParamKeyFilterInput = this.onParamKeyFilterInput.bind(this);
    this.onMetricKeyFilterInput = this.onMetricKeyFilterInput.bind(this);
    this.onSearchInput = this.onSearchInput.bind(this);
    this.onSearch = this.onSearch.bind(this);
    this.onClear = this.onClear.bind(this);
  }

  static propTypes = {
    onSearch: PropTypes.func.isRequired,
    runInfos: PropTypes.arrayOf(RunInfo).isRequired,
    experiment: PropTypes.instanceOf(Experiment).isRequired,

    // List of all parameter keys available in the runs we're viewing
    paramKeyList: PropTypes.arrayOf(String).isRequired,
    // List of all metric keys available in the runs we're viewing
    metricKeyList: PropTypes.arrayOf(String).isRequired,

    // List of list of params in all the visible runs
    paramsList: PropTypes.arrayOf(Array).isRequired,
    // List of list of metrics in all the visible runs
    metricsList: PropTypes.arrayOf(Array).isRequired,

    // Input to the paramKeyFilter field
    paramKeyFilter: PropTypes.instanceOf(KeyFilter).isRequired,
    // Input to the paramKeyFilter field
    metricKeyFilter: PropTypes.instanceOf(KeyFilter).isRequired,

    // The initial searchInput
    searchInput: PropTypes.string.required,
  };

  state = {
    runsSelected: {},
    paramKeyFilterInput: '',
    metricKeyFilterInput: '',
    searchInput: '',
    searchErrorMessage: undefined,
  };

  static getDerivedStateFromProps(nextProps, prevState) {
    const { searchInput, paramKeyFilter, metricKeyFilter } = nextProps;
    const paramKeyFilterInput = paramKeyFilter.getFilterString();
    const metricKeyFilterInput = metricKeyFilter.getFilterString();
    return {
      ...prevState,
      searchInput,
      paramKeyFilterInput,
      metricKeyFilterInput,
    };
  }

  render() {
    const { experiment_id, name, artifact_location } = this.props.experiment;
    const {
      runInfos,
      paramsList,
      metricsList,
      paramKeyFilter,
      metricKeyFilter
    } = this.props;

    // Apply our parameter and metric key filters to just pass the filtered, sorted lists
    // of parameter and metric names around later
    const paramKeyList = paramKeyFilter.apply(this.props.paramKeyList);
    const metricKeyList = metricKeyFilter.apply(this.props.metricKeyList);

    const columns = ExperimentView.getColumnHeaders(paramKeyList, metricKeyList);
    const metricRanges = ExperimentView.computeMetricRanges(metricsList);
    const rows = [...Array(runInfos.length).keys()].map((idx) => ({
        key: runInfos[idx].run_uuid,
        contents: ExperimentView.runInfoToRow(
          runInfos[idx],
          this.onCheckbox,
          paramKeyList,
          metricKeyList,
          paramsList[idx],
          metricsList[idx],
          metricRanges)
      })
    );
    const compareDisabled = Object.keys(this.state.runsSelected).length < 2;
    return (
      <div className="ExperimentView">
        <h1>{name}</h1>
        <div className="metadata">
          <span className="metadata">
            <span className="metadata-header">Experiment ID:</span>
            {experiment_id}
          </span>
          <span className="metadata">
            <span className="metadata-header">Artifact Location:</span>
            {artifact_location}
          </span>
        </div>
        <div className="ExperimentView-runs">
          {this.state.searchErrorMessage !== undefined ?
            <div className="error-message">
              <span className="error-message">{this.state.searchErrorMessage}</span>
            </div>:
            null
          }
          <form className="ExperimentView-search-controls"  onSubmit={this.onSearch}>
            <div className="ExperimentView-search-buttons">
              <input type="submit"
                     className="search-button btn btn-primary"
                     onClick={this.onSearch}
                     preventDefault
                     value="Search"
              />
              <Button className="clear-button" onClick={this.onClear}>Clear</Button>
            </div>
            <div className="ExperimentView-search-inputs">
              <div className="ExperimentView-search">
                <label className="filter-label">Search Runs:</label>
                <div className="filter-wrapper">
                  <input type="text"
                         placeholder={'metrics.rmse < 1 and params.model = "tree"'}
                         value={this.state.searchInput}
                         onChange={this.onSearchInput}
                  />
                </div>
              </div>
              <div className="ExperimentView-keyFilters">
                <div className="ExperimentView-paramKeyFilter">
                  <label className="filter-label">Filter Params:</label>
                  <div className="filter-wrapper">
                    <input type="text"
                           placeholder="alpha, lr"
                           value={this.state.paramKeyFilterInput}
                           onChange={this.onParamKeyFilterInput}
                    />
                  </div>
                </div>
                <div className="ExperimentView-metricKeyFilter">
                  <label className="filter-label">Filter Metrics:</label>
                  <div className="filter-wrapper">
                    <input type="text"
                           placeholder="rmse, r2"
                           value={this.state.metricKeyFilterInput}
                           onChange={this.onMetricKeyFilterInput}
                    />
                  </div>
                </div>
              </div>
            </div>
          </form>
          <div className="ExperimentView-run-buttons">
            <span className="run-count">{rows.length} matching {rows.length === 1 ? 'run' : 'runs'}</span>
            <Button className="btn-primary" disabled={compareDisabled} onClick={this.onCompare}>
              Compare Selected
            </Button>
            <Button onClick={this.onDownloadCsv}>
              Download CSV <i className="fas fa-download"/>
            </Button>
          </div>
          <Table hover>
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
            { rows.map(row => <tr key={row.key}>{row.contents}</tr> )}
            </tbody>
          </Table>
        </div>
      </div>
    );
  }

  onCheckbox(runUuid) {
    const newState = Object.assign({}, this.state);
    if (this.state.runsSelected[runUuid]) {
      delete newState.runsSelected[runUuid];
      this.setState(newState);
    } else {
      this.setState({
        runsSelected: {
          ...this.state.runsSelected,
          [runUuid]: true,
        }
      })
    }
  }

  onParamKeyFilterInput(event) {
    this.setState({ paramKeyFilterInput: event.target.value });
  }

  onMetricKeyFilterInput(event) {
    this.setState({ metricKeyFilterInput: event.target.value });
  }

  onSearchInput(event) {
    this.setState({ searchInput: event.target.value });
  }

  onSearch(e) {
    e.preventDefault();
    const { paramKeyFilterInput, metricKeyFilterInput, searchInput } = this.state;
    const paramKeyFilter = new KeyFilter(paramKeyFilterInput);
    const metricKeyFilter = new KeyFilter(metricKeyFilterInput);
    try {
      const andedExpressions = SearchUtils.parseSearchInput(searchInput);
      this.props.onSearch(paramKeyFilter, metricKeyFilter, andedExpressions, searchInput);
    } catch(e) {
      this.setState({ searchErrorMessage: e.errorMessage });
    }
  }

  onClear() {
    const paramKeyFilter = new KeyFilter();
    const metricKeyFilter = new KeyFilter();
    const andedExpressions = [];
    this.props.onSearch(paramKeyFilter, metricKeyFilter, andedExpressions, "");
  }

  onCompare() {
    const runsSelectedList = Object.keys(this.state.runsSelected);
    this.props.history.push(Routes.getCompareRunPageRoute(runsSelectedList));
  }

  onDownloadCsv() {
    const csv = ExperimentView.runInfosToCsv(
      this.props.runInfos,
      this.props.paramKeyFilter.apply(this.props.paramKeyList),
      this.props.metricKeyFilter.apply(this.props.metricKeyList),
      this.props.paramsList,
      this.props.metricsList);
    const blob = new Blob([csv], { type: 'application/csv;charset=utf-8' });
    saveAs(blob, "runs.csv");
  }

  /**
   * Generate a row for a specific run, extracting the params and metrics in the given lists.
   */
  static runInfoToRow(
    runInfo,
    onCheckbox,
    paramKeyList,
    metricKeyList,
    params,
    metrics,
    metricRanges) {

    const numParams = paramKeyList.length;
    const numMetrics = metricKeyList.length;
    const row = [
      <td><input type="checkbox" onClick={() => onCheckbox(runInfo.run_uuid)}/></td>,
      <td>
        <Link to={Routes.getRunPageRoute(runInfo.experiment_id, runInfo.run_uuid)}>
          {runInfo.start_time ? Utils.formatTimestamp(runInfo.start_time) : '(unknown)'}
        </Link>
      </td>,
      <td>{Utils.formatUser(runInfo.user_id)}</td>,
      <td>{Utils.renderSource(runInfo)}</td>,
      <td>{Utils.renderVersion(runInfo)}</td>,
    ];

    const paramsMap = ExperimentView.toParamsMap(params);
    const metricsMap = ExperimentView.toMetricsMap(metrics);

    let firstParam = true;
    paramKeyList.forEach((paramKey) => {
      const className = firstParam ? "left-border": undefined;
      firstParam = false;
      if (paramsMap[paramKey]) {
        row.push(<td className={className}>
          {paramsMap[paramKey].getValue()}
        </td>);
      } else {
        row.push(<td className={className}/>);
      }
    });
    if (numParams === 0) {
      row.push(<td className="left-border"/>);
    }

    let firstMetric = true;
    metricKeyList.forEach((metricKey) => {
      const className = firstMetric ? "left-border": undefined;
      firstMetric = false;
      if (metricsMap[metricKey]) {
        const metric = metricsMap[metricKey].getValue();
        const range = metricRanges[metricKey];
        let fraction = 1.0;
        if (range.max > range.min) {
          fraction = (metric - range.min) / (range.max - range.min);
        }
        const percent = (fraction * 100) + "%";
        row.push(
          <td className={className}>
            <div className="metric-filler-bg">
              <div className="metric-filler-fg" style={{width: percent}}/>
              <div className="metric-text">
                {Utils.formatMetric(metric)}
              </div>
            </div>
          </td>
        );
      } else {
        row.push(<td className={className}/>);
      }
    });
    if (numMetrics === 0) {
      row.push(<td className="left-border"/>);
    }
    return row;
  }

  static getColumnHeaders(paramKeyList, metricKeyList) {
    const numParams = paramKeyList.length;
    const numMetrics = metricKeyList.length;
    const columns = [
      <th className="bottom-row"/>,  // TODO: checkbox for select-all
      <th className="bottom-row">Date</th>,
        <th className="bottom-row">User</th>,
      <th className="bottom-row">Source</th>,
      <th className="bottom-row">Version</th>,
    ];
    let firstParam = true;
    paramKeyList.forEach((paramKey) => {
      const className = "bottom-row" + (firstParam ? " left-border" : "");
      firstParam = false;
      columns.push(<th className={className}>{paramKey}</th>);
    });
    if (numParams === 0) {
      columns.push(<th className="bottom-row left-border">(n/a)</th>);
    }

    let firstMetric = true;
    metricKeyList.forEach((metricKey) => {
      const className = "bottom-row" + (firstMetric ? " left-border" : "");
      firstMetric = false;
      columns.push(<th className={className}>{metricKey}</th>);
    });
    if (numMetrics === 0) {
      columns.push(<th className="bottom-row left-border">(n/a)</th>);
    }

    return columns;
  }

  static computeMetricRanges(metricsByRun) {
    let ret = {};
    metricsByRun.forEach(metrics => {
      metrics.forEach(metric => {
        if (!ret.hasOwnProperty(metric.key)) {
          ret[metric.key] = {min: Math.min(metric.value, metric.value * 0.7), max: metric.value}
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
    let ret = {};
    metrics.forEach((metric) => {
      ret[metric.key] = metric;
    });
    return ret;
  }

  /**
   * Turn a list of metrics to a map of metric key to metric.
   */
  static toParamsMap(params) {
    let ret = {};
    params.forEach((param) => {
      ret[param.key] = param;
    });
    return ret;
  }

  /**
   * Format a string for insertion into a CSV file.
   */
  static csvEscape(str) {
    if (str === undefined) {
      return "";
    }
    if (/[,"\r\n]/.test(str)) {
      return '"' + str.replace(/"/g, '""') + '"';
    }
    return str;
  }

  /**
   * Convert a table to a CSV string.
   *
   * @param columns Names of columns
   * @param data Array of rows, each of which are an array of field values
   */
  static tableToCsv(columns, data) {
    let csv = '';
    let i;

    for (i = 0; i < columns.length; i++) {
      csv += ExperimentView.csvEscape(columns[i]);
      if (i < columns.length - 1) {
        csv += ',';
      }
    }
    csv += '\n';

    for (i = 0; i < data.length; i++) {
      for (let j = 0; j < data[i].length; j++) {
        csv += ExperimentView.csvEscape(data[i][j]);
        if (j < data[i].length - 1) {
          csv += ',';
        }
      }
      csv += '\n';
    }

    return csv;
  };

  /**
   * Convert an array of run infos to a CSV string, extracting the params and metrics in the
   * provided lists.
   */
  static runInfosToCsv(
    runInfos,
    paramKeyList,
    metricKeyList,
    paramsList,
    metricsList) {

    const columns = [
      "Run ID",
      "Name",
      "Source Type",
      "Source Name",
      "User",
      "Status",
    ];
    paramKeyList.forEach(paramKey => {
      columns.push(paramKey);
    });
    metricKeyList.forEach(metricKey => {
      columns.push(metricKey);
    });

    const data = runInfos.map((runInfo, index) => {
      const row = [
        runInfo.run_uuid,
        runInfo.name,
        runInfo.source_type,
        runInfo.source_name,
        runInfo.user_id,
        runInfo.status,
      ];
      const paramsMap = ExperimentView.toParamsMap(paramsList[index]);
      const metricsMap = ExperimentView.toMetricsMap(metricsList[index]);
      paramKeyList.forEach((paramKey) => {
        if (paramsMap[paramKey]) {
          row.push(paramsMap[paramKey].getValue());
        } else {
          row.push("");
        }
      });
      metricKeyList.forEach((metricKey) => {
        if (metricsMap[metricKey]) {
          row.push(metricsMap[metricKey].getValue());
        } else {
          row.push("");
        }
      });
      return row;
    });

    return ExperimentView.tableToCsv(columns, data)
  }
}

const mapStateToProps = (state, ownProps) => {
  const { searchRunsRequestId } = ownProps;
  const searchRunApi = getApis([searchRunsRequestId], state)[0];
  // The runUuids we should serve.
  let runUuids;
  if (searchRunApi.data.runs) {
    runUuids = new Set(searchRunApi.data.runs.map((r) => r.info.run_uuid));
  } else {
    runUuids = new Set();
  }
  const runInfos = getRunInfos(state).filter((rInfo) =>
    runUuids.has(rInfo.getRunUuid())
  );
  const experiment = getExperiment(ownProps.experimentId, state);
  const metricKeysSet = new Set();
  const paramKeysSet = new Set();
  const metricsList = runInfos.map((runInfo) => {
    const metrics = Object.values(getLatestMetrics(runInfo.getRunUuid(), state));
    metrics.forEach((metric) => {
      metricKeysSet.add(metric.key);
    });
    return metrics
  });
  const paramsList = runInfos.map((runInfo) => {
    const params = Object.values(getParams(runInfo.getRunUuid(), state));
    params.forEach((param) => {
      paramKeysSet.add(param.key);
    });
    return params;
  });
  return {
    runInfos,
    experiment,
    metricKeyList: Array.from(metricKeysSet.values()).sort(),
    paramKeyList: Array.from(paramKeysSet.values()).sort(),
    metricsList,
    paramsList,
  };
};

export default withRouter(connect(mapStateToProps)(ExperimentView));
