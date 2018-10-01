import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import './ExperimentView.css';
import { getApis, getExperiment, getParams, getRunInfos, getRunTags } from '../reducers/Reducers';
import 'react-virtualized/styles.css';
import { Link, withRouter } from 'react-router-dom';
import Routes from '../Routes';
import { Button, DropdownButton, MenuItem } from 'react-bootstrap';
import Table from 'react-bootstrap/es/Table';
import { Experiment, RunInfo } from '../sdk/MlflowMessages';
import { SearchUtils } from '../utils/SearchUtils';
import { saveAs } from 'file-saver';
import { getLatestMetrics } from '../reducers/MetricReducer';
import KeyFilter from '../utils/KeyFilter';
import Utils from '../utils/Utils';
import { LIFECYCLE_FILTER } from './ExperimentPage';
import DeleteRunModal from './modals/DeleteRunModal';
import RestoreRunModal from './modals/RestoreRunModal';

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
    this.onDeleteRun = this.onDeleteRun.bind(this);
    this.onRestoreRun = this.onRestoreRun.bind(this);
    this.onLifecycleFilterInput = this.onLifecycleFilterInput.bind(this);
    this.onCloseDeleteRunModal = this.onCloseDeleteRunModal.bind(this);
    this.onCloseRestoreRunModal = this.onCloseRestoreRunModal.bind(this);
  }

  static propTypes = {
    onSearch: PropTypes.func.isRequired,
    runInfos: PropTypes.arrayOf(RunInfo).isRequired,
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    history: PropTypes.any,

    // List of all parameter keys available in the runs we're viewing
    paramKeyList: PropTypes.arrayOf(String).isRequired,
    // List of all metric keys available in the runs we're viewing
    metricKeyList: PropTypes.arrayOf(String).isRequired,

    // List of list of params in all the visible runs
    paramsList: PropTypes.arrayOf(Array).isRequired,
    // List of list of metrics in all the visible runs
    metricsList: PropTypes.arrayOf(Array).isRequired,
    // List of tags dictionary in all the visible runs.
    tagsList: PropTypes.arrayOf(Object).isRequired,

    // Input to the paramKeyFilter field
    paramKeyFilter: PropTypes.instanceOf(KeyFilter).isRequired,
    // Input to the paramKeyFilter field
    metricKeyFilter: PropTypes.instanceOf(KeyFilter).isRequired,

    // Input to the lifecycleFilter field
    lifecycleFilter: PropTypes.string.isRequired,

    // The initial searchInput
    searchInput: PropTypes.string.isRequired,
  };

  state = {
    runsSelected: {},
    paramKeyFilterInput: '',
    metricKeyFilterInput: '',
    lifecycleFilterInput: LIFECYCLE_FILTER.ACTIVE,
    searchInput: '',
    searchErrorMessage: undefined,
    sort: {
      ascending: false,
      isMetric: false,
      isParam: false,
      key: "start_time"
    },
    showDeleteRunModal: false,
    showRestoreRunModal: false,
  };

  shouldComponentUpdate(nextProps, nextState) {
    // Don't update the component if a modal is showing before and after the update try.
    if (this.state.showDeleteRunModal && nextState.showDeleteRunModal) return false;
    if (this.state.showRestoreRunModal && nextState.showRestoreRunModal) return false;
    return true;
  }


  static getDerivedStateFromProps(nextProps, prevState) {
    // Compute the actual runs selected. (A run cannot be selected if it is not passed in as a
    // prop)
    const newRunsSelected = {};
    nextProps.runInfos.forEach((rInfo) => {
      const prevRunSelected = prevState.runsSelected[rInfo.run_uuid];
      if (prevRunSelected) {
        newRunsSelected[rInfo.run_uuid] = prevRunSelected;
      }
    });
    const { searchInput, paramKeyFilter, metricKeyFilter, lifecycleFilter } = nextProps;
    const paramKeyFilterInput = paramKeyFilter.getFilterString();
    const metricKeyFilterInput = metricKeyFilter.getFilterString();
    return {
      ...prevState,
      searchInput,
      paramKeyFilterInput,
      metricKeyFilterInput,
      lifecycleFilterInput: lifecycleFilter,
      runsSelected: newRunsSelected,
    };
  }

  onDeleteRun() {
    this.setState({ showDeleteRunModal: true });
  }

  onRestoreRun() {
    this.setState({ showRestoreRunModal: true });
  }

  onCloseDeleteRunModal() {
    this.setState({ showDeleteRunModal: false });
  }

  onCloseRestoreRunModal() {
    this.setState({ showRestoreRunModal: false });
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

    const sort = this.state.sort;
    const columns = ExperimentView.getColumnHeaders(
      paramKeyList,
      metricKeyList,
      this.onCheckAll.bind(this),
      this.isAllChecked(),
      this.onSortBy.bind(this),
      sort);

    const metricRanges = ExperimentView.computeMetricRanges(metricsList);
    const rows = [...Array(runInfos.length).keys()].map((idx) => {
      const runInfo = runInfos[idx];
      const paramsMap = ExperimentView.toParamsMap(paramsList[idx]);
      const metricsMap = ExperimentView.toMetricsMap(metricsList[idx]);
      let sortValue;
      if (sort.isMetric || sort.isParam) {
        sortValue = (sort.isMetric ? metricsMap : paramsMap)[sort.key];
        sortValue = sortValue === undefined ? undefined : sortValue.value;
      } else if (sort.key === 'user_id') {
        sortValue = Utils.formatUser(runInfo.user_id);
      } else if (sort.key === 'source') {
        sortValue = Utils.formatSource(runInfo, this.props.tagsList[idx]);
      } else {
        sortValue = runInfo[sort.key];
      }

      return {
        key: runInfo.run_uuid,
        sortValue: sortValue,
        contents: ExperimentView.runInfoToRow({
          runInfo,
          onCheckbox: this.onCheckbox,
          paramKeyList,
          metricKeyList,
          paramsMap,
          metricsMap,
          tags: this.props.tagsList[idx],
          metricRanges,
          selected: !!this.state.runsSelected[runInfo.run_uuid]})
      };
    });
    rows.sort((a, b) => {
      if (a.sortValue === undefined) {
        return 1;
      } else if (b.sortValue === undefined) {
        return -1;
      } else if (!this.state.sort.ascending) {
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

    const compareDisabled = Object.keys(this.state.runsSelected).length < 2;
    const deleteDisabled = Object.keys(this.state.runsSelected).length < 1;
    const restoreDisabled = Object.keys(this.state.runsSelected).length < 1;
    return (
      <div className="ExperimentView">
        <DeleteRunModal
          isOpen={this.state.showDeleteRunModal}
          onClose={this.onCloseDeleteRunModal}
          selectedRunIds={Object.keys(this.state.runsSelected)}
        />
        <RestoreRunModal
          isOpen={this.state.showRestoreRunModal}
          onClose={this.onCloseRestoreRunModal}
          selectedRunIds={Object.keys(this.state.runsSelected)}
        />
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
            </div> :
            null
          }
          <form className="ExperimentView-search-controls" onSubmit={this.onSearch}>
            <div className="ExperimentView-search-buttons">
              <input type="submit"
                     className="search-button btn btn-primary"
                     onClick={this.onSearch}
                     value="Search"
              />
              <Button className="clear-button" onClick={this.onClear}>Clear</Button>
            </div>
            <div className="ExperimentView-search-inputs">
              <div className="ExperimentView-search">
                <div className="ExperimentView-search-input">
                  <label className="filter-label">Search Runs:</label>
                  <div className="filter-wrapper">
                    <input type="text"
                           placeholder={'metrics.rmse < 1 and params.model = "tree"'}
                           value={this.state.searchInput}
                           onChange={this.onSearchInput}
                    />
                  </div>
                </div>
                <div className="ExperimentView-lifecycle-input">
                  <label className="filter-label" style={styles.lifecycleButtonLabel}>State:</label>
                  <div className="filter-wrapper" style={styles.lifecycleButtonFilterWrapper}>
                    <DropdownButton
                      id={"ExperimentView-lifecycle-button-id"}
                      className="ExperimentView-lifecycle-button"
                      key={this.state.lifecycleFilterInput}
                      bsStyle='default'
                      title={this.state.lifecycleFilterInput}
                    >
                      <MenuItem
                        active={this.state.lifecycleFilterInput === LIFECYCLE_FILTER.ACTIVE}
                        onSelect={this.onLifecycleFilterInput}
                        eventKey={LIFECYCLE_FILTER.ACTIVE}
                      >
                        {LIFECYCLE_FILTER.ACTIVE}
                      </MenuItem>
                      <MenuItem
                        active={this.state.lifecycleFilterInput === LIFECYCLE_FILTER.DELETED}
                        onSelect={this.onLifecycleFilterInput}
                        eventKey={LIFECYCLE_FILTER.DELETED}
                      >
                        {LIFECYCLE_FILTER.DELETED}
                      </MenuItem>
                    </DropdownButton>
                  </div>
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
            <span className="run-count">
              {rows.length} matching {rows.length === 1 ? 'run' : 'runs'}
            </span>
            <Button className="btn-primary" disabled={compareDisabled} onClick={this.onCompare}>
              Compare
            </Button>
            {
              this.props.lifecycleFilter === LIFECYCLE_FILTER.ACTIVE ?
              <Button disabled={deleteDisabled} onClick={this.onDeleteRun}>
                Delete
              </Button> : null
            }
            {
              this.props.lifecycleFilter === LIFECYCLE_FILTER.DELETED ?
              <Button disabled={restoreDisabled} onClick={this.onRestoreRun}>
                Restore
              </Button> : null
            }
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
            { rows.map(row => <tr key={row.key}>{row.contents}</tr>)}
            </tbody>
          </Table>
        </div>
      </div>
    );
  }

  onSortBy(isMetric, isParam, key) {
    const sort = this.state.sort;
    if (sort.key === key && sort.isMetric === isMetric && sort.isParam === isParam) {
      this.setState({sort: {
        ...sort,
        ascending: !sort.ascending
      }});
    } else {
      this.setState({sort: {
        ascending: true,
        key: key,
        isMetric: isMetric,
        isParam: isParam
      }});
    }
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
      });
    }
  }

  isAllChecked() {
    return Object.keys(this.state.runsSelected).length === this.props.runInfos.length;
  }

  onCheckAll() {
    if (this.isAllChecked()) {
      this.setState({runsSelected: {}});
    } else {
      const runsSelected = {};
      this.props.runInfos.forEach(({run_uuid}) => {
        runsSelected[run_uuid] = true;
      });
      this.setState({runsSelected: runsSelected});
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

  onLifecycleFilterInput(newLifecycleInput) {
    this.setState({ lifecycleFilterInput: newLifecycleInput }, this.onSearch);
  }

  onSearch(e) {
    if (e !== undefined) {
      e.preventDefault();
    }
    const {
      paramKeyFilterInput,
      metricKeyFilterInput,
      searchInput,
      lifecycleFilterInput
    } = this.state;
    const paramKeyFilter = new KeyFilter(paramKeyFilterInput);
    const metricKeyFilter = new KeyFilter(metricKeyFilterInput);
    try {
      const andedExpressions = SearchUtils.parseSearchInput(searchInput);
      this.props.onSearch(paramKeyFilter, metricKeyFilter, andedExpressions, searchInput,
        lifecycleFilterInput);
    } catch (ex) {
      this.setState({ searchErrorMessage: ex.errorMessage });
    }
  }

  onClear() {
    const paramKeyFilter = new KeyFilter();
    const metricKeyFilter = new KeyFilter();
    const andedExpressions = [];
    this.props.onSearch(paramKeyFilter, metricKeyFilter, andedExpressions, "",
      LIFECYCLE_FILTER.ACTIVE);
  }

  onCompare() {
    const runsSelectedList = Object.keys(this.state.runsSelected);
    this.props.history.push(Routes.getCompareRunPageRoute(
      runsSelectedList, this.props.experiment.getExperimentId()));
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
  static runInfoToRow({
    runInfo,
    onCheckbox,
    paramKeyList,
    metricKeyList,
    paramsMap,
    metricsMap,
    tags,
    metricRanges,
    selected}) {
    const numParams = paramKeyList.length;
    const numMetrics = metricKeyList.length;
    const row = [
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

    paramKeyList.forEach((paramKey, i) => {
      const className = i === 0 ? "left-border" : undefined;
      const keyname = "param-" + paramKey;
      if (paramsMap[paramKey]) {
        row.push(<td className={className} key={keyname}>
          {paramsMap[paramKey].getValue()}
        </td>);
      } else {
        row.push(<td className={className} key={keyname}/>);
      }
    });
    if (numParams === 0) {
      row.push(<td className="left-border" key={"meta-param-empty"}/>);
    }

    metricKeyList.forEach((metricKey, i) => {
      const className = i === 0 ? "left-border" : undefined;
      const keyname = "metric-" + metricKey;
      if (metricsMap[metricKey]) {
        const metric = metricsMap[metricKey].getValue();
        const range = metricRanges[metricKey];
        let fraction = 1.0;
        if (range.max > range.min) {
          fraction = (metric - range.min) / (range.max - range.min);
        }
        const percent = (fraction * 100) + "%";
        row.push(
          <td className={className} key={keyname}>
            <div className="metric-filler-bg">
              <div className="metric-filler-fg" style={{width: percent}}/>
              <div className="metric-text">
                {Utils.formatMetric(metric)}
              </div>
            </div>
          </td>
        );
      } else {
        row.push(<td className={className} key={keyname}/>);
      }
    });
    if (numMetrics === 0) {
      row.push(<td className="left-border" key="meta-metric-empty" />);
    }
    return row;
  }

  static getColumnHeaders(paramKeyList, metricKeyList,
    onCheckAll,
    isAllChecked,
    onSortBy,
    sortState) {
    const sortedClassName = (isMetric, isParam, key) => {
      if (sortState.isMetric !== isMetric
        || sortState.isParam !== isParam
        || sortState.key !== key) {
        return "sortable";
      }
      return "sortable sorted " + (sortState.ascending ? "asc" : "desc");
    };
    const getHeaderCell = (key, text, sortable) => {
      let onClick = () => {};
      if (sortable) {
        onClick = () => onSortBy(false, false, key);
      }
      return <th key={"meta-" + key} className={"bottom-row " + sortedClassName(false, false, key)}
        onClick={onClick}>{text}</th>;
    };

    const numParams = paramKeyList.length;
    const numMetrics = metricKeyList.length;
    const columns = [
      <th key="meta-check" className="bottom-row">
        <input type="checkbox" onChange={onCheckAll} checked={isAllChecked} />
      </th>,
      getHeaderCell("start_time", <span>{"Date"}</span>, true),
      getHeaderCell("user_id", <span>{"User"}</span>, true),
      getHeaderCell("source", <span>{"Source"}</span>, true),
      getHeaderCell("source_version", <span>{"Version"}</span>, true)
    ];
    paramKeyList.forEach((paramKey, i) => {
      const className = "bottom-row "
        + (i === 0 ? "left-border " : "")
        + sortedClassName(false, true, paramKey);
      columns.push(<th key={'param-' + paramKey} className={className}
        onClick={() => onSortBy(false, true, paramKey)}>{paramKey}</th>);
    });
    if (numParams === 0) {
      columns.push(<th key="meta-param-empty" className="bottom-row left-border">(n/a)</th>);
    }

    let firstMetric = true;
    metricKeyList.forEach((metricKey) => {
      const className = "bottom-row "
        + (firstMetric ? "left-border " : "")
        + sortedClassName(true, false, metricKey);
      firstMetric = false;
      columns.push(<th key={'metric-' + metricKey} className={className}
        onClick={() => onSortBy(true, false, metricKey)}>{metricKey}</th>);
    });
    if (numMetrics === 0) {
      columns.push(<th key="meta-metric-empty" className="bottom-row left-border">(n/a)</th>);
    }

    return columns;
  }

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
  }

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

    return ExperimentView.tableToCsv(columns, data);
  }
}

const mapStateToProps = (state, ownProps) => {
  const { lifecycleFilter, searchRunsRequestId } = ownProps;
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
  ).filter((rInfo) => {
    if (lifecycleFilter === LIFECYCLE_FILTER.ACTIVE) {
      return rInfo.lifecycle_stage === 'active';
    } else {
      return rInfo.lifecycle_stage === 'deleted';
    }
  });
  const experiment = getExperiment(ownProps.experimentId, state);
  const metricKeysSet = new Set();
  const paramKeysSet = new Set();
  const metricsList = runInfos.map((runInfo) => {
    const metrics = Object.values(getLatestMetrics(runInfo.getRunUuid(), state));
    metrics.forEach((metric) => {
      metricKeysSet.add(metric.key);
    });
    return metrics;
  });
  const paramsList = runInfos.map((runInfo) => {
    const params = Object.values(getParams(runInfo.getRunUuid(), state));
    params.forEach((param) => {
      paramKeysSet.add(param.key);
    });
    return params;
  });

  const tagsList = runInfos.map((runInfo) => getRunTags(runInfo.getRunUuid(), state));
  return {
    runInfos,
    experiment,
    metricKeyList: Array.from(metricKeysSet.values()).sort(),
    paramKeyList: Array.from(paramKeysSet.values()).sort(),
    metricsList,
    paramsList,
    tagsList,
  };
};

const styles = {
  lifecycleButtonLabel: {
    width: '60px'
  },
  lifecycleButtonFilterWrapper: {
    marginLeft: '60px',
  }
};

export default withRouter(connect(mapStateToProps)(ExperimentView));
