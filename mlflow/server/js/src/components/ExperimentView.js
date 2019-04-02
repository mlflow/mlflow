import React, { Component } from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';
import { connect } from 'react-redux';
import './ExperimentView.css';
import { getApis, getExperiment, getParams, getRunInfos, getRunTags } from '../reducers/Reducers';
import { withRouter } from 'react-router-dom';
import Routes from '../Routes';
import { Button, ButtonGroup, DropdownButton, MenuItem } from 'react-bootstrap';
import { Experiment, RunInfo } from '../sdk/MlflowMessages';
import { saveAs } from 'file-saver';
import { getLatestMetrics } from '../reducers/MetricReducer';
import KeyFilter from '../utils/KeyFilter';

import ExperimentRunsTableMultiColumnView from "./ExperimentRunsTableMultiColumnView";
import ExperimentRunsTableCompactView from "./ExperimentRunsTableCompactView";
import { LIFECYCLE_FILTER } from './ExperimentPage';
import ExperimentViewUtil from './ExperimentViewUtil';
import DeleteRunModal from './modals/DeleteRunModal';
import RestoreRunModal from './modals/RestoreRunModal';

import LocalStorageUtils from "../utils/LocalStorageUtils";
import { ExperimentViewPersistedState } from "../sdk/MlflowLocalStorageMessages";

import Utils from '../utils/Utils';
import {Spinner} from "./Spinner";

export const DEFAULT_EXPANDED_VALUE = false;


export class ExperimentView extends Component {
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
    this.onSortBy = this.onSortBy.bind(this);
    this.isAllChecked = this.isAllChecked.bind(this);
    this.onCheckbox = this.onCheckbox.bind(this);
    this.onCheckAll = this.onCheckAll.bind(this);
    this.setSortBy = this.setSortBy.bind(this);
    this.onDeleteRun = this.onDeleteRun.bind(this);
    this.onRestoreRun = this.onRestoreRun.bind(this);
    this.onLifecycleFilterInput = this.onLifecycleFilterInput.bind(this);
    this.onCloseDeleteRunModal = this.onCloseDeleteRunModal.bind(this);
    this.onCloseRestoreRunModal = this.onCloseRestoreRunModal.bind(this);
    this.onExpand = this.onExpand.bind(this);
    this.addBagged = this.addBagged.bind(this);
    this.removeBagged = this.removeBagged.bind(this);
    const store = ExperimentView.getLocalStore(this.props.experiment.experiment_id);
    const persistedState = new ExperimentViewPersistedState(store.loadComponentState());
    this.state = {
      ...ExperimentView.getDefaultUnpersistedState(),
      persistedState: persistedState.toJSON(),
    };
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
    searchRunsError: PropTypes.string,
    isLoading: PropTypes.bool.isRequired,
  };

  /** Returns default values for state attributes that aren't persisted in local storage. */
  static getDefaultUnpersistedState() {
    return {
      // Object mapping from run UUID -> boolean (whether the run is selected)
      runsSelected: {},
      // Text entered into the param filter field
      paramKeyFilterInput: '',
      // Text entered into the metric filter field
      metricKeyFilterInput: '',
      // Lifecycle stage of runs to display
      lifecycleFilterInput: '',
      // Text entered into the runs-search field
      searchInput: '',
      // String error message, if any, from an attempted search
      searchErrorMessage: undefined,
      // True if a model for deleting one or more runs should be displayed
      showDeleteRunModal: false,
      // True if a model for restoring one or more runs should be displayed
      showRestoreRunModal: false,
    };
  }

  /**
   * Returns a LocalStorageStore instance that can be used to persist data associated with the
   * ExperimentView component (e.g. component state such as table sort settings), for the
   * specified experiment.
   */
  static getLocalStore(experimentId) {
    return LocalStorageUtils.getStoreForComponent("ExperimentView", experimentId);
  }


  shouldComponentUpdate(nextProps, nextState) {
    // Don't update the component if a modal is showing before and after the update try.
    if (this.state.showDeleteRunModal && nextState.showDeleteRunModal) return false;
    if (this.state.showRestoreRunModal && nextState.showRestoreRunModal) return false;
    return true;
  }

  /**
   * Returns true if search filter text was updated, e.g. if a user entered new text into the
   * param filter, metric filter, or search text boxes.
   */
  filtersDidUpdate(prevState) {
    return prevState.paramKeyFilterInput !== this.state.paramKeyFilterInput ||
      prevState.metricKeyFilterInput !== this.state.metricKeyFilterInput ||
      prevState.searchInput !== this.state.searchInput;
  }

  /** Snapshots desired attributes of the component's current state in local storage. */
  snapshotComponentState() {
    const store = ExperimentView.getLocalStore(this.props.experiment.experiment_id);
    store.saveComponentState(new ExperimentViewPersistedState(this.state.persistedState));
  }

  componentDidUpdate(prevProps, prevState) {
    // Don't snapshot state on changes to search filter text; we only want to save these on search
    // in ExperimentPage
    if (!this.filtersDidUpdate(prevState)) {
      this.snapshotComponentState();
    }
  }

  componentWillUnmount() {
    // Snapshot component state on unmounts to ensure we've captured component state in cases where
    // componentDidUpdate doesn't fire.
    this.snapshotComponentState();
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

  setShowMultiColumns(value) {
    this.setState({
      persistedState: new ExperimentViewPersistedState({
        ...this.state.persistedState,
        showMultiColumns: value,
      }).toJSON(),
    });
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

  /**
   * Mark a column as bagged by removing it from the appropriate array of unbagged columns.
   * @param isParam If true, the column is assumed to be a metric column; if false, the column is
   *                assumed to be a param column.
   * @param colName Name of the column (metric or param key).
   */
  addBagged(isParam, colName) {
    const unbagged = isParam ? this.state.persistedState.unbaggedParams :
      this.state.persistedState.unbaggedMetrics;
    const idx = unbagged.indexOf(colName);
    const newUnbagged = idx >= 0 ?
      unbagged.slice(0, idx).concat(unbagged.slice(idx + 1, unbagged.length)) : unbagged;
    const stateKey = isParam ? "unbaggedParams" : "unbaggedMetrics";
    this.setState(
      {
        persistedState: new ExperimentViewPersistedState({
          ...this.state.persistedState,
          [stateKey]: newUnbagged,
        }).toJSON(),
      });
  }

  /**
   * Mark a column as unbagged by adding it to the appropriate array of unbagged columns.
   * @param isParam If true, the column is assumed to be a metric column; if false, the column is
   *                assumed to be a param column.
   * @param colName Name of the column (metric or param key).
   */
  removeBagged(isParam, colName) {
    const unbagged = isParam ? this.state.persistedState.unbaggedParams :
      this.state.persistedState.unbaggedMetrics;
    const stateKey = isParam ? "unbaggedParams" : "unbaggedMetrics";
    this.setState(
      {
        persistedState: new ExperimentViewPersistedState({
          ...this.state.persistedState,
          [stateKey]: unbagged.concat([colName])
        }).toJSON()
      });
  }

  render() {
    const { experiment_id, name, artifact_location } = this.props.experiment;
    const {
      runInfos,
      paramKeyFilter,
      metricKeyFilter,
    } = this.props;

    // Apply our parameter and metric key filters to just pass the filtered, sorted lists
    // of parameter and metric names around later
    const paramKeyList = paramKeyFilter.apply(this.props.paramKeyList);
    const metricKeyList = metricKeyFilter.apply(this.props.metricKeyList);
    const unbaggedParamKeyList = paramKeyFilter.apply(this.state.persistedState.unbaggedParams);
    const unbaggedMetricKeyList = metricKeyFilter.apply(this.state.persistedState.unbaggedMetrics);

    const compareDisabled = Object.keys(this.state.runsSelected).length < 2;
    const deleteDisabled = Object.keys(this.state.runsSelected).length < 1;
    const restoreDisabled = Object.keys(this.state.runsSelected).length < 1;
    return (
      <div className="ExperimentView runs-table-flex-container">
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
        <div className="ExperimentView-runs runs-table-flex-container">
          {this.props.searchRunsError ?
            <div className="error-message">
              <span className="error-message">{this.props.searchRunsError}</span>
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
                    <input
                      className="ExperimentView-searchInput"
                      type="text"
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
              {runInfos.length} matching {runInfos.length === 1 ? 'run' : 'runs'}
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
            <span style={{cursor: "pointer"}}>
                <ButtonGroup style={styles.tableToggleButtonGroup}>
                <Button
                  onClick={() => this.setShowMultiColumns(false)}
                  title="Compact view"
                  className={classNames({ "active": !this.state.persistedState.showMultiColumns })}
                >
                  <i className={"fas fa-list"}/>
                </Button>
                <Button
                  onClick={() => this.setShowMultiColumns(true)}
                  title="Grid view"
                  className={classNames({ "active": this.state.persistedState.showMultiColumns })}
                >
                  <i className={"fas fa-table"}/>
                </Button>
                </ButtonGroup>
            </span>
          </div>
          {this.props.isLoading ?
            <Spinner showImmediately/> :
            (this.state.persistedState.showMultiColumns ?
                <ExperimentRunsTableMultiColumnView
                  onCheckbox={this.onCheckbox}
                  runInfos={this.props.runInfos}
                  paramsList={this.props.paramsList}
                  metricsList={this.props.metricsList}
                  tagsList={this.props.tagsList}
                  paramKeyList={paramKeyList}
                  metricKeyList={metricKeyList}
                  onCheckAll={this.onCheckAll}
                  isAllChecked={this.isAllChecked()}
                  onSortBy={this.onSortBy}
                  sortState={this.state.persistedState.sort}
                  runsSelected={this.state.runsSelected}
                  runsExpanded={this.state.persistedState.runsExpanded}
                  onExpand={this.onExpand}
                /> :
                <ExperimentRunsTableCompactView
                  onCheckbox={this.onCheckbox}
                  runInfos={this.props.runInfos}
                  // Bagged param and metric keys
                  paramKeyList={paramKeyList}
                  metricKeyList={metricKeyList}
                  paramsList={this.props.paramsList}
                  metricsList={this.props.metricsList}
                  tagsList={this.props.tagsList}
                  onCheckAll={this.onCheckAll}
                  isAllChecked={this.isAllChecked()}
                  onSortBy={this.onSortBy}
                  sortState={this.state.persistedState.sort}
                  runsSelected={this.state.runsSelected}
                  setSortByHandler={this.setSortBy}
                  runsExpanded={this.state.persistedState.runsExpanded}
                  onExpand={this.onExpand}
                  unbaggedMetrics={unbaggedMetricKeyList}
                  unbaggedParams={unbaggedParamKeyList}
                  onAddBagged={this.addBagged}
                  onRemoveBagged={this.removeBagged}
                />
            )
          }
        </div>
      </div>
    );
  }

  onSortBy(isMetric, isParam, key) {
    const sort = this.state.persistedState.sort;
    this.setSortBy(isMetric, isParam, key, !sort.ascending);
  }

  setSortBy(isMetric, isParam, key, ascending) {
    const newSortState = {
      ascending: ascending,
      key: key,
      isMetric: isMetric,
      isParam: isParam
    };
    this.setState(
      {
        persistedState: new ExperimentViewPersistedState({
          ...this.state.persistedState,
          sort: newSortState,
        }).toJSON(),
      });
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

  onExpand(runId, childrenIds) {
    const newExpanderState = !ExperimentViewUtil.isExpanderOpen(
      this.state.persistedState.runsExpanded, runId);
    const newRunsHiddenByExpander = {...this.state.persistedState.runsHiddenByExpander};
    childrenIds.forEach((childId) => {
      newRunsHiddenByExpander[childId] = !newExpanderState;
    });
    const newPersistedStateFields = {
      runsExpanded: {
        ...this.state.persistedState.runsExpanded,
        [runId]: newExpanderState,
      },
      runsHiddenByExpander: newRunsHiddenByExpander,
    };
    this.setState({
      persistedState: new ExperimentViewPersistedState({
        ...this.state.persistedState,
        ...newPersistedStateFields,
      }).toJSON(),
    });
    // Deselect the children
    const newRunsSelected = {...this.state.runsSelected};
    if (newExpanderState === false) {
      childrenIds.forEach((childId) => {
        if (newRunsSelected[childId]) {
          delete newRunsSelected[childId];
        }
      });
      this.setState({ runsSelected: newRunsSelected });
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
    try {
      this.props.onSearch(paramKeyFilterInput, metricKeyFilterInput, searchInput,
        lifecycleFilterInput);
    } catch (ex) {
      this.setState({ searchErrorMessage: ex.errorMessage });
    }
  }

  onClear() {
    // When user clicks "Clear", preserve multicolumn toggle state but reset other persisted state
    // attributes to their default values.
    const newPersistedState = new ExperimentViewPersistedState({
      showMultiColumns: this.state.persistedState.showMultiColumns,
    });
    this.setState({persistedState: newPersistedState.toJSON()}, () => {
      this.snapshotComponentState();
      this.props.onSearch("", "", "", LIFECYCLE_FILTER.ACTIVE);
    });
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
      this.props.metricsList,
      this.props.tagsList);
    const blob = new Blob([csv], { type: 'application/csv;charset=utf-8' });
    saveAs(blob, "runs.csv");
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
    metricsList,
    tagsList) {
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
        Utils.getRunName(tagsList[index]), // add run name to csv export row
        runInfo.source_type,
        runInfo.source_name,
        runInfo.user_id,
        runInfo.status,
      ];

      const paramsMap = ExperimentViewUtil.toParamsMap(paramsList[index]);
      const metricsMap = ExperimentViewUtil.toMetricsMap(metricsList[index]);

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

export const mapStateToProps = (state, ownProps) => {
  const { lifecycleFilter, searchRunsRequestId } = ownProps;
  const searchRunApi = getApis([searchRunsRequestId], state)[0];
  // The runUuids we should serve.
  let runUuids;
  if (searchRunApi.data && searchRunApi.data.runs) {
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
    width: '32px'
  },
  lifecycleButtonFilterWrapper: {
    marginLeft: '48px',
  },
  tableToggleButtonGroup: {
    marginLeft: '16px',
  },
};

export default withRouter(connect(mapStateToProps)(ExperimentView));
