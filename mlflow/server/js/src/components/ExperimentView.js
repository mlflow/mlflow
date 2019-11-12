import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import './ExperimentView.css';
import {
  getExperiment,
  getParams,
  getRunInfo,
  getRunTags,
  getExperimentTags } from '../reducers/Reducers';
import { setExperimentTagApi, getUUID } from '../Actions';
import { withRouter } from 'react-router-dom';
import Routes from '../Routes';
import { Button, DropdownButton, MenuItem } from 'react-bootstrap';
import { Experiment, RunInfo } from '../sdk/MlflowMessages';
import { saveAs } from 'file-saver';
import { getLatestMetrics } from '../reducers/MetricReducer';
import KeyFilter from '../utils/KeyFilter';
import ExperimentRunsTableCompactView from "./ExperimentRunsTableCompactView";
import { LIFECYCLE_FILTER } from './ExperimentPage';
import ExperimentViewUtil from './ExperimentViewUtil';
import DeleteRunModal from './modals/DeleteRunModal';
import RestoreRunModal from './modals/RestoreRunModal';
import { NoteInfo, NOTE_CONTENT_TAG } from "../utils/NoteUtils";
import LocalStorageUtils from "../utils/LocalStorageUtils";
import { ExperimentViewPersistedState } from "../sdk/MlflowLocalStorageMessages";
import { Icon, Popover, Descriptions } from 'antd';
import { CollapsibleSection } from '../common/components/CollapsibleSection';
import { EditableNote } from '../common/components/EditableNote';


import Utils from '../utils/Utils';
import { Spinner } from "./Spinner";

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
    this.initiateSearch = this.initiateSearch.bind(this);
    this.onDeleteRun = this.onDeleteRun.bind(this);
    this.onRestoreRun = this.onRestoreRun.bind(this);
    this.onLifecycleFilterInput = this.onLifecycleFilterInput.bind(this);
    this.onCloseDeleteRunModal = this.onCloseDeleteRunModal.bind(this);
    this.onCloseRestoreRunModal = this.onCloseRestoreRunModal.bind(this);
    this.onExpand = this.onExpand.bind(this);
    this.addBagged = this.addBagged.bind(this);
    this.removeBagged = this.removeBagged.bind(this);
    this.renderNoteSection = this.renderNoteSection.bind(this);
    this.handleSubmitEditNote = this.handleSubmitEditNote.bind(this);
    this.handleCancelEditNote = this.handleCancelEditNote.bind(this);
    const store = ExperimentView.getLocalStore(this.props.experiment.experiment_id);
    const persistedState = new ExperimentViewPersistedState(store.loadComponentState());
    this.state = {
      ...ExperimentView.getDefaultUnpersistedState(),
      persistedState: persistedState.toJSON(),
      showNotesEditor: false,
      showNotes: true,
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
    // Object of experiment tags
    experimentTags: PropTypes.instanceOf(Object).isRequired,

    // Input to the paramKeyFilter field
    paramKeyFilter: PropTypes.instanceOf(KeyFilter).isRequired,
    // Input to the paramKeyFilter field
    metricKeyFilter: PropTypes.instanceOf(KeyFilter).isRequired,

    // Input to the lifecycleFilter field
    lifecycleFilter: PropTypes.string.isRequired,

    orderByKey: PropTypes.string,
    orderByAsc: PropTypes.bool.isRequired,

    // The initial searchInput
    searchInput: PropTypes.string.isRequired,
    searchRunsError: PropTypes.string,
    isLoading: PropTypes.bool.isRequired,

    nextPageToken: PropTypes.string,
    handleLoadMoreRuns: PropTypes.func.isRequired,
    loadingMore: PropTypes.bool.isRequired,
    setExperimentTagApi: PropTypes.func.isRequired,
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

  handleSubmitEditNote(note) {
    const { experiment_id } = this.props.experiment;
    this.props
      .setExperimentTagApi(experiment_id, NOTE_CONTENT_TAG, note, getUUID())
      .then(() => this.setState({ showNotesEditor: false }));
  }

  handleCancelEditNote() {
    this.setState({showNotesEditor: false});
  }

  startEditingDescription = (e) => {
    e.stopPropagation();
    this.setState({ showNotesEditor: true });
  };

  renderNoteSection(noteInfo) {
    const { showNotesEditor } = this.state;

    const editIcon = <a onClick={this.startEditingDescription}><Icon type='form' /></a>;

    return (
      <CollapsibleSection
        title={<span>Notes {showNotesEditor ? null : editIcon}</span>}
        forceOpen={showNotesEditor}
      >
        <EditableNote
          defaultMarkdown={noteInfo && noteInfo.content}
          onSubmit={this.handleSubmitEditNote}
          onCancel={this.handleCancelEditNote}
          showEditor={showNotesEditor}
        />
      </CollapsibleSection>
    );
  }

  render() {
    const { experiment_id, name, artifact_location } = this.props.experiment;
    const { experimentTags } = this.props;
    const {
      runInfos,
      paramKeyFilter,
      metricKeyFilter,
      isLoading,
      loadingMore,
      nextPageToken,
      handleLoadMoreRuns,
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
    const noteInfo = NoteInfo.fromTags(experimentTags);
    const searchInputHelpTooltipContent = (
      <div className="search-input-tooltip-content">
        Search runs using a simplified version of the SQL <b>WHERE</b> clause.<br/>
        <a
          href="https://www.mlflow.org/docs/latest/search-syntax.html"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn more
        </a>
      </div>
    );
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
        <Descriptions className='metadata-list'>
          <Descriptions.Item label='Experiment ID'>{experiment_id}</Descriptions.Item>
          <Descriptions.Item label='Artifact Location'>{artifact_location}</Descriptions.Item>
        </Descriptions>
        <div className="ExperimentView-info">
          {this.renderNoteSection(noteInfo)}
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
                      placeholder={'metrics.rmse < 1 and params.model = "tree" and ' +
                                   'tags.mlflow.source.type = "LOCAL"'}
                      value={this.state.searchInput}
                      onChange={this.onSearchInput}
                    />
                  </div>
                </div>
                <Popover
                  overlayClassName="search-input-tooltip"
                  content={searchInputHelpTooltipContent}
                  placement="bottom"
                >
                  <Icon
                    type="question-circle"
                    className="ExperimentView-search-help"
                    theme="filled"
                  />
                </Popover>
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
                  <label className="filter-label metrics-filter-label">Filter Metrics:</label>
                  <div className="filter-wrapper metrics-filter-wrapper">
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
              Showing {runInfos.length} matching {runInfos.length === 1 ? 'run' : 'runs'}
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
          {isLoading ?
            <Spinner showImmediately/> : (
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
                orderByKey={this.props.orderByKey}
                orderByAsc={this.props.orderByAsc}
                runsSelected={this.state.runsSelected}
                runsExpanded={this.state.persistedState.runsExpanded}
                onExpand={this.onExpand}
                unbaggedMetrics={unbaggedMetricKeyList}
                unbaggedParams={unbaggedParamKeyList}
                onAddBagged={this.addBagged}
                onRemoveBagged={this.removeBagged}
                nextPageToken={nextPageToken}
                handleLoadMoreRuns={handleLoadMoreRuns}
                loadingMore={loadingMore}
              />
            )
          }
        </div>
      </div>
    );
  }

  onSortBy(orderByKey, orderByAsc) {
    this.initiateSearch({orderByKey, orderByAsc});
  }

  initiateSearch({
      paramKeyFilterInput,
      metricKeyFilterInput,
      searchInput,
      lifecycleFilterInput,
      orderByKey,
      orderByAsc,
    }) {
    const myParamKeyFilterInput = (paramKeyFilterInput !== undefined ?
      paramKeyFilterInput : this.state.paramKeyFilterInput);
    const myMetricKeyFilterInput = (metricKeyFilterInput !== undefined ?
      metricKeyFilterInput : this.state.metricKeyFilterInput);
    const mySearchInput = (searchInput !== undefined ? searchInput : this.state.searchInput);
    const myLifecycleFilterInput = (lifecycleFilterInput !== undefined ?
      lifecycleFilterInput : this.state.lifecycleFilterInput);
    const myOrderByKey = (orderByKey !== undefined ? orderByKey : this.props.orderByKey);
    const myOrderByAsc = (orderByAsc !== undefined ? orderByAsc : this.props.orderByAsc);

    try {
      this.props.onSearch(myParamKeyFilterInput, myMetricKeyFilterInput, mySearchInput,
        myLifecycleFilterInput, myOrderByKey, myOrderByAsc);
    } catch (ex) {
      if (ex.errorMessage !== undefined) {
        this.setState({ searchErrorMessage: ex.errorMessage });
      } else {
        throw ex;
      }
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
    this.initiateSearch({paramKeyFilterInput, metricKeyFilterInput, searchInput,
      lifecycleFilterInput});
  }

  onClear() {
    // When user clicks "Clear", reset persisted state attributes to their default values.
    const newPersistedState = new ExperimentViewPersistedState();
    this.setState({persistedState: newPersistedState.toJSON()}, () => {
      this.snapshotComponentState();
      this.initiateSearch({paramKeyFilterInput: "", metricKeyFilterInput: "",
        searchInput: "", lifecycleFilterInput: LIFECYCLE_FILTER.ACTIVE,
        orderByKey: null, orderByAsc: true});
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
    if ((/[,"\r\n]/).test(str)) {
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
        Utils.getSourceType(tagsList[index]),
        Utils.getSourceName(tagsList[index]),
        Utils.getUser(runInfo, tagsList[index]),
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
  const { lifecycleFilter } = ownProps;

  // The runUuids we should serve.
  const { runInfosByUuid } = state.entities;
  const runUuids = Object.values(runInfosByUuid)
    .filter((r) => r.experiment_id === ownProps.experimentId.toString())
    .map((r) => r.run_uuid);

  const runInfos = runUuids.map((run_id) => getRunInfo(run_id, state))
    .filter((rInfo) => {
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
    const metricsByRunUuid = getLatestMetrics(runInfo.getRunUuid(), state);
    const metrics = Object.values(metricsByRunUuid || {});
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
  const experimentTags = getExperimentTags(experiment.experiment_id, state);
  return {
    runInfos,
    experiment,
    metricKeyList: Array.from(metricKeysSet.values()).sort(),
    paramKeyList: Array.from(paramKeysSet.values()).sort(),
    metricsList,
    paramsList,
    tagsList,
    experimentTags,
  };
};

const mapDispatchToProps = {
  setExperimentTagApi,
};

const styles = {
  lifecycleButtonLabel: {
    width: '32px'
  },
  lifecycleButtonFilterWrapper: {
    marginLeft: '48px',
  },
};

export default withRouter(connect(mapStateToProps, mapDispatchToProps)(ExperimentView));
