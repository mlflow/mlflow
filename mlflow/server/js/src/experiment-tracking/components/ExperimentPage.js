import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import _ from 'lodash';
import { withRouter } from 'react-router-dom';

import './ExperimentPage.css';
import {
  getExperimentApi,
  searchRunsApi,
  loadMoreRunsApi,
  searchRunsPayload,
  setCompareExperiments,
} from '../actions';
import { searchModelVersionsApi } from '../../model-registry/actions';
import ExperimentView from './ExperimentView';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import { ViewType } from '../sdk/MlflowEnums';
import LocalStorageUtils from '../../common/utils/LocalStorageUtils';
import { ExperimentPagePersistedState } from '../sdk/MlflowLocalStorageMessages';
import Utils from '../../common/utils/Utils';
import { ErrorCodes } from '../../common/constants';
import { PermissionDeniedView } from './PermissionDeniedView';
import { Spinner } from '../../common/components/Spinner';
import { getUUID } from '../../common/utils/ActionUtils';
import { MAX_RUNS_IN_SEARCH_MODEL_VERSIONS_FILTER } from '../../model-registry/constants';
import { getExperiment } from '../reducers/Reducers';
import { Experiment } from '../sdk/MlflowMessages';
import { injectIntl } from 'react-intl';
import {
  ATTRIBUTE_COLUMN_SORT_KEY,
  COLUMN_TYPES,
  LIFECYCLE_FILTER,
  MAX_DETECT_NEW_RUNS_RESULTS,
  PAGINATION_DEFAULT_STATE,
  POLL_INTERVAL,
} from '../constants';

export const isNewRun = (lastRunsRefreshTime, run) => {
  if (run && run.info) {
    const { start_time, end_time } = run.info;

    return start_time >= lastRunsRefreshTime || (end_time !== 0 && end_time >= lastRunsRefreshTime);
  }

  return false;
};

export class ExperimentPage extends Component {
  static propTypes = {
    experimentIds: PropTypes.arrayOf(PropTypes.string).isRequired,
    experiments: PropTypes.arrayOf(PropTypes.instanceOf(Experiment)),
    getExperimentApi: PropTypes.func.isRequired,
    searchRunsApi: PropTypes.func.isRequired,
    searchModelVersionsApi: PropTypes.func.isRequired,
    loadMoreRunsApi: PropTypes.func.isRequired,
    history: PropTypes.object.isRequired,
    location: PropTypes.object,
    searchForNewRuns: PropTypes.func,
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
    compareExperiments: PropTypes.bool,
    setCompareExperiments: PropTypes.func,
  };

  static defaultProps = {
    compareExperiments: false,
    /*
      The runs table reads directly from the redux store, so we are intentionally not using a redux
      action to search for new runs. We do not want to change the runs displayed on the runs table
      when searching for new runs.
     */
    searchForNewRuns: searchRunsPayload,
  };

  getExperimentsParam(experimentIds) {
    return (
      'experiments=[' +
      experimentIds
        .slice()
        .sort()
        .map((experimentId) => `%22${experimentId}%22`)
        .join(',') +
      ']'
    );
  }

  /* Returns a LocalStorageStore instance that can be used to persist data associated with the
   * ExperimentView component (e.g. component state such as table sort settings), for the
   * specified experiment.
   */
  static getLocalStore(experimentIds) {
    return LocalStorageUtils.getStoreForComponent(
      'ExperimentPage',
      JSON.stringify(experimentIds.slice().sort()),
    );
  }

  constructor(props) {
    super(props);
    const store = ExperimentPage.getLocalStore(this.props.experimentIds);
    const urlState = Utils.getSearchParamsFromUrl(props.location.search);
    this.state = {
      lastRunsRefreshTime: Date.now(),
      numberOfNewRuns: 0,
      // Last experiment, if any, displayed by this instance of ExperimentPage
      lastExperimentIds: undefined,
      ...PAGINATION_DEFAULT_STATE,
      getExperimentRequestIds: this.generateGetExperimentRequestIds(),
      searchRunsRequestId: getUUID(),
      urlState: props.location.search,
      persistedState: new ExperimentPagePersistedState({
        ...store.loadComponentState(),
        ...urlState,
      }).toJSON(),
      pollingState: {
        newRuns: true,
      },
    };
  }

  updateCompareExperimentsState() {
    const { experimentIds, compareExperiments } = this.props;
    const comparedExperimentIds = compareExperiments ? experimentIds : [];
    const hasComparedExperimentsBefore = compareExperiments;
    this.props.setCompareExperiments({ comparedExperimentIds, hasComparedExperimentsBefore });
  }

  componentDidMount() {
    this.updateCompareExperimentsState();
    this.loadData();
    this.pollTimer = setInterval(() => this.pollInfo(), POLL_INTERVAL);
  }

  componentDidUpdate(prevProps, prevState) {
    if (!_.isEqual(this.props.experimentIds, prevProps.experimentIds)) {
      this.updateCompareExperimentsState();
    }
    this.maybeReloadData(prevProps, prevState);
  }

  /** Snapshots desired attributes of the component's current state in local storage. */
  snapshotComponentState() {
    const store = ExperimentPage.getLocalStore(this.props.experimentIds);
    store.saveComponentState(new ExperimentPagePersistedState(this.state.persistedState));
  }

  static getDerivedStateFromProps(props, state) {
    const experimentChanged = !_.isEqual(props.experimentIds, state.lastExperimentIds);
    const urlStateChanged =
      props.location.search !== state.urlState && props.history.action === 'POP';

    // Early return if experiment & urlState are unchanged
    if (!experimentChanged && !urlStateChanged) {
      return null;
    }

    const store = ExperimentPage.getLocalStore(props.experimentIds);
    const returnValue = {
      searchRunsRequestId: getUUID(),
      lastRunsRefreshTime: Date.now(),
      lastExperimentIds: props.experimentIds,
      ...PAGINATION_DEFAULT_STATE,
    };

    if (experimentChanged) {
      returnValue.getExperimentRequestId = getUUID();
      returnValue.persistedState =
        state.lastExperimentIds === undefined
          ? state.persistedState
          : new ExperimentPagePersistedState({
              ...store.loadComponentState(),
              ...Utils.getSearchParamsFromUrl(props.location.search),
            }).toJSON();
    }

    if (urlStateChanged) {
      returnValue.persistedState = new ExperimentPagePersistedState({
        ...Utils.getSearchParamsFromUrl(props.location.search),
      }).toJSON();
      returnValue.urlState = props.location.search;
    }

    return returnValue;
  }

  componentWillUnmount() {
    clearInterval(this.pollTimer);
    this.pollTimer = null;
    this.snapshotComponentState();
  }

  searchModelVersionsRequestId = getUUID();
  loadMoreRunsRequestId = getUUID();

  generateGetExperimentRequestIds() {
    // On OSS, we need to call `getExperimentApi` for each experiment ID
    return this.props.experimentIds.map((_experimentId) => getUUID());
  }

  loadData() {
    const { experimentIds } = this.props;
    experimentIds.map((experimentId, index) =>
      this.props
        .getExperimentApi(experimentId, this.state.getExperimentRequestIds[index])
        .catch((e) => {
          console.error(e);
        }),
    );

    this.handleGettingRuns(this.props.searchRunsApi, this.state.searchRunsRequestId);
  }

  maybeReloadData(prevProps, prevState) {
    if (!_.isEqual(this.props.experimentIds, prevProps.experimentIds)) {
      this.loadData();
    } else if (this.filtersDidUpdate(prevState) || this.lastRunsRefreshTimeDidUpdate(prevState)) {
      // Reload data if filter state change requires it
      this.handleGettingRuns(this.props.searchRunsApi, this.state.searchRunsRequestId);
    }
  }

  lastRunsRefreshTimeDidUpdate(prevState) {
    return this.state.lastRunsRefreshTime !== prevState.lastRunsRefreshTime;
  }

  filtersDidUpdate(prevState) {
    const { persistedState } = this.state;
    return (
      persistedState.searchInput !== prevState.persistedState.searchInput ||
      persistedState.orderByKey !== prevState.persistedState.orderByKey ||
      persistedState.orderByAsc !== prevState.persistedState.orderByAsc ||
      persistedState.startTime !== prevState.persistedState.startTime ||
      persistedState.lifecycleFilter !== prevState.persistedState.lifecycleFilter ||
      persistedState.modelVersionFilter !== prevState.persistedState.modelVersionFilter
    );
  }

  updateNumRunsFromLatestSearch = (response = {}) => {
    const { value } = response;
    if (value && value.runs) {
      this.setState({ numRunsFromLatestSearch: response.value.runs.length });
    }
    return response;
  };

  updateNextPageToken = (response = {}) => {
    const { value } = response;
    let nextPageToken = null;
    if (value && value.next_page_token) {
      nextPageToken = value.next_page_token;
    }
    this.setState({ nextPageToken, loadingMore: false });
    return response;
  };

  fetchModelVersionsForRuns = (response = {}) => {
    const { value } = response;
    if (value) {
      const { runs } = value;
      if (runs && runs.length > 0) {
        _.chunk(runs, MAX_RUNS_IN_SEARCH_MODEL_VERSIONS_FILTER).forEach((runsChunk) => {
          this.props.searchModelVersionsApi(
            { run_id: runsChunk.map((run) => run.info.run_id) },
            this.searchModelVersionsRequestId,
          );
        });
      }
    }

    return response;
  };

  static StartTimeColumnOffset = {
    ALL: null,
    LAST_HOUR: 1 * 60 * 60 * 1000,
    LAST_24_HOURS: 24 * 60 * 60 * 1000,
    LAST_7_DAYS: 7 * 24 * 60 * 60 * 1000,
    LAST_30_DAYS: 30 * 24 * 60 * 60 * 1000,
    LAST_YEAR: 12 * 30 * 24 * 60 * 60 * 1000,
  };

  getStartTimeExpr() {
    const startTimeColumnOffset = ExperimentPage.StartTimeColumnOffset;
    const { startTime } = this.state.persistedState;
    const offset = startTimeColumnOffset[startTime];
    if (!startTime || !offset || startTime === 'ALL') {
      return null;
    }
    const startTimeOffset = new Date() - offset;

    return `attributes.start_time >= ${startTimeOffset}`;
  }

  handleGettingRuns = (getRunsAction, requestId) => {
    const { persistedState, nextPageToken } = this.state;
    const { searchInput, lifecycleFilter } = persistedState;
    const viewType = lifecycleFilterToRunViewType(lifecycleFilter);
    const orderBy = this.getOrderByExpr();
    const startTime = this.getStartTimeExpr();
    let filter = searchInput;
    if (startTime) {
      if (filter.length > 0) {
        filter = `${filter} and ${startTime}`;
      } else {
        filter = startTime;
      }
    }
    const shouldFetchParents = this.shouldNestChildrenAndFetchParents();
    return getRunsAction({
      filter,
      runViewType: viewType,
      experimentIds: this.props.experimentIds,
      orderBy,
      pageToken: nextPageToken,
      shouldFetchParents,
      id: requestId,
    })
      .then(this.updateNextPageToken)
      .then(this.updateNumRunsFromLatestSearch)
      .then(this.fetchModelVersionsForRuns)
      .catch((e) => {
        Utils.logErrorAndNotifyUser(e);
        this.setState({ ...PAGINATION_DEFAULT_STATE });
      });
  };

  handleLoadMoreRuns = () => {
    this.setState({ loadingMore: true });
    this.handleGettingRuns(this.props.loadMoreRunsApi, this.loadMoreRunsRequestId);
  };

  /*
    If this function returns true, the ExperimentView should nest children underneath their parents
    and fetch all root level parents of visible runs. If this function returns false, the views will
    not nest children or fetch any additional parents. Will always return true if the orderByKey is
    'attributes.start_time'
  */
  shouldNestChildrenAndFetchParents() {
    const { orderByKey, searchInput } = this.state.persistedState;
    return (!orderByKey && !searchInput) || orderByKey === ATTRIBUTE_COLUMN_SORT_KEY.DATE;
  }

  onSearch = (searchValue) => {
    const { persistedState } = this.state;
    this.setState(
      (prevState) => ({
        lastRunsRefreshTime: Date.now(),
        numberOfNewRuns: 0,
        persistedState: new ExperimentPagePersistedState({
          ...persistedState,
          ...searchValue,
        }).toJSON(),
        nextPageToken: null,
        pollingState: {
          ...prevState.pollingState,
          newRuns: true,
        },
      }),
      () => {
        this.updateUrlWithViewState();
        this.snapshotComponentState();
      },
    );
  };

  onClear = () => {
    // When user clicks "Clear", preserve multicolumn toggle state but reset other persisted state
    // attributes to their default values.
    this.setState(
      {
        lastRunsRefreshTime: Date.now(),
        numberOfNewRuns: 0,
        persistedState: new ExperimentPagePersistedState({
          showMultiColumns: this.state.persistedState.showMultiColumns,
        }).toJSON(),
        nextPageToken: null,
      },
      () => {
        this.updateUrlWithViewState();
        this.snapshotComponentState();
      },
    );
  };

  setShowMultiColumns = (value) => {
    this.setState(
      {
        persistedState: new ExperimentPagePersistedState({
          ...this.state.persistedState,
          showMultiColumns: value,
        }).toJSON(),
      },
      () => {
        this.updateUrlWithViewState();
        this.snapshotComponentState();
      },
    );
  };

  handleColumnSelectionCheck = (categorizedUncheckedKeys) => {
    this.setState(
      {
        persistedState: new ExperimentPagePersistedState({
          ...this.state.persistedState,
          categorizedUncheckedKeys,
        }).toJSON(),
      },
      () => {
        this.updateUrlWithViewState();
        this.snapshotComponentState();
      },
    );
  };

  handleDiffSwitchChange = (switchPersistedState) => {
    this.setState(
      {
        persistedState: new ExperimentPagePersistedState({
          ...this.state.persistedState,
          diffSwitchSelected: !this.state.persistedState.diffSwitchSelected,
          ...switchPersistedState,
        }).toJSON(),
      },
      () => {
        this.handleColumnSelectionCheck(switchPersistedState.categorizedUncheckedKeys);
      },
    );
  };

  updateUrlWithViewState = () => {
    const getCategorizedUncheckedKeysForUrl = (keys) => {
      // Empty arrays are set to an array with a single null value
      // so that the object can be stringified to the urlState
      return {
        [COLUMN_TYPES.ATTRIBUTES]: _.isEmpty(keys[COLUMN_TYPES.ATTRIBUTES])
          ? [null]
          : keys[COLUMN_TYPES.ATTRIBUTES],
        [COLUMN_TYPES.PARAMS]: _.isEmpty(keys[COLUMN_TYPES.PARAMS])
          ? [null]
          : keys[COLUMN_TYPES.PARAMS],
        [COLUMN_TYPES.METRICS]: _.isEmpty(keys[COLUMN_TYPES.METRICS])
          ? [null]
          : keys[COLUMN_TYPES.METRICS],
        [COLUMN_TYPES.TAGS]: _.isEmpty(keys[COLUMN_TYPES.TAGS]) ? [null] : keys[COLUMN_TYPES.TAGS],
      };
    };

    const { persistedState } = this.state;
    const { experimentIds, history } = this.props;
    persistedState.categorizedUncheckedKeys = getCategorizedUncheckedKeysForUrl(
      persistedState.categorizedUncheckedKeys,
    );
    persistedState.preSwitchCategorizedUncheckedKeys = getCategorizedUncheckedKeysForUrl(
      persistedState.preSwitchCategorizedUncheckedKeys,
    );
    persistedState.postSwitchCategorizedUncheckedKeys = getCategorizedUncheckedKeysForUrl(
      persistedState.postSwitchCategorizedUncheckedKeys,
    );

    const params = Utils.getSearchUrlFromState(persistedState);
    const newUrl = this.props.compareExperiments
      ? `/compare-experiments/s?${this.getExperimentsParam(experimentIds)}&${params}`
      : `/experiments/${experimentIds[0]}/s?${params}`;
    if (newUrl !== history.location.pathname + history.location.search) {
      history.push(newUrl);
    }
  };

  /*
  The component will call pollInfo for any info it needs to poll, and pollingState is responsible
  for keeping track of which things to poll.
  */
  async pollInfo() {
    if (Utils.isBrowserTabVisible()) {
      const promiseArray = [];
      if (this.state.pollingState.newRuns) {
        promiseArray.push(this.pollNewRuns());
      }
      await Promise.all(promiseArray);
    }
  }

  async pollNewRuns() {
    const lastRunsRefreshTime = this.state.lastRunsRefreshTime || 0;
    const latestRuns = await this.props.searchForNewRuns({
      experimentIds: this.props.experimentIds,
      maxResults: MAX_DETECT_NEW_RUNS_RESULTS,
    });
    let numberOfNewRuns = 0;
    if (latestRuns && latestRuns.runs) {
      numberOfNewRuns = latestRuns.runs.filter((run) => isNewRun(lastRunsRefreshTime, run)).length;
    }

    this.setState((previousState) => {
      if (previousState.numberOfNewRuns !== numberOfNewRuns) {
        if (numberOfNewRuns >= MAX_DETECT_NEW_RUNS_RESULTS) {
          return {
            pollingState: {
              ...previousState.pollingState,
              newRuns: false,
            },
            numberOfNewRuns: numberOfNewRuns,
          };
        }
        return { numberOfNewRuns };
      }

      // Don't re-render the component if the state is exactly the same
      return null;
    });
  }

  getOrderByExpr() {
    const { orderByKey, orderByAsc } = this.state.persistedState;
    let orderBy = [];
    if (orderByKey) {
      if (orderByAsc) {
        orderBy = [orderByKey + ' ASC'];
      } else {
        orderBy = [orderByKey + ' DESC'];
      }
    }
    return orderBy;
  }

  renderExperimentView = (isLoading, shouldRenderError, requests) => {
    let searchRunsError;
    const getExperimentRequests = requests.filter((req) =>
      this.state.getExperimentRequestIds.includes(req.id),
    );

    if (shouldRenderError) {
      const searchRunsRequest = Utils.getRequestWithId(requests, this.state.searchRunsRequestId);
      const permissionDeniedRequests = getExperimentRequests.filter(
        (req) => req.error && req.error.getErrorCode() === ErrorCodes.PERMISSION_DENIED,
      );
      if (permissionDeniedRequests.length > 0) {
        const errorMessage = permissionDeniedRequests
          .map((req) => req.error.getMessageField())
          .join('\n');
        return <PermissionDeniedView errorMessage={errorMessage} />;
      } else if (searchRunsRequest && searchRunsRequest.error) {
        searchRunsError = searchRunsRequest.error.getMessageField();
      } else {
        return undefined;
      }
    }
    if (getExperimentRequests.length === 0 || getExperimentRequests.some((req) => req.active)) {
      return <Spinner />;
    }

    const {
      searchInput,
      orderByKey,
      orderByAsc,
      startTime,
      lifecycleFilter,
      modelVersionFilter,
      showMultiColumns,
      categorizedUncheckedKeys,
      diffSwitchSelected,
      preSwitchCategorizedUncheckedKeys,
      postSwitchCategorizedUncheckedKeys,
    } = this.state.persistedState;

    // In a batch get call to fetch all requested experiments. If the BE doesn't return back
    // all the requested experiments, the UI should throw an error
    if (
      this.props.experiments.filter((experiment) => experiment !== undefined && experiment !== null)
        .length !== this.props.experimentIds.length
    ) {
      const experimentFetchError = this.props.intl.formatMessage({
        defaultMessage: 'Unable to view experiments',
        description: "Error message when all the requested experiments couldn't be fetched",
      });
      throw new Error(experimentFetchError);
    }

    const experimentViewProps = {
      compareExperiments: this.props.compareExperiments,
      experiments: this.props.experiments,
      searchRunsRequestId: this.state.searchRunsRequestId,
      onSearch: this.onSearch,
      onClear: this.onClear,
      setShowMultiColumns: this.setShowMultiColumns,
      handleColumnSelectionCheck: this.handleColumnSelectionCheck,
      handleDiffSwitchChange: this.handleDiffSwitchChange,
      updateUrlWithViewState: this.updateUrlWithViewState,
      searchRunsError: searchRunsError,
      isLoading: isLoading && !searchRunsError,
      searchInput: searchInput,
      orderByKey: orderByKey,
      orderByAsc: orderByAsc,
      startTime: startTime,
      modelVersionFilter: modelVersionFilter,
      lifecycleFilter: lifecycleFilter,
      showMultiColumns: showMultiColumns,
      categorizedUncheckedKeys: categorizedUncheckedKeys,
      diffSwitchSelected: diffSwitchSelected,
      preSwitchCategorizedUncheckedKeys: preSwitchCategorizedUncheckedKeys,
      postSwitchCategorizedUncheckedKeys: postSwitchCategorizedUncheckedKeys,
      nextPageToken: this.state.nextPageToken,
      numRunsFromLatestSearch: this.state.numRunsFromLatestSearch,
      handleLoadMoreRuns: this.handleLoadMoreRuns,
      loadingMore: this.state.loadingMore,
      nestChildren: this.shouldNestChildrenAndFetchParents(orderByKey, searchInput),
      numberOfNewRuns: this.state.numberOfNewRuns,
    };

    return <ExperimentView {...experimentViewProps} />;
  };

  render() {
    return (
      <div className='ExperimentPage runs-table-flex-container' style={{ height: '100%' }}>
        <RequestStateWrapper shouldOptimisticallyRender requestIds={this.getRequestIds()}>
          {this.renderExperimentView}
        </RequestStateWrapper>
      </div>
    );
  }

  getRequestIds() {
    return [...this.state.getExperimentRequestIds, this.state.searchRunsRequestId];
  }
}

const mapStateToProps = (state, ownProps) => {
  const experiments = ownProps.experimentIds.map((id) => getExperiment(id, state));
  return { experiments };
};

const mapDispatchToProps = {
  getExperimentApi,
  searchRunsApi,
  loadMoreRunsApi,
  searchModelVersionsApi,
  setCompareExperiments,
};

export const lifecycleFilterToRunViewType = (lifecycleFilter) => {
  if (lifecycleFilter === LIFECYCLE_FILTER.ACTIVE) {
    return ViewType.ACTIVE_ONLY;
  } else {
    return ViewType.DELETED_ONLY;
  }
};

export default withRouter(connect(mapStateToProps, mapDispatchToProps)(injectIntl(ExperimentPage)));
