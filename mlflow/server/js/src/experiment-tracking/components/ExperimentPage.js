import React, { Component } from 'react';
import './ExperimentPage.css';
import PropTypes from 'prop-types';
import { getExperimentApi, searchRunsApi, loadMoreRunsApi } from '../actions';
import { searchModelVersionsApi } from '../../model-registry/actions';
import { connect } from 'react-redux';
import ExperimentView from './ExperimentView';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import KeyFilter from '../utils/KeyFilter';
import { ViewType } from '../sdk/MlflowEnums';
import { ExperimentPagePersistedState } from '../sdk/MlflowLocalStorageMessages';
import Utils from '../../common/utils/Utils';
import { ErrorCodes } from '../../common/constants';
import { PermissionDeniedView } from './PermissionDeniedView';
import { Spinner } from '../../common/components/Spinner';
import { withRouter } from 'react-router-dom';
import { getUUID } from '../../common/utils/ActionUtils';
import { MAX_RUNS_IN_SEARCH_MODEL_VERSIONS_FILTER } from '../../model-registry/constants';
import _ from 'lodash';

export const LIFECYCLE_FILTER = { ACTIVE: 'Active', DELETED: 'Deleted' };
export const MODEL_VERSION_FILTER = {
  WITH_MODEL_VERSIONS: 'With Model Versions',
  WTIHOUT_MODEL_VERSIONS: 'Without Model Versions',
  ALL_RUNS: 'All Runs',
};

export const PAGINATION_DEFAULT_STATE = {
  nextPageToken: null,
  numRunsFromLatestSearch: null, // number of runs returned from the most recent search request
  loadingMore: false,
};

export class ExperimentPage extends Component {
  constructor(props) {
    super(props);
    const urlState = Utils.getSearchParamsFromUrl(props.location.search);
    this.state = {
      // Last experiment, if any, displayed by this instance of ExperimentPage
      lastExperimentId: undefined,
      // Lifecycle filter of runs to display
      lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
      // Filter of model versions to display
      modelVersionFilter: MODEL_VERSION_FILTER.ALL_RUNS,
      ...PAGINATION_DEFAULT_STATE,
      persistedState: {
        paramKeyFilterString: urlState.params === undefined ? '' : urlState.params,
        metricKeyFilterString: urlState.metrics === undefined ? '' : urlState.metrics,
        searchInput: urlState.search === undefined ? '' : urlState.search,
        orderByKey: urlState.orderByKey === undefined ? null : urlState.orderByKey,
        orderByAsc: urlState.orderByAsc === undefined ? true : urlState.orderByAsc === 'true',
      },
    };
  }

  getExperimentRequestId = getUUID();
  searchRunsRequestId = getUUID();
  searchModelVersionsRequestId = getUUID();
  loadMoreRunsRequestId = getUUID();

  loadData() {
    this.props.getExperimentApi(this.props.experimentId, this.getExperimentRequestId).catch((e) => {
      console.error(e);
    });

    this.handleGettingRuns(this.props.searchRunsApi, this.searchRunsRequestId);
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

  handleGettingRuns = (getRunsAction, requestId) => {
    const { persistedState, lifecycleFilter, nextPageToken } = this.state;
    const { searchInput } = persistedState;
    const viewType = lifecycleFilterToRunViewType(lifecycleFilter);
    const orderBy = this.getOrderByExpr();
    const shouldFetchParents = this.shouldNestChildrenAndFetchParents();
    return getRunsAction({
      filter: searchInput,
      runViewType: viewType,
      experimentIds: [this.props.experimentId],
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

  static propTypes = {
    experimentId: PropTypes.string.isRequired,
    getExperimentApi: PropTypes.func.isRequired,
    searchRunsApi: PropTypes.func.isRequired,
    searchModelVersionsApi: PropTypes.func.isRequired,
    loadMoreRunsApi: PropTypes.func.isRequired,
    history: PropTypes.object.isRequired,
    location: PropTypes.object,
  };

  componentDidMount() {
    this.loadData();
  }

  componentDidUpdate(prevProps) {
    this.maybeReloadData(prevProps);
  }

  static getDerivedStateFromProps(props, state) {
    if (props.experimentId !== state.lastExperimentId) {
      return {
        persistedState:
          state.lastExperimentId === undefined
            ? state.persistedState
            : new ExperimentPagePersistedState().toJSON(),
        lastExperimentId: props.experimentId,
        lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
      };
    }
    return null;
  }

  /*
    If this function returns true, the ExperimentViews should nest children underneath their parents
    and fetch all root level parents of visible runs. If this function returns false, the views will
    not nest children nor fetch any additional parents.
  */
  shouldNestChildrenAndFetchParents() {
    const { orderByKey, searchInput } = this.state.persistedState;
    return !orderByKey && !searchInput;
  }

  maybeReloadData(prevProps) {
    if (this.props.experimentId !== prevProps.experimentId) {
      this.loadData();
    }
  }

  onSearch = (
    paramKeyFilterString,
    metricKeyFilterString,
    searchInput,
    lifecycleFilterInput,
    orderByKey,
    orderByAsc,
    modelVersionFilterInput,
  ) => {
    this.updateUrlWithSearchFilter({
      paramKeyFilterString,
      metricKeyFilterString,
      searchInput,
      orderByKey,
      orderByAsc,
    });

    this.setState(
      {
        persistedState: new ExperimentPagePersistedState({
          paramKeyFilterString,
          metricKeyFilterString,
          searchInput,
          orderByKey,
          orderByAsc,
        }).toJSON(),
        lifecycleFilter: lifecycleFilterInput,
        modelVersionFilter: modelVersionFilterInput,
        nextPageToken: null,
      },
      () => this.handleGettingRuns(this.props.searchRunsApi, this.searchRunsRequestId),
    );
  };

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

  updateUrlWithSearchFilter({
    paramKeyFilterString,
    metricKeyFilterString,
    searchInput,
    orderByKey,
    orderByAsc,
  }) {
    const state = {};
    if (paramKeyFilterString) {
      state['params'] = paramKeyFilterString;
    }
    if (metricKeyFilterString) {
      state['metrics'] = metricKeyFilterString;
    }
    if (searchInput) {
      state['search'] = searchInput;
    }
    if (orderByKey) {
      state['orderByKey'] = orderByKey;
    }
    // orderByAsc defaults to true, so only encode it if it is false.
    if (orderByAsc === false) {
      state['orderByAsc'] = orderByAsc;
    }
    const newUrl = `/experiments/${this.props.experimentId}/s?${Utils.getSearchUrlFromState(
      state,
    )}`;
    if (newUrl !== this.props.history.location.pathname + this.props.history.location.search) {
      this.props.history.push(newUrl);
    }
  }

  renderExperimentView = (isLoading, shouldRenderError, requests) => {
    let searchRunsError;
    const getExperimentRequest = Utils.getRequestWithId(requests, this.getExperimentRequestId);

    if (shouldRenderError) {
      const searchRunsRequest = Utils.getRequestWithId(requests, this.searchRunsRequestId);
      if (
        getExperimentRequest.error &&
        getExperimentRequest.error.getErrorCode() === ErrorCodes.PERMISSION_DENIED
      ) {
        return <PermissionDeniedView errorMessage={getExperimentRequest.error.getMessageField()} />;
      } else if (searchRunsRequest.error) {
        searchRunsError = searchRunsRequest.error.getMessageField();
      } else {
        return undefined;
      }
    }
    if (!getExperimentRequest || getExperimentRequest.active) {
      return <Spinner />;
    }

    const {
      paramKeyFilterString,
      metricKeyFilterString,
      searchInput,
      orderByKey,
      orderByAsc,
    } = this.state.persistedState;

    return (
      <ExperimentView
        paramKeyFilter={new KeyFilter(paramKeyFilterString)}
        metricKeyFilter={new KeyFilter(metricKeyFilterString)}
        experimentId={this.props.experimentId}
        searchRunsRequestId={this.searchRunsRequestId}
        modelVersionFilter={this.state.modelVersionFilter}
        lifecycleFilter={this.state.lifecycleFilter}
        onSearch={this.onSearch}
        searchRunsError={searchRunsError}
        searchInput={searchInput}
        isLoading={isLoading && !searchRunsError}
        orderByKey={orderByKey}
        orderByAsc={orderByAsc}
        nextPageToken={this.state.nextPageToken}
        numRunsFromLatestSearch={this.state.numRunsFromLatestSearch}
        handleLoadMoreRuns={this.handleLoadMoreRuns}
        loadingMore={this.state.loadingMore}
        nestChildren={this.shouldNestChildrenAndFetchParents(orderByKey, searchInput)}
      />
    );
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
    return [this.getExperimentRequestId, this.searchRunsRequestId];
  }
}

const mapDispatchToProps = {
  getExperimentApi,
  searchRunsApi,
  loadMoreRunsApi,
  searchModelVersionsApi,
};

export const lifecycleFilterToRunViewType = (lifecycleFilter) => {
  if (lifecycleFilter === LIFECYCLE_FILTER.ACTIVE) {
    return ViewType.ACTIVE_ONLY;
  } else {
    return ViewType.DELETED_ONLY;
  }
};

export default withRouter(connect(undefined, mapDispatchToProps)(ExperimentPage));
