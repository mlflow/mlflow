import React, { Component } from 'react';
import './ExperimentPage.css';
import PropTypes from 'prop-types';
import {
  getExperimentApi,
  getUUID,
  searchRunsApi,
  loadMoreRunsApi,
} from '../Actions';
import { connect } from 'react-redux';
import ExperimentView from './ExperimentView';
import RequestStateWrapper from './RequestStateWrapper';
import KeyFilter from '../utils/KeyFilter';
import { ViewType } from '../sdk/MlflowEnums';
import { ExperimentPagePersistedState } from "../sdk/MlflowLocalStorageMessages";
import Utils from "../utils/Utils";
import ErrorCodes from "../sdk/ErrorCodes";
import PermissionDeniedView from "./PermissionDeniedView";
import {Spinner} from "./Spinner";
import { withRouter } from 'react-router-dom';

export const LIFECYCLE_FILTER = { ACTIVE: 'Active', DELETED: 'Deleted' };

export class ExperimentPage extends Component {
  constructor(props) {
    super(props);
    const urlState = Utils.getSearchParamsFromUrl(props.location.search);
    this.state = {
      ...ExperimentPage.getDefaultUnpersistedState(),
      persistedState: {
        paramKeyFilterString: urlState.params === undefined ? "" : urlState.params,
        metricKeyFilterString: urlState.metrics === undefined ? "" : urlState.metrics,
        searchInput: urlState.search === undefined ? "" : urlState.search,
        orderByKey: urlState.orderByKey === undefined ? null : urlState.orderByKey,
        orderByAsc: urlState.orderByAsc === undefined ? true : urlState.orderByAsc === "true",
      },
      nextPageToken: null,
      loadingMore: false,
    };
  }

  getExperimentRequestId = getUUID();
  searchRunsRequestId = getUUID();
  loadMoreRunsRequestId = getUUID();

  loadData() {
    const { persistedState, lifecycleFilter } = this.state;
    const { experimentId } = this.props;
    const { orderByKey, orderByAsc, searchInput } = persistedState;
    const orderBy = ExperimentPage.getOrderByExpr(orderByKey, orderByAsc);
    const viewType = lifecycleFilterToRunViewType(lifecycleFilter);

    this.props.getExperimentApi(experimentId, this.getExperimentRequestId);
    this.props
      .searchRunsApi([experimentId], searchInput, viewType, orderBy, this.searchRunsRequestId)
      .then(this.updateNextPageToken)
      .catch((e) => {
        Utils.logErrorAndNotifyUser(e);
        this.setState({ nextPageToken: null, loadingMore: false });
      });
  }

  updateNextPageToken = (response = {}) => {
    const { value } = response;
    let nextPageToken = null;
    if (value && value.next_page_token) {
      nextPageToken = value.next_page_token;
    }
    this.setState({ nextPageToken, loadingMore: false });
  };

  handleLoadMoreRuns = () => {
    const { experimentId } = this.props;
    const { persistedState, lifecycleFilter, nextPageToken } = this.state;
    const { orderByKey, orderByAsc, searchInput } = persistedState;
    const orderBy = ExperimentPage.getOrderByExpr(orderByKey, orderByAsc);
    const viewType = lifecycleFilterToRunViewType(lifecycleFilter);
    this.setState({ loadingMore: true });
    this.props
      .loadMoreRunsApi(
        [experimentId],
        searchInput,
        viewType,
        orderBy,
        nextPageToken,
        this.loadMoreRunsRequestId,
      )
      .then(this.updateNextPageToken)
      .catch((e) => {
        Utils.logErrorAndNotifyUser(e);
        this.setState({ nextPageToken: null, loadingMore: false });
      });
  };

  static propTypes = {
    experimentId: PropTypes.number.isRequired,
    getExperimentApi: PropTypes.func.isRequired,
    searchRunsApi: PropTypes.func.isRequired,
    loadMoreRunsApi: PropTypes.func.isRequired,
    history: PropTypes.object.isRequired,
    location: PropTypes.object,
  };

  /** Returns default values for state attributes that aren't persisted in the URL. */
  static getDefaultUnpersistedState() {
    return {
      // Last experiment, if any, displayed by this instance of ExperimentPage
      lastExperimentId: undefined,
      // Lifecycle filter of runs to display
      lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
    };
  }

  componentDidMount() {
    this.loadData();
  }

  componentDidUpdate(prevProps) {
    this.maybeReloadData(prevProps);
  }

  static getDerivedStateFromProps(props, state) {
    if (props.experimentId !== state.lastExperimentId) {
      return {
        ...ExperimentPage.getDefaultUnpersistedState(),
        persistedState: state.lastExperimentId === undefined ?
            state.persistedState : (new ExperimentPagePersistedState()).toJSON(),
        lastExperimentId: props.experimentId,
        lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
      };
    }
    return null;
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
      orderByAsc
  ) => {
    this.setState({
      persistedState: new ExperimentPagePersistedState({
        paramKeyFilterString,
        metricKeyFilterString,
        searchInput,
        orderByKey,
        orderByAsc,
      }).toJSON(),
      lifecycleFilter: lifecycleFilterInput,
    });

    const orderBy = ExperimentPage.getOrderByExpr(orderByKey, orderByAsc);
    this.props
      .searchRunsApi(
        [this.props.experimentId],
        searchInput,
        lifecycleFilterToRunViewType(lifecycleFilterInput),
        orderBy,
        this.searchRunsRequestId,
      )
      .then(this.updateNextPageToken)
      .catch((e) => {
        Utils.logErrorAndNotifyUser(e);
        this.setState({ nextPageToken: null, loadingMore: false });
      });

    this.updateUrlWithSearchFilter({
      paramKeyFilterString,
      metricKeyFilterString,
      searchInput,
      orderByKey,
      orderByAsc,
    });
  };

  static getOrderByExpr(orderByKey, orderByAsc) {
    let orderBy = [];
    if (orderByKey) {
      if (orderByAsc) {
        orderBy = [orderByKey + " ASC"];
      } else {
        orderBy = [orderByKey + " DESC"];
      }
    }
    return orderBy;
  }

  updateUrlWithSearchFilter(
      {paramKeyFilterString, metricKeyFilterString, searchInput, orderByKey, orderByAsc}) {
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
    const newUrl = `/experiments/${this.props.experimentId}` +
      `/s?${Utils.getSearchUrlFromState(state)}`;
    if (newUrl !== (this.props.history.location.pathname
      + this.props.history.location.search)) {
      this.props.history.push(newUrl);
    }
  }

  render() {
    return (
      <div className="ExperimentPage runs-table-flex-container" style={{height: "100%"}}>
        <RequestStateWrapper shouldOptimisticallyRender requestIds={this.getRequestIds()}>
          {(isLoading, shouldRenderError, requests) => {
            let searchRunsError;
            const getExperimentRequest = Utils.getRequestWithId(
              requests, this.getExperimentRequestId);
            if (shouldRenderError) {
              const searchRunsRequest = Utils.getRequestWithId(
                requests, this.searchRunsRequestId);
              if (searchRunsRequest.error) {
                searchRunsError = searchRunsRequest.error.getMessageField();
              } else if (getExperimentRequest.error.getErrorCode() ===
                  ErrorCodes.PERMISSION_DENIED) {
                return (<PermissionDeniedView
                  errorMessage={getExperimentRequest.error.xhr.responseJSON.message}
                />);
              } else {
                return undefined;
              }
            }
            if (!getExperimentRequest || getExperimentRequest.active) {
              return <Spinner/>;
            }

            return <ExperimentView
              paramKeyFilter={new KeyFilter(this.state.persistedState.paramKeyFilterString)}
              metricKeyFilter={new KeyFilter(this.state.persistedState.metricKeyFilterString)}
              experimentId={this.props.experimentId}
              searchRunsRequestId={this.searchRunsRequestId}
              lifecycleFilter={this.state.lifecycleFilter}
              onSearch={this.onSearch}
              searchRunsError={searchRunsError}
              searchInput={this.state.persistedState.searchInput}
              isLoading={isLoading && !searchRunsError}
              orderByKey={this.state.persistedState.orderByKey}
              orderByAsc={this.state.persistedState.orderByAsc}
              nextPageToken={this.state.nextPageToken}
              handleLoadMoreRuns={this.handleLoadMoreRuns}
              loadingMore={this.state.loadingMore}
            />;
          }}
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
};

const lifecycleFilterToRunViewType = (lifecycleFilter) => {
  if (lifecycleFilter === LIFECYCLE_FILTER.ACTIVE) {
    return ViewType.ACTIVE_ONLY;
  } else {
    return ViewType.DELETED_ONLY;
  }
};

export default withRouter(connect(undefined, mapDispatchToProps)(ExperimentPage));
