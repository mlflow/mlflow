import React, { Component } from 'react';
import './ExperimentPage.css';
import PropTypes from 'prop-types';
import { getExperimentApi, getUUID, searchRunsApi } from '../Actions';
import { connect } from 'react-redux';
import ExperimentView from './ExperimentView';
import RequestStateWrapper from './RequestStateWrapper';
import KeyFilter from '../utils/KeyFilter';
import { ViewType } from '../sdk/MlflowEnums';
import LocalStorageUtils from "../utils/LocalStorageUtils";
import { ExperimentPagePersistedState } from "../sdk/MlflowLocalStorageMessages";
import Utils from "../utils/Utils";
import ErrorCodes from "../sdk/ErrorCodes";
import PermissionDeniedView from "./PermissionDeniedView";
import {Spinner} from "./Spinner";

export const LIFECYCLE_FILTER = { ACTIVE: 'Active', DELETED: 'Deleted' };

class ExperimentPage extends Component {
  constructor(props) {
    super(props);
    this.onSearch = this.onSearch.bind(this);
    this.getRequestIds = this.getRequestIds.bind(this);
    const store = ExperimentPage.getLocalStore(this.props.experimentId);
    // Load state data persisted in localStorage. If data isn't present in localStorage (e.g. the
    // first time we construct this component in a browser), the default values in
    // ExperimentPagePersistedState will take precedence.
    const persistedState = new ExperimentPagePersistedState(store.loadComponentState());
    this.state = {
      ...ExperimentPage.getDefaultUnpersistedState(),
      persistedState: persistedState.toJSON(),
    };
  }

  static propTypes = {
    experimentId: PropTypes.number.isRequired,
    dispatchSearchRuns: PropTypes.func.isRequired,
  };

  /** Returns default values for state attributes that aren't persisted in local storage. */
  static getDefaultUnpersistedState() {
    return {
      // String UUID associated with a GetExperiment API request
      getExperimentRequestId: getUUID(),
      // String UUID associated with a SearchRuns API request
      searchRunsRequestId: getUUID(),
      // Last experiment, if any, displayed by this instance of ExperimentPage
      lastExperimentId: undefined,
      // Lifecycle filter of runs to display
      lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
    };
  }

  /**
   * Returns a LocalStorageStore instance that can be used to persist data associated with the
   * ExperimentPage component (e.g. component state like metric/param filter info), for the
   * specified experiment.
   */
  static getLocalStore(experimentId) {
    return LocalStorageUtils.getStoreForComponent("ExperimentPage", experimentId);
  }

  snapshotComponentState() {
    const store = ExperimentPage.getLocalStore(this.props.experimentId);
    store.saveComponentState(new ExperimentPagePersistedState(this.state.persistedState));
  }

  componentDidUpdate() {
    this.snapshotComponentState();
  }

  componentWillUnmount() {
    // Snapshot component state on unmounts to ensure we've captured component state in cases where
    // componentDidUpdate doesn't fire.
    this.snapshotComponentState();
  }

  static getDerivedStateFromProps(props, state) {
    if (props.experimentId !== state.lastExperimentId) {
      const store = ExperimentPage.getLocalStore(props.experimentId);
      const loadedState = new ExperimentPagePersistedState(store.loadComponentState()).toJSON();
      const newState = {
        ...ExperimentPage.getDefaultUnpersistedState(),
        persistedState: loadedState,
        lastExperimentId: props.experimentId,
        lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
      };
      props.dispatch(getExperimentApi(props.experimentId, newState.getExperimentRequestId));
      props.dispatch(searchRunsApi(
        [props.experimentId],
        newState.persistedState.searchInput,
        lifecycleFilterToRunViewType(newState.lifecycleFilter),
        newState.searchRunsRequestId));
      return newState;
    }
    return null;
  }

  onSearch(paramKeyFilterString, metricKeyFilterString, searchInput, lifecycleFilterInput) {
    this.setState({
      persistedState: new ExperimentPagePersistedState({
        paramKeyFilterString,
        metricKeyFilterString,
        searchInput,
      }).toJSON(),
      lifecycleFilter: lifecycleFilterInput,
    });
    const searchRunsRequestId = this.props.dispatchSearchRuns(
      this.props.experimentId, searchInput, lifecycleFilterInput);
    this.setState({ searchRunsRequestId });
  }

  render() {
    return (
      <div className="ExperimentPage runs-table-flex-container" style={{height: "100%"}}>
        <RequestStateWrapper shouldOptimisticallyRender requestIds={this.getRequestIds()}>
          {(isLoading, shouldRenderError, requests) => {
            let searchRunsError;
            const getExperimentRequest = Utils.getRequestWithId(
              requests, this.state.getExperimentRequestId);
            if (shouldRenderError) {
              const searchRunsRequest = Utils.getRequestWithId(
                requests, this.state.searchRunsRequestId);
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
            if (getExperimentRequest.active) {
              return <Spinner/>;
            }
            return <ExperimentView
              paramKeyFilter={new KeyFilter(this.state.persistedState.paramKeyFilterString)}
              metricKeyFilter={new KeyFilter(this.state.persistedState.metricKeyFilterString)}
              experimentId={this.props.experimentId}
              searchRunsRequestId={this.state.searchRunsRequestId}
              lifecycleFilter={this.state.lifecycleFilter}
              onSearch={this.onSearch}
              searchRunsError={searchRunsError}
              searchInput={this.state.persistedState.searchInput}
              isLoading={isLoading && !searchRunsError}
            />;
          }}
        </RequestStateWrapper>
      </div>
    );
  }

  getRequestIds() {
    return [this.state.getExperimentRequestId, this.state.searchRunsRequestId];
  }
}

const mapDispatchToProps = (dispatch) => {
  return {
    dispatch,
    dispatchSearchRuns: (experimentId, filter, lifecycleFilterInput) => {
      const requestId = getUUID();
      dispatch(searchRunsApi([experimentId], filter,
        lifecycleFilterToRunViewType(lifecycleFilterInput), requestId));
      return requestId;
    }
  };
};

const lifecycleFilterToRunViewType = (lifecycleFilter) => {
  if (lifecycleFilter === LIFECYCLE_FILTER.ACTIVE) {
    return ViewType.ACTIVE_ONLY;
  } else {
    return ViewType.DELETED_ONLY;
  }
};

export default connect(undefined, mapDispatchToProps)(ExperimentPage);
