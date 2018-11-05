import React, { Component } from 'react';
import './ExperimentPage.css';
import PropTypes from 'prop-types';
import { getExperimentApi, getUUID, searchRunsApi } from '../Actions';
import { connect } from 'react-redux';
import ExperimentView from './ExperimentView';
import RequestStateWrapper from './RequestStateWrapper';
import KeyFilter from '../utils/KeyFilter';
import { ViewType } from '../sdk/MlflowEnums';
import _ from "lodash";
import {SearchUtils} from "../utils/SearchUtils";
import LocalStorageUtils from "../utils/LocalStorageUtils";


export const LIFECYCLE_FILTER = { ACTIVE: 'Active', DELETED: 'Deleted' };

class ExperimentPage extends Component {
  constructor(props) {
    super(props);
    this.onSearch = this.onSearch.bind(this);
    this.getRequestIds = this.getRequestIds.bind(this);
  }

  static propTypes = {
    experimentId: PropTypes.number.isRequired,
    dispatchSearchRuns: PropTypes.func.isRequired,
  };

  static defaultState = {
    paramKeyFilterString: "",
    metricKeyFilterString: "",
    getExperimentRequestId: getUUID(),
    searchRunsRequestId: getUUID(),
    searchInput: '',
    lastExperimentId: undefined,
    lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
  };

  static getStore(experimentId) {
    return LocalStorageUtils.getStore("ExperimentPage", experimentId);
  }

  store = ExperimentPage.getStore(this.props.experimentId);
  state = LocalStorageUtils.loadComponentState(this.store, ExperimentPage.defaultState);

  /** Wrapper over setState that caches certain fields in local storage. */
  setStateWrapper(newState) {
    const { paramKeyFilterString, metricKeyFilterString, searchInput } = newState;
    this.setState(newState, () => {
      LocalStorageUtils.saveComponentState(this.store, {
        paramKeyFilterString,
        metricKeyFilterString,
        searchInput,
      });
    });
  }

  static getDerivedStateFromProps(props, state) {
    if (props.experimentId !== state.lastExperimentId) {
      const store = ExperimentPage.getStore(props.experimentId);
      const newState = {
        ...LocalStorageUtils.loadComponentState(store, ExperimentPage.defaultState),
        getExperimentRequestId: getUUID(),
        searchRunsRequestId: getUUID(),
        lastExperimentId: props.experimentId,
        lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
      };
      props.dispatch(getExperimentApi(props.experimentId, newState.getExperimentRequestId));
      props.dispatch(searchRunsApi(
        [props.experimentId],
        SearchUtils.parseSearchInput(newState.searchInput),
        lifecycleFilterToRunViewType(newState.lifecycleFilter),
        newState.searchRunsRequestId));
      return newState;
    }
    return null;
  }

  onSearch(paramKeyFilterString, metricKeyFilterString, searchInput, lifecycleFilterInput) {
    const andedExpressions = SearchUtils.parseSearchInput(searchInput);
    this.setStateWrapper({
      paramKeyFilterString,
      metricKeyFilterString,
      searchInput,
      lifecycleFilter: lifecycleFilterInput
    });
    const searchRunsRequestId = this.props.dispatchSearchRuns(
      this.props.experimentId, andedExpressions, lifecycleFilterInput);
    this.setState({ searchRunsRequestId });
  }

  render() {
    return (
      <div className="ExperimentPage">
        <RequestStateWrapper requestIds={this.getRequestIds()}>
          <ExperimentView
            paramKeyFilter={new KeyFilter(this.state.paramKeyFilterString)}
            metricKeyFilter={new KeyFilter(this.state.metricKeyFilterString)}
            experimentId={this.props.experimentId}
            searchRunsRequestId={this.state.searchRunsRequestId}
            lifecycleFilter={this.state.lifecycleFilter}
            onSearch={this.onSearch}
            searchInput={this.state.searchInput}
          />
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
    dispatchSearchRuns: (experimentId, andedExpressions, lifecycleFilterInput) => {
      const requestId = getUUID();
      dispatch(searchRunsApi([experimentId], andedExpressions,
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
