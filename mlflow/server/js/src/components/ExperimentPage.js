import React, { Component } from 'react';
import './ExperimentPage.css';
import PropTypes from 'prop-types';
import { getExperimentApi, getUUID, searchRunsApi } from '../Actions';
import { connect } from 'react-redux';
import ExperimentView from './ExperimentView';
import RequestStateWrapper from './RequestStateWrapper';
import KeyFilter from '../utils/KeyFilter';
import { ViewType } from '../sdk/MlflowEnums';


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

  state = {
    paramKeyFilter: new KeyFilter(),
    metricKeyFilter: new KeyFilter(),
    getExperimentRequestId: getUUID(),
    searchRunsRequestId: getUUID(),
    searchInput: '',
    lastExperimentId: undefined,
    lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
  };

  static getDerivedStateFromProps(props, state) {
    if (props.experimentId !== state.lastExperimentId) {
      const newState = {
        paramKeyFilter: new KeyFilter(),
        metricKeyFilter: new KeyFilter(),
        getExperimentRequestId: getUUID(),
        searchRunsRequestId: getUUID(),
        searchInput: '',
        lastExperimentId: props.experimentId,
        lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
      };
      props.dispatch(getExperimentApi(props.experimentId, newState.getExperimentRequestId));
      props.dispatch(searchRunsApi(
        [props.experimentId],
        [],
        lifecycleFilterToRunViewType(newState.lifecycleFilter),
        newState.searchRunsRequestId));
      return newState;
    }
    return null;
  }

  onSearch(paramKeyFilter, metricKeyFilter, andedExpressions, searchInput, lifecycleFilterInput) {
    this.setState({
      paramKeyFilter,
      metricKeyFilter,
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
            paramKeyFilter={this.state.paramKeyFilter}
            metricKeyFilter={this.state.metricKeyFilter}
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

const mapStateToProps = (state, ownProps) => {
  const { match } = ownProps;
  if (match.url === "/") {
    return { experimentId: 0 };
  }
  return { experimentId: parseInt(match.params.experimentId, 10) };
};

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

export default connect(mapStateToProps, mapDispatchToProps)(ExperimentPage);
