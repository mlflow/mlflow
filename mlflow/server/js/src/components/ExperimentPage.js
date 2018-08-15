import React, { Component } from 'react';
import './ExperimentPage.css';
import PropTypes from 'prop-types';
import { getExperimentApi, getUUID, searchRunsApi } from '../Actions';
import { connect } from 'react-redux';
import ExperimentView from './ExperimentView';
import RequestStateWrapper from './RequestStateWrapper';
import KeyFilter from '../utils/KeyFilter';


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
      };
      props.dispatch(getExperimentApi(props.experimentId, newState.getExperimentRequestId));
      props.dispatch(searchRunsApi([props.experimentId], [], newState.searchRunsRequestId));
      return newState;
    }
    return null;
  }

  onSearch(paramKeyFilter, metricKeyFilter, andedExpressions, searchInput) {
    this.setState({paramKeyFilter, metricKeyFilter, searchInput});
    const searchRunsRequestId = this.props.dispatchSearchRuns(
      this.props.experimentId, andedExpressions);
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

// eslint-disable-next-line no-unused-vars
const mapDispatchToProps = (dispatch, ownProps) => {
  return {
    dispatch,
    dispatchSearchRuns: (experimentId, andedExpressions) => {
      const requestId = getUUID();
      dispatch(searchRunsApi([experimentId], andedExpressions, requestId));
      return requestId;
    }
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(ExperimentPage);
