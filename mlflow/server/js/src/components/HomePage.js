import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { getUUID, listExperimentsApi } from '../Actions';
import RequestStateWrapper from './RequestStateWrapper';
import './HomePage.css';
import HomeView from './HomeView';

class HomePage extends Component {
  static propTypes = {
    dispatchListExperimentsApi: PropTypes.func.isRequired,
    experimentId: PropTypes.number,
  };

  state = {
    listExperimentsRequestId: getUUID(),
  };

  componentWillMount() {
    this.props.dispatchListExperimentsApi(this.state.listExperimentsRequestId);
  }

  render() {
    return (
      <RequestStateWrapper requestIds={[this.state.listExperimentsRequestId]}>
        <HomeView experimentId={this.props.experimentId}/>
      </RequestStateWrapper>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { match } = ownProps;
  if (match.url === "/") {
    return {};
  }
  return { experimentId: parseInt(match.params.experimentId, 10) };
};

const mapDispatchToProps = (dispatch) => {
  return {
    dispatchListExperimentsApi: (requestId) => {
      return dispatch(listExperimentsApi(requestId));
    }
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(HomePage);
