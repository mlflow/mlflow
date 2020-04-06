import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import { listExperimentsApi } from '../actions';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import './HomePage.css';
import HomeView from './HomeView';
import { getUUID } from '../../common/utils/ActionUtils';

export class HomePageImpl extends Component {
  static propTypes = {
    dispatchListExperimentsApi: PropTypes.func.isRequired,
    experimentId: PropTypes.number,
  };

  state = {
    listExperimentsRequestId: getUUID(),
  };

  componentWillMount() {
    if (process.env.HIDE_EXPERIMENT_LIST !== 'true') {
      this.props.dispatchListExperimentsApi(this.state.listExperimentsRequestId);
    }
  }

  render() {
    const homeView = <HomeView experimentId={this.props.experimentId}/>;
    return process.env.HIDE_EXPERIMENT_LIST === 'true' ? homeView : (
      <RequestStateWrapper requestIds={[this.state.listExperimentsRequestId]}>
        {homeView}
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
    },
  };
};

export const HomePage = connect(mapStateToProps, mapDispatchToProps)(HomePageImpl);
