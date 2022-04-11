import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import qs from 'qs';
import { listExperimentsApi } from '../actions';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import './HomePage.css';
import HomeView from './HomeView';
import { getUUID } from '../../common/utils/ActionUtils';
import Routes from '../routes';

export class HomePageImpl extends Component {
  static propTypes = {
    history: PropTypes.shape({}),
    dispatchListExperimentsApi: PropTypes.func.isRequired,
    experimentIds: PropTypes.arrayOf(PropTypes.string),
    compareExperiments: PropTypes.bool,
  };

  static defaultProps = {
    compareExperiments: false,
  };

  state = {
    listExperimentsRequestId: getUUID(),
  };

  componentDidMount() {
    if (process.env.HIDE_EXPERIMENT_LIST !== 'true') {
      this.props.dispatchListExperimentsApi(this.state.listExperimentsRequestId);
    }
  }

  render() {
    const homeView = (
      <HomeView
        history={this.props.history}
        experimentIds={this.props.experimentIds}
        compareExperiments={this.props.compareExperiments}
      />
    );
    return process.env.HIDE_EXPERIMENT_LIST === 'true' ? (
      homeView
    ) : (
      <RequestStateWrapper requestIds={[this.state.listExperimentsRequestId]}>
        {homeView}
      </RequestStateWrapper>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { match } = ownProps;
  if (match.url === '/') {
    return {};
  }

  if (match.url.startsWith('/experiments')) {
    return { experimentIds: [match.params.experimentId], compareExperiments: false };
  }

  if (match.url.startsWith(Routes.compareExperimentsPageRoute)) {
    const { location } = ownProps;
    const searchValues = qs.parse(location.search);
    const experimentIds = JSON.parse(searchValues['?experiments']);
    return { experimentIds, compareExperiments: true };
  }

  return {};
};

const mapDispatchToProps = (dispatch) => {
  return {
    dispatchListExperimentsApi: (requestId) => {
      return dispatch(listExperimentsApi(requestId));
    },
  };
};

export const HomePage = connect(mapStateToProps, mapDispatchToProps)(HomePageImpl);
