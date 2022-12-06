import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import qs from 'qs';
import { loadMoreExperimentsApi } from '../actions';
import { getExperiments } from '../reducers/Reducers';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import './HomePage.css';
import HomeView from './HomeView';
import { getUUID } from '../../common/utils/ActionUtils';
import Routes from '../routes';

export class HomePageImpl extends Component {
  static propTypes = {
    history: PropTypes.shape({}),
    dispatchLoadMoreExperimentsApi: PropTypes.func.isRequired,
    experimentIds: PropTypes.arrayOf(PropTypes.string),
    compareExperiments: PropTypes.bool,
  };

  static defaultProps = {
    compareExperiments: false,
  };

  state = {
    searchExperimentsRequestId: getUUID(),
  };

  componentDidMount() {
    if (process.env.HIDE_EXPERIMENT_LIST !== 'true') {
      this.props.dispatchLoadMoreExperimentsApi({ id: this.state.searchExperimentsRequestId });
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
      <RequestStateWrapper
        requestIds={[this.state.searchExperimentsRequestId]}
        // eslint-disable-next-line no-trailing-spaces
      >
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
  const idsFromState = getExperiments(state).map((e) => e.experiment_id);

  // Can only route to experiments they exist in state
  // A new search can clear the state.
  if (
    match.url.startsWith('/experiments') &&
    idsFromState.some((e) => e === match.params.experimentId)
  ) {
    return { experimentIds: [match.params.experimentId], compareExperiments: false };
  }

  if (match.url.startsWith(Routes.compareExperimentsPageRoute)) {
    const { location } = ownProps;
    const searchValues = qs.parse(location.search, { ignoreQueryPrefix: true });
    const experimentIds = JSON.parse(searchValues['experiments']);
    // Make sure we have all of them locally in case the search input
    // changes and removes some.
    const allExperimentsInState = experimentIds.every((e) => idsFromState.includes(e));

    if (allExperimentsInState) {
      return { experimentIds, compareExperiments: true };
    }
  }

  return {};
};

const mapDispatchToProps = (dispatch) => {
  return {
    dispatchLoadMoreExperimentsApi: (params) => {
      return dispatch(loadMoreExperimentsApi(params));
    },
  };
};

export const HomePage = connect(mapStateToProps, mapDispatchToProps)(HomePageImpl);
