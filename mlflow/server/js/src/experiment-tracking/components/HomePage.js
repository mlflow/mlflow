import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import qs from 'qs';
import { loadMoreExperimentsApi, getExperimentApi } from '../actions';
import { getExperiments } from '../reducers/Reducers';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import './HomePage.css';
import HomeView from './HomeView';
import { getUUID } from '../../common/utils/ActionUtils';
import Routes from '../routes';

export class HomePageImpl extends Component {
  static propTypes = {
    history: PropTypes.shape({}),
    experimentIds: PropTypes.arrayOf(PropTypes.string),
    compareExperiments: PropTypes.bool,
    dispatchLoadMoreExperimentsApi: PropTypes.func.isRequired,
    dispatchGetExperimentApi: PropTypes.func.isRequired,
    allExperimentIds: PropTypes.arrayOf(PropTypes.string).isRequired,
  };

  static defaultProps = {
    compareExperiments: false,
  };

  state = {
    searchExperimentsRequestId: getUUID(),
  };

  constructor(props) {
    super(props);
    this.requestIds = [];
  }

  fetchExperiments() {
    const { allExperimentIds, experimentIds } = this.props;
    const notInState = experimentIds.filter((id) => !allExperimentIds.includes(id));
    return notInState.map((experimentId) => {
      const id = getUUID();
      this.props.dispatchGetExperimentApi(experimentId, id);
      return id;
    });
  }
  componentDidMount() {
    if (process.env.HIDE_EXPERIMENT_LIST !== 'true') {
      this.requestIds.push(this.state.searchExperimentsRequestId);
      this.props.dispatchLoadMoreExperimentsApi({ id: this.state.searchExperimentsRequestId });
    }
    // No experiemnts and no ids
    if (typeof this.props.experimentIds === 'undefined') {
      return;
    }
    if (this.props.experimentIds.length > 0) {
      const getExperimentsRequestIds = this.fetchExperiments();
      this.requestIds.push(...getExperimentsRequestIds);
    }
  }

  renderPageContent() {
    const { history, experimentIds, compareExperiments } = this.props;
    return (
      <HomeView
        history={history}
        experimentIds={experimentIds}
        compareExperiments={compareExperiments}
      />
    );
  }
  render() {
    return (
      <RequestStateWrapper requestIds={this.requestIds}>
        {this.renderPageContent()}
      </RequestStateWrapper>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { match } = ownProps;
  // TODO need to figure out how to get a new one that isn't in the state here.
  const experiments = getExperiments(state);
  const allExperimentIds = experiments.map((e) => e.experiment_id) || [];
  const props = { allExperimentIds, experimentIds: [] };
  if (match.url === '/') {
    return props;
  }
  // Can only route to experiments they exist in state
  // A new search can clear the state.
  if (match.url.startsWith('/experiments')) {
    return {
      ...props,
      ...{ experimentIds: [match.params.experimentId], compareExperiments: false },
    };
  }

  if (match.url.startsWith(Routes.compareExperimentsPageRoute)) {
    const { location } = ownProps;
    const searchValues = qs.parse(location.search, { ignoreQueryPrefix: true });
    const experimentIds = JSON.parse(searchValues['experiments']);
    return { ...props, ...{ experimentIds, compareExperiments: true } };
  }

  return props;
};

const mapDispatchToProps = (dispatch) => {
  return {
    dispatchLoadMoreExperimentsApi: (params) => {
      return dispatch(loadMoreExperimentsApi(params));
    },
    dispatchGetExperimentApi: (experimentId, id) => {
      return dispatch(getExperimentApi(experimentId, id));
    },
  };
};

export const HomePage = connect(mapStateToProps, mapDispatchToProps)(HomePageImpl);
