import React, { Component } from 'react';
import { connect } from 'react-redux';
import qs from 'qs';
import { searchExperimentsApi } from '../actions';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import './HomePage.css';
import HomeView from './HomeView';
import { getUUID } from '../../common/utils/ActionUtils';
import Routes from '../routes';

type OwnHomePageImplProps = {
  history?: Record<string, never>;
  dispatchSearchExperimentsApi: (...args: any[]) => any;
  experimentIds?: string[];
  compareExperiments?: boolean;
};

type HomePageImplState = any;

type HomePageImplProps = OwnHomePageImplProps & typeof HomePageImpl.defaultProps;

export class HomePageImpl extends Component<HomePageImplProps, HomePageImplState> {
  static defaultProps = {
    compareExperiments: false,
  };

  state = {
    searchExperimentsRequestId: getUUID(),
  };

  componentDidMount() {
    // @ts-expect-error TS(4111): Property 'HIDE_EXPERIMENT_LIST' comes from an inde... Remove this comment to see the full error message
    if (process.env.HIDE_EXPERIMENT_LIST !== 'true') {
      this.props.dispatchSearchExperimentsApi(this.state.searchExperimentsRequestId);
    }
  }

  render() {
    const homeView = (
      <HomeView
        // @ts-expect-error TS(2322): Type '{ history: Record<string, never> | undefined... Remove this comment to see the full error message
        history={this.props.history}
        experimentIds={this.props.experimentIds}
        compareExperiments={this.props.compareExperiments}
      />
    );
    // @ts-expect-error TS(4111): Property 'HIDE_EXPERIMENT_LIST' comes from an inde... Remove this comment to see the full error message
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

const mapStateToProps = (state: any, ownProps: any) => {
  const { match } = ownProps;
  if (match.url === '/') {
    return {};
  }

  if (match.url.startsWith('/experiments')) {
    return { experimentIds: [match.params.experimentId], compareExperiments: false };
  }

  if (match.url.startsWith(Routes.compareExperimentsPageRoute)) {
    const { location } = ownProps;
    const searchValues = qs.parse(location.search, { ignoreQueryPrefix: true });
    // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
    const experimentIds = JSON.parse(searchValues['experiments']);
    return { experimentIds, compareExperiments: true };
  }

  return {};
};

const mapDispatchToProps = (dispatch: any) => {
  return {
    dispatchSearchExperimentsApi: (requestId: any) => {
      return dispatch(searchExperimentsApi(requestId));
    },
  };
};

// @ts-expect-error TS(2345): Argument of type 'typeof HomePageImpl' is not assi... Remove this comment to see the full error message
export const HomePage = connect(mapStateToProps, mapDispatchToProps)(HomePageImpl);
