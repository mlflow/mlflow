/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { connect } from 'react-redux';
import type { Dispatch } from 'redux';
import qs from 'qs';
import { searchExperimentsApi } from '../actions';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import './HomePage.css';
import HomeView from './HomeView';
import { getUUID } from '../../common/utils/ActionUtils';
import Routes from '../routes';
import { withRouterNext } from '../../common/utils/withRouterNext';
import type { WithRouterNextProps } from '../../common/utils/withRouterNext';

type HomePageImplProps = {
  dispatchSearchExperimentsApi: (...args: any[]) => any;
  experimentIds?: string[];
  compareExperiments?: boolean;
};

type HomePageImplState = {
  searchExperimentsRequestId: string;
};

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

const mapStateToProps = (
  state: any,
  { location, params }: WithRouterNextProps<{ experimentId: string }>,
) => {
  if (location.pathname === '/') {
    return {};
  }

  if (location.pathname.startsWith('/experiments')) {
    return { experimentIds: [params.experimentId], compareExperiments: false };
  }

  if (location.pathname.startsWith(Routes.compareExperimentsPageRoute)) {
    const searchValues = qs.parse(location.search, { ignoreQueryPrefix: true });
    if (searchValues['experiments']) {
      const experimentIds = JSON.parse(searchValues['experiments'].toString());
      return { experimentIds, compareExperiments: true };
    }
  }

  return {};
};

const mapDispatchToProps = (dispatch: Dispatch) => {
  return {
    dispatchSearchExperimentsApi: (requestId: any) => {
      return dispatch(searchExperimentsApi(requestId));
    },
  };
};

export const HomePage = withRouterNext(connect(mapStateToProps, mapDispatchToProps)(HomePageImpl));
export default HomePage;
