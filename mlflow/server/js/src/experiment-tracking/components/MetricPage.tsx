import React, { Component } from 'react';
import { connect } from 'react-redux';
import qs from 'qs';
import { getExperimentApi, getRunApi } from '../actions';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import NotFoundPage from './NotFoundPage';
import { MetricView } from './MetricView';
import { getUUID } from '../../common/utils/ActionUtils';
import { PageContainer } from '../../common/components/PageContainer';

type MetricPageImplProps = {
  runUuids: string[];
  metricKey: string;
  experimentIds?: string[];
  dispatch: (...args: any[]) => any;
};

export class MetricPageImpl extends Component<MetricPageImplProps> {
  requestIds: any;

  constructor(props: MetricPageImplProps) {
    super(props);
    this.requestIds = [];
  }

  fetchExperiments() {
    // @ts-expect-error TS(2532): Object is possibly 'undefined'.
    return this.props.experimentIds.map((experimentId) => {
      const experimentRequestId = getUUID();
      this.props.dispatch(getExperimentApi(experimentId, experimentRequestId));
      return experimentRequestId;
    });
  }

  componentDidMount() {
    if (this.props.experimentIds !== null) {
      const getExperimentsRequestIds = this.fetchExperiments();
      this.requestIds.push(...getExperimentsRequestIds);
    }
    this.props.runUuids.forEach((runUuid) => {
      // Fetch tags for each run. TODO: it'd be nice if we could just fetch the tags directly
      const getRunRequestId = getUUID();
      this.requestIds.push(getRunRequestId);
      this.props.dispatch(getRunApi(runUuid, getRunRequestId));
    });
  }

  renderPageContent() {
    const { runUuids } = this.props;
    return runUuids.length >= 1 ? (
      <MetricView
        runUuids={this.props.runUuids}
        metricKey={this.props.metricKey}
        experimentIds={this.props.experimentIds}
      />
    ) : (
      <NotFoundPage />
    );
  }

  render() {
    return (
      <PageContainer>
        <RequestStateWrapper
          requestIds={this.requestIds}
          // eslint-disable-next-line no-trailing-spaces
        >
          {this.renderPageContent()}
        </RequestStateWrapper>
      </PageContainer>
    );
  }
}

const mapStateToProps = (state: any, ownProps: any) => {
  const { match, location } = ownProps;
  const searchValues = qs.parse(location.search);
  // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
  const runUuids = JSON.parse(searchValues['?runs']);
  let experimentIds = null;
  if (searchValues.hasOwnProperty('experiments')) {
    // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
    experimentIds = JSON.parse(searchValues['experiments']);
  }
  const { metricKey } = match.params;
  return {
    runUuids,
    metricKey,
    experimentIds,
  };
};

export const MetricPage = connect(mapStateToProps)(MetricPageImpl);
