/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { connect } from 'react-redux';
import qs from 'qs';
import { getExperimentApi, getRunApi } from '../actions';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import NotFoundPage from './NotFoundPage';
import { MetricView } from './MetricView';
import { getUUID } from '../../common/utils/ActionUtils';
import { PageContainer } from '../../common/components/PageContainer';
import { withRouterNext } from '../../common/utils/withRouterNext';
import type { WithRouterNextProps } from '../../common/utils/withRouterNext';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import Utils from '../../common/utils/Utils';
import { injectIntl, type IntlShape } from 'react-intl';

type MetricPageImplProps = {
  runUuids: string[];
  metricKey: string;
  experimentIds?: string[];
  dispatch: (...args: any[]) => any;
  loadError?: unknown;
  intl: IntlShape;
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
    if (this.props.loadError instanceof Error) {
      const message = this.props.intl.formatMessage({
        defaultMessage: 'Error during metric page load: invalid URL',
        description: 'Error message when loading metric page fails',
      });
      Utils.displayGlobalErrorNotification(message);
    }
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

const mapStateToProps = (state: any, ownProps: WithRouterNextProps<{ metricKey: string }>) => {
  const { location } = ownProps;
  const searchValues = qs.parse(location.search);
  try {
    // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
    const runUuids = JSON.parse(searchValues['?runs']);
    // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
    const metricKey = JSON.parse(searchValues['metric']);
    let experimentIds = null;
    if (searchValues.hasOwnProperty('experiments')) {
      // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
      experimentIds = JSON.parse(searchValues['experiments']);
    }

    return {
      runUuids,
      metricKey,
      experimentIds,
    };
  } catch (e) {
    return {
      runUuids: [],
      metricKey: '',
      experimentIds: [],
      loadError: e,
    };
  }
};

const MetricPageWithRouter = withRouterNext(connect(mapStateToProps)(injectIntl(MetricPageImpl)));

export const MetricPage = withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, MetricPageWithRouter);
export default MetricPage;
