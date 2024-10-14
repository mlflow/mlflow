/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import qs from 'qs';
import { connect } from 'react-redux';
import { getRunApi, getExperimentApi } from '../actions';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import CompareRunView from './CompareRunView';
import { getUUID } from '../../common/utils/ActionUtils';
import { PageContainer } from '../../common/components/PageContainer';
import { withRouterNext } from '../../common/utils/withRouterNext';
import type { WithRouterNextProps } from '../../common/utils/withRouterNext';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';

type CompareRunPageProps = {
  experimentIds: string[];
  runUuids: string[];
  dispatch: (...args: any[]) => any;
};

class CompareRunPageImpl extends Component<CompareRunPageProps> {
  requestIds: any;

  constructor(props: CompareRunPageProps) {
    super(props);
    this.requestIds = [];
  }

  fetchExperiments() {
    return this.props.experimentIds.map((experimentId) => {
      const experimentRequestId = getUUID();
      this.props.dispatch(getExperimentApi(experimentId, experimentRequestId));
      return experimentRequestId;
    });
  }

  componentDidMount() {
    this.requestIds.push(...this.fetchExperiments());
    this.props.runUuids.forEach((runUuid) => {
      const requestId = getUUID();
      this.requestIds.push(requestId);
      this.props.dispatch(getRunApi(runUuid, requestId));
    });
  }

  render() {
    return (
      <PageContainer>
        <RequestStateWrapper
          requestIds={this.requestIds}
          // eslint-disable-next-line no-trailing-spaces
        >
          <CompareRunView runUuids={this.props.runUuids} experimentIds={this.props.experimentIds} />
        </RequestStateWrapper>
      </PageContainer>
    );
  }
}

/**
 * When integrated via IFrame in Kubeflow it re-encodes the URI (sometimes multiple times), leading to an unparsable JSON.
 * This function decodes the URI until it is parsable.
 */
const decodeURI = (uri: string): string => {
  const decodedURI = decodeURIComponent(uri);
  if (uri !== decodedURI) {
    return decodeURI(decodedURI);
  }
  return decodedURI;
};

const mapStateToProps = (state: any, ownProps: WithRouterNextProps) => {
  const { location } = ownProps;
  const locationSearchDecoded = decodeURI(location.search);
  const searchValues = qs.parse(locationSearchDecoded);
  // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
  const runUuids = JSON.parse(searchValues['?runs']);
  // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
  const experimentIds = JSON.parse(searchValues['experiments']);
  return { experimentIds, runUuids };
};

const CompareRunPage = withRouterNext(connect(mapStateToProps)(CompareRunPageImpl));

export default withErrorBoundary(ErrorUtils.mlflowServices.RUN_TRACKING, CompareRunPage);
