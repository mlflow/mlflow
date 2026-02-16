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
import type { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { DangerIcon, Empty, Spinner } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import Utils from '../../common/utils/Utils';
import { FallbackProps } from 'react-error-boundary';

type CompareRunPageProps = {
  experimentIds: string[];
  runUuids: string[];
  urlDecodeError?: boolean;
  dispatch: (...args: any[]) => any;
};

class CompareRunPageImpl extends Component<CompareRunPageProps> {
  requestIds: any;

  constructor(props: CompareRunPageProps) {
    super(props);
    this.requestIds = [];
  }

  state: {
    requestError?: Error | ErrorWrapper;
  } = {
    requestError: undefined,
  };

  fetchExperiments() {
    return this.props.experimentIds.map((experimentId) => {
      const experimentRequestId = getUUID();
      this.props
        .dispatch(getExperimentApi(experimentId, experimentRequestId))
        .catch((requestError: Error | ErrorWrapper) => this.setState({ requestError }));
      return experimentRequestId;
    });
  }

  componentDidMount() {
    this.requestIds.push(...this.fetchExperiments());
    this.props.runUuids.forEach((runUuid) => {
      const requestId = getUUID();
      this.requestIds.push(requestId);

      this.props.dispatch(getRunApi(runUuid, requestId)).catch((requestError: Error | ErrorWrapper) => {
        this.setState({ requestError });
      });
    });
  }

  render() {
    // If the error is set, throw it to be caught by the error boundary
    if (this.state.requestError) {
      const { requestError } = this.state;
      const errorToThrow = requestError instanceof Error ? requestError : new Error(requestError.getMessageField?.());
      throw errorToThrow;
    }
    return (
      <PageContainer>
        <RequestStateWrapper
          // We suppress throwing error by RequestStateWrapper since we handle it using component and error boundary
          suppressErrorThrow
          requestIds={this.requestIds}
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
  try {
    const { location } = ownProps;
    const locationSearchDecoded = decodeURI(location.search);
    const searchValues = qs.parse(locationSearchDecoded);
    // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
    const runUuids = JSON.parse(searchValues['?runs']);
    // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
    const experimentIds = JSON.parse(searchValues['experiments']);
    return { experimentIds, runUuids };
  } catch (e) {
    if (e instanceof SyntaxError) {
      throw new SyntaxError(`Error while parsing URL: ${e.message}`);
    }

    throw e;
  }
};

const CompareRunPageErrorFallback = ({ error }: { error: Error }) => (
  <div css={{ height: '100%', alignItems: 'center', justifyContent: 'center', display: 'flex' }}>
    <Empty
      title={
        <FormattedMessage
          defaultMessage="Error while loading compare runs page"
          description="Title of the error state on the run compare page"
        />
      }
      description={error.message}
      image={<DangerIcon />}
    />
  </div>
);

const CompareRunPage = withRouterNext(connect(mapStateToProps)(CompareRunPageImpl));

export default withErrorBoundary(
  ErrorUtils.mlflowServices.RUN_TRACKING,
  CompareRunPage,
  undefined,
  CompareRunPageErrorFallback,
);
