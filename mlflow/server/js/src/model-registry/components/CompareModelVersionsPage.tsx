/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import qs from 'qs';
import { connect } from 'react-redux';
import { getRunApi } from '../../experiment-tracking/actions';
import { getUUID } from '../../common/utils/ActionUtils';
import { getRegisteredModelApi, getModelVersionApi, getModelVersionArtifactApi, parseMlModelFile } from '../actions';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import { CompareModelVersionsView } from './CompareModelVersionsView';
import { without } from 'lodash';
import { PageContainer } from '../../common/components/PageContainer';
import { withRouterNext } from '../../common/utils/withRouterNext';
import type { WithRouterNextProps } from '../../common/utils/withRouterNext';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';

type CompareModelVersionsPageImplProps = {
  modelName: string;
  versionsToRuns: any;
  getRunApi: (...args: any[]) => any;
  getRegisteredModelApi: (...args: any[]) => any;
  getModelVersionApi: (...args: any[]) => any;
  getModelVersionArtifactApi: (...args: any[]) => any;
  parseMlModelFile: (...args: any[]) => any;
};

type CompareModelVersionsPageImplState = any;

// TODO: Write integration tests for this component
export class CompareModelVersionsPageImpl extends Component<
  CompareModelVersionsPageImplProps,
  CompareModelVersionsPageImplState
> {
  registeredModelRequestId = getUUID();
  versionRequestId = getUUID();
  runRequestId = getUUID();
  getMlModelFileRequestId = getUUID();

  state = {
    requestIds: [
      // requests that must be fulfilled before rendering
      this.registeredModelRequestId,
      this.runRequestId,
      this.versionRequestId,
      this.getMlModelFileRequestId,
    ],
    requestIdsWith404ErrorsToIgnore: [this.runRequestId, this.getMlModelFileRequestId],
  };

  removeRunRequestId() {
    this.setState((prevState: any) => ({
      requestIds: without(prevState.requestIds, this.runRequestId),
    }));
  }

  componentDidMount() {
    this.props.getRegisteredModelApi(this.props.modelName, this.registeredModelRequestId);
    for (const modelVersion in this.props.versionsToRuns) {
      if ({}.hasOwnProperty.call(this.props.versionsToRuns, modelVersion)) {
        const runID = this.props.versionsToRuns[modelVersion];
        if (runID) {
          this.props.getRunApi(runID, this.runRequestId).catch(() => {
            // Failure of this call should not block the page. Here we remove
            // `runRequestId` from `requestIds` to unblock RequestStateWrapper
            // from rendering its content
            this.removeRunRequestId();
          });
        } else {
          this.removeRunRequestId();
        }
        const { modelName } = this.props;
        this.props.getModelVersionApi(modelName, modelVersion, this.versionRequestId);
        this.props
          .getModelVersionArtifactApi(modelName, modelVersion)
          .then((content: any) =>
            this.props.parseMlModelFile(modelName, modelVersion, content.value, this.getMlModelFileRequestId),
          )
          .catch(() => {
            // Failure of this call chain should not block the page. Here we remove
            // `getMlModelFileRequestId` from `requestIds` to unblock RequestStateWrapper
            // from rendering its content
            this.setState((prevState: any) => ({
              requestIds: without(prevState.requestIds, this.getMlModelFileRequestId),
            }));
          });
      }
    }
  }

  render() {
    return (
      <PageContainer>
        <RequestStateWrapper
          requestIds={this.state.requestIds}
          requestIdsWith404sToIgnore={this.state.requestIdsWith404ErrorsToIgnore}
        >
          <CompareModelVersionsView modelName={this.props.modelName} versionsToRuns={this.props.versionsToRuns} />
        </RequestStateWrapper>
      </PageContainer>
    );
  }
}

const mapStateToProps = (state: any, ownProps: WithRouterNextProps) => {
  const { location } = ownProps;
  const searchValues = qs.parse(location.search);
  // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
  const modelName = decodeURIComponent(JSON.parse(searchValues['?name']));
  // @ts-expect-error TS(2345): Argument of type 'string | string[] | ParsedQs | P... Remove this comment to see the full error message
  const versionsToRuns = JSON.parse(searchValues['runs']);
  return { modelName, versionsToRuns };
};

const mapDispatchToProps = {
  getRunApi,
  getRegisteredModelApi,
  getModelVersionApi,
  getModelVersionArtifactApi,
  parseMlModelFile,
};

const CompareModelVersionsPageWithRouter = withRouterNext(
  connect(mapStateToProps, mapDispatchToProps)(CompareModelVersionsPageImpl),
);

export const CompareModelVersionsPage = withErrorBoundary(
  ErrorUtils.mlflowServices.MODEL_REGISTRY,
  CompareModelVersionsPageWithRouter,
);

export default CompareModelVersionsPage;
