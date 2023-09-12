/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import { getExperimentApi, getRunApi, setTagApi } from '../actions';
import { searchModelVersionsApi } from '../../model-registry/actions';
import { connect } from 'react-redux';
import { RunView } from './RunView';
import Routes from '../routes';
import Utils from '../../common/utils/Utils';
import { ErrorCodes } from '../../common/constants';
import { RunNotFoundView } from './RunNotFoundView';
import { getUUID } from '../../common/utils/ActionUtils';
import { Spinner } from '../../common/components/Spinner';
import { PageContainer } from '../../common/components/PageContainer';
import { withRouterNext } from '../../common/utils/withRouterNext';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';

type RunPageImplProps = {
  runUuid: string;
  experimentId: string;
  getRunApi: (...args: any[]) => any;
  getExperimentApi: (...args: any[]) => any;
  searchModelVersionsApi: (...args: any[]) => any;
  setTagApi: (...args: any[]) => any;
};

export class RunPageImpl extends Component<RunPageImplProps> {
  getRunRequestId = getUUID();

  getExperimentRequestId = getUUID();

  searchModelVersionsRequestId = getUUID();
  setTagRequestId = getUUID();

  componentDidMount() {
    const { experimentId, runUuid } = this.props;
    this.props.getRunApi(runUuid, this.getRunRequestId);
    this.props.getExperimentApi(experimentId, this.getExperimentRequestId);
    if (runUuid) {
      this.props.searchModelVersionsApi({ run_id: runUuid }, this.searchModelVersionsRequestId);
    }
  }

  handleSetRunTag = (tagName: any, value: any) => {
    const { runUuid } = this.props;
    return this.props
      .setTagApi(runUuid, tagName, value, this.setTagRequestId)
      .then(() => getRunApi(runUuid, this.getRunRequestId));
  };

  renderRunView = (isLoading: any, shouldRenderError: any, requests: any) => {
    if (isLoading) {
      return <Spinner />;
    } else if (shouldRenderError) {
      const getRunRequest = Utils.getRequestWithId(requests, this.getRunRequestId);
      if (getRunRequest.error.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST) {
        return <RunNotFoundView runId={this.props.runUuid} />;
      }
      return null;
    }
    return (
      <RunView
        runUuid={this.props.runUuid}
        getMetricPagePath={(key: any) =>
          Routes.getMetricPageRoute([this.props.runUuid], key, [this.props.experimentId])
        }
        experimentId={this.props.experimentId}
        handleSetRunTag={this.handleSetRunTag}
      />
    );
  };

  render() {
    const requestIds = [this.getRunRequestId, this.getExperimentRequestId];
    return (
      <PageContainer>
        <RequestStateWrapper
          requestIds={requestIds}
          // eslint-disable-next-line no-trailing-spaces
        >
          {this.renderRunView}
        </RequestStateWrapper>
      </PageContainer>
    );
  }
}

const mapStateToProps = (state: any, ownProps: any) => {
  const { params } = ownProps;
  const { runUuid, experimentId } = params;
  return {
    runUuid,
    experimentId,
    // so that we re-render the component when the route changes
    key: runUuid + experimentId,
  };
};

const mapDispatchToProps = {
  getRunApi,
  getExperimentApi,
  searchModelVersionsApi,
  setTagApi,
};

const RunPageWithRouter = withRouterNext(connect(mapStateToProps, mapDispatchToProps)(RunPageImpl));

export const RunPage = withErrorBoundary(ErrorUtils.mlflowServices.RUN_TRACKING, RunPageWithRouter);
