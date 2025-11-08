/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { first, isEmpty, isUndefined } from 'lodash';
import React, { Component } from 'react';
import { FormattedMessage } from 'react-intl';
import type { WithRouterNextProps } from '../../common/utils/withRouterNext';
import { withRouterNext } from '../../common/utils/withRouterNext';
import { ArtifactView } from './ArtifactView';
import { Spinner } from '../../common/components/Spinner';
import { listArtifactsApi, listArtifactsLoggedModelApi } from '../actions';
import { searchModelVersionsApi } from '../../model-registry/actions';
import { connect } from 'react-redux';
import { getArtifactRootUri, getArtifacts } from '../reducers/Reducers';
import { MODEL_VERSION_STATUS_POLL_INTERVAL as POLL_INTERVAL } from '../../model-registry/constants';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import Utils from '../../common/utils/Utils';
import { getUUID } from '../../common/utils/ActionUtils';
import { getLoggedModelPathsFromTags } from '../../common/utils/TagUtils';
import { ArtifactViewBrowserSkeleton } from './artifact-view-components/ArtifactViewSkeleton';
import { DangerIcon, Empty } from '@databricks/design-system';
import { ArtifactViewErrorState } from './artifact-view-components/ArtifactViewErrorState';
import type { LoggedModelArtifactViewerProps } from './artifact-view-components/ArtifactViewComponents.types';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import type { UseGetRunQueryResponseOutputs } from './run-page/hooks/useGetRunQuery';
import type { ReduxState } from '../../redux-types';
import { asyncGetLoggedModel } from '../hooks/logged-models/useGetLoggedModelQuery';
import type { KeyValueEntity } from '../../common/types';

type ArtifactPageImplProps = {
  runUuid?: string;
  initialSelectedArtifactPath?: string;
  artifactRootUri?: string;
  apis: any;
  listArtifactsApi: (...args: any[]) => any;
  listArtifactsLoggedModelApi: typeof listArtifactsLoggedModelApi;
  searchModelVersionsApi: (...args: any[]) => any;
  runTags?: any;
  runOutputs?: UseGetRunQueryResponseOutputs;
  entityTags?: Partial<KeyValueEntity>[];

  /**
   * If true, the artifact browser will try to use all available height
   */
  useAutoHeight?: boolean;
} & LoggedModelArtifactViewerProps;

type ArtifactPageImplState = {
  errorThrown: boolean;
  activeNodeIsDirectory: boolean;
  fallbackEntityTags?: Partial<KeyValueEntity>[];
};

export class ArtifactPageImpl extends Component<ArtifactPageImplProps, ArtifactPageImplState> {
  pollIntervalId: any;

  getFailedtoListArtifactsMsg = () => {
    return (
      <span>
        <FormattedMessage
          // eslint-disable-next-line max-len
          defaultMessage="Unable to list artifacts stored under {artifactUri} for the current run. Please contact your tracking server administrator to notify them of this error, which can happen when the tracking server lacks permission to list artifacts under the current run's root artifact directory."
          // eslint-disable-next-line max-len
          description="Error message when the artifact is unable to load. This message is displayed in the open source ML flow only"
          values={{ artifactUri: this.props.artifactRootUri }}
        />
      </span>
    );
  };

  state: ArtifactPageImplState = { activeNodeIsDirectory: false, errorThrown: false };

  searchRequestId = getUUID();

  listArtifactRequestIds = [getUUID()].concat(
    this.props.initialSelectedArtifactPath
      ? this.props.initialSelectedArtifactPath.split('/').map((s) => getUUID())
      : [],
  );

  pollModelVersionsForCurrentRun = async () => {
    const { apis, runUuid, isLoggedModelsMode } = this.props;
    const { activeNodeIsDirectory } = this.state;
    const searchRequest = apis[this.searchRequestId];
    // Do not poll for run's model versions if we are in the logged models mode
    if (isLoggedModelsMode && !runUuid) {
      return;
    }
    if (activeNodeIsDirectory && !(searchRequest && searchRequest.active)) {
      try {
        // searchModelVersionsApi may be sync or async so we're not using <promise>.catch() syntax
        await this.props.searchModelVersionsApi({ run_id: runUuid }, this.searchRequestId);
      } catch (error) {
        // We're not reporting errors more than once when polling
        // in order to avoid flooding logs
        if (!this.state.errorThrown) {
          const errorString = error instanceof Error ? error.toString() : JSON.stringify(error);
          const errorMessage = `Error while fetching model version for run: ${errorString}`;
          Utils.logErrorAndNotifyUser(errorMessage);
          this.setState({ errorThrown: true });
        }
      }
    }
  };

  handleActiveNodeChange = (activeNodeIsDirectory: any) => {
    this.setState({ activeNodeIsDirectory });
  };

  pollArtifactsForCurrentRun = async () => {
    const { runUuid, loggedModelId, isFallbackToLoggedModelArtifacts } = this.props;

    const usingLoggedModels = this.props.isLoggedModelsMode;

    let fallbackEntityTags: Partial<KeyValueEntity>[] | undefined = undefined;

    // In the logged models mode, fetch artifacts for the model instead of the run
    if (usingLoggedModels && loggedModelId) {
      // If falling back from run artifacts to logged model artifacts, fetch the logged model's tags
      // in order to correctly resolve artifact storage path.
      if (isFallbackToLoggedModelArtifacts) {
        const loggedModelData = await asyncGetLoggedModel(loggedModelId, true);
        fallbackEntityTags = loggedModelData?.model?.info?.tags;
        this.setState({
          fallbackEntityTags,
        });
      }
      await this.props.listArtifactsLoggedModelApi(
        this.props.loggedModelId,
        undefined,
        this.props.experimentId,
        this.listArtifactRequestIds[0],
        fallbackEntityTags ?? this.props.entityTags,
      );
    } else {
      await this.props.listArtifactsApi(runUuid, undefined, this.listArtifactRequestIds[0]);
    }
    if (this.props.initialSelectedArtifactPath) {
      const parts = this.props.initialSelectedArtifactPath.split('/');
      let pathSoFar = '';
      for (let i = 0; i < parts.length; i++) {
        pathSoFar += parts[i];
        // ML-12477: ListArtifacts API requests need to be sent and fulfilled for parent
        // directories before nested child directories, as our Reducers assume that parent
        // directories are listed before their children to construct the correct artifact tree.
        // Index i + 1 because listArtifactRequestIds[0] would have been used up by
        // root-level artifact API call above.

        // In the logged models mode, fetch artifacts for the model instead of the run
        if (usingLoggedModels && loggedModelId) {
          await this.props.listArtifactsLoggedModelApi(
            this.props.loggedModelId,
            pathSoFar,
            this.props.experimentId,
            this.listArtifactRequestIds[i + 1],
            fallbackEntityTags ?? this.props.entityTags,
          );
        } else {
          await this.props.listArtifactsApi(runUuid, pathSoFar, this.listArtifactRequestIds[i + 1]);
        }
        pathSoFar += '/';
      }
    }
  };

  componentDidMount() {
    if (this.props.runUuid && this.isWorkspaceModelRegistryEnabled) {
      this.pollModelVersionsForCurrentRun();
      this.pollIntervalId = setInterval(this.pollModelVersionsForCurrentRun, POLL_INTERVAL);
    }
    this.pollArtifactsForCurrentRun();
  }

  componentDidUpdate(prevProps: ArtifactPageImplProps) {
    if (prevProps.runUuid !== this.props.runUuid) {
      this.setState({
        errorThrown: false,
      });
    }
    // If the component eventually falls back to logged model artifacts, poll artifacts for the current run
    if (!prevProps.isFallbackToLoggedModelArtifacts && this.props.isFallbackToLoggedModelArtifacts) {
      this.pollArtifactsForCurrentRun();
    }
  }

  get isWorkspaceModelRegistryEnabled() {
    return Utils.isModelRegistryEnabled();
  }

  componentWillUnmount() {
    if (this.isWorkspaceModelRegistryEnabled && !isUndefined(this.pollIntervalId)) {
      clearInterval(this.pollIntervalId);
    }
  }

  renderErrorCondition = (shouldRenderError: any) => {
    return shouldRenderError;
  };

  renderArtifactView = (isLoading: any, shouldRenderError: any, requests: any) => {
    if (isLoading && !shouldRenderError) {
      return <ArtifactViewBrowserSkeleton />;
    }
    if (this.renderErrorCondition(shouldRenderError)) {
      const failedReq = requests[0];
      if (failedReq && failedReq.error) {
        // eslint-disable-next-line no-console -- TODO(FEINF-3587)
        console.error(failedReq.error);
      }
      const errorDescription = (() => {
        const error = failedReq?.error;
        if (error instanceof ErrorWrapper) {
          return error.getMessageField();
        }

        return this.getFailedtoListArtifactsMsg();
      })();
      return (
        <ArtifactViewErrorState
          css={{ flex: this.props.useAutoHeight ? 1 : 'unset', height: this.props.useAutoHeight ? 'auto' : undefined }}
          data-testid="artifact-view-error"
          description={errorDescription}
        />
      );
    }
    return (
      <ArtifactView
        {...this.props}
        entityTags={this.state.fallbackEntityTags ?? this.props.entityTags}
        handleActiveNodeChange={this.handleActiveNodeChange}
        useAutoHeight={this.props.useAutoHeight}
      />
    );
  };

  render() {
    return (
      <RequestStateWrapper
        requestIds={this.listArtifactRequestIds}
        // eslint-disable-next-line no-trailing-spaces
      >
        {this.renderArtifactView}
      </RequestStateWrapper>
    );
  }
}

type ArtifactPageOwnProps = Omit<
  ArtifactPageImplProps,
  | 'apis'
  | 'initialSelectedArtifactPath'
  | 'listArtifactsApi'
  | 'listArtifactsLoggedModelApi'
  | 'searchModelVersionsApi'
  /* prettier-ignore */
>;

const validVolumesPrefix = ['/Volumes/', 'dbfs:/Volumes/'];

// Internal utility function to determine if the component should fallback to logged model artifacts
// if there are no run artifacts available
const shouldFallbackToLoggedModelArtifacts = (
  state: ReduxState,
  ownProps: ArtifactPageOwnProps & WithRouterNextProps,
): {
  isFallbackToLoggedModelArtifacts: boolean;
  fallbackLoggedModelId?: string;
} => {
  const isVolumePath = validVolumesPrefix.some((prefix) => ownProps.artifactRootUri?.startsWith(prefix));

  // Execute only if feature is enabled and we are currently fetching >run< artifacts.
  // Also, do not fallback to logged model artifacts for Volume-based artifact paths.
  if (!ownProps.isLoggedModelsMode) {
    // Let's check if the root artifact is already present (i.e. run artifacts are fetched)
    const rootArtifact = getArtifacts(ownProps.runUuid, state);
    const isRunArtifactsEmpty = rootArtifact && !rootArtifact.fileInfo && isEmpty(rootArtifact.children);

    // Check if we have a logged model id to fallback to
    const loggedModelId = first(ownProps.runOutputs?.modelOutputs)?.modelId;

    // If true, return relevant information to the component
    if (isRunArtifactsEmpty && loggedModelId) {
      return {
        isFallbackToLoggedModelArtifacts: true,
        fallbackLoggedModelId: loggedModelId,
      };
    }
  }
  // Otherwise, do not fallback to logged model artifacts
  return {
    isFallbackToLoggedModelArtifacts: false,
  };
};

const mapStateToProps = (state: any, ownProps: ArtifactPageOwnProps & WithRouterNextProps) => {
  const { runUuid, location, runOutputs } = ownProps;
  const currentPathname = location?.pathname || '';

  const initialSelectedArtifactPathMatch = currentPathname.match(/\/(?:artifactPath|artifacts)\/(.+)/);

  // Check the conditions to fallback to logged model artifacts
  const { isFallbackToLoggedModelArtifacts, fallbackLoggedModelId } = shouldFallbackToLoggedModelArtifacts(
    state,
    ownProps,
  );

  // The dot ("*") parameter behavior is not stable between implementations
  // so we'll extract the catch-all after /artifactPath, e.g.
  // `/experiments/123/runs/321/artifactPath/models/requirements.txt`
  // is getting transformed into
  // `models/requirements.txt`
  const initialSelectedArtifactPath = initialSelectedArtifactPathMatch?.[1] || undefined;

  const { apis } = state;
  const artifactRootUri = ownProps.artifactRootUri ?? getArtifactRootUri(runUuid, state);

  // Autoselect most recently created logged model
  let selectedPath = initialSelectedArtifactPath;
  if (!selectedPath) {
    const loggedModelPaths = getLoggedModelPathsFromTags(ownProps.runTags ?? {});
    if (loggedModelPaths.length > 0) {
      selectedPath = first(loggedModelPaths);
    }
  }
  return {
    artifactRootUri,
    apis,
    initialSelectedArtifactPath: selectedPath,

    // Use the run outputs if available, otherwise fallback to the run outputs from the Redux store
    isLoggedModelsMode: isFallbackToLoggedModelArtifacts ? true : ownProps.isLoggedModelsMode,
    loggedModelId: isFallbackToLoggedModelArtifacts ? fallbackLoggedModelId : ownProps.loggedModelId,
    isFallbackToLoggedModelArtifacts,
  };
};

const mapDispatchToProps = {
  listArtifactsApi,
  listArtifactsLoggedModelApi,
  searchModelVersionsApi,
};

export const ConnectedArtifactPage = connect(mapStateToProps, mapDispatchToProps)(ArtifactPageImpl);

export default withRouterNext(ConnectedArtifactPage);
