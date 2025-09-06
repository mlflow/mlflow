/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { connect } from 'react-redux';
import {
  getModelVersionApi,
  getRegisteredModelApi,
  updateModelVersionApi,
  deleteModelVersionApi,
  transitionModelVersionStageApi,
  getModelVersionArtifactApi,
  parseMlModelFile,
} from '../actions';
import { getRunApi } from '../../experiment-tracking/actions';
import { getModelVersion, getModelVersionSchemas } from '../reducers';
import { ModelVersionView } from './ModelVersionView';
import type { PendingModelVersionActivity } from '../constants';
import { ActivityTypes, MODEL_VERSION_STATUS_POLL_INTERVAL as POLL_INTERVAL } from '../constants';
import Utils from '../../common/utils/Utils';
import { getRunInfo, getRunTags } from '../../experiment-tracking/reducers/Reducers';
import RequestStateWrapper, { triggerError } from '../../common/components/RequestStateWrapper';
import { ErrorView } from '../../common/components/ErrorView';
import { Spinner } from '../../common/components/Spinner';
import { ModelRegistryRoutes } from '../routes';
import { getProtoField } from '../utils';
import { getUUID } from '../../common/utils/ActionUtils';
import { without } from 'lodash';
import { PageContainer } from '../../common/components/PageContainer';
import { withRouterNext } from '../../common/utils/withRouterNext';
import type { WithRouterNextProps } from '../../common/utils/withRouterNext';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import type { ModelEntity, RunInfoEntity } from '../../experiment-tracking/types';
import type { ReduxState } from '../../redux-types';
import { ErrorCodes } from '../../common/constants';
import { injectIntl } from 'react-intl';

type ModelVersionPageImplProps = WithRouterNextProps & {
  modelName: string;
  version: string;
  modelVersion?: any;
  runInfo?: any;
  runDisplayName?: string;
  modelEntity?: ModelEntity;
  getModelVersionApi: (...args: any[]) => any;
  getRegisteredModelApi: typeof getRegisteredModelApi;
  updateModelVersionApi: (...args: any[]) => any;
  transitionModelVersionStageApi: (...args: any[]) => any;
  deleteModelVersionApi: (...args: any[]) => any;
  getRunApi: (...args: any[]) => any;
  apis: any;
  getModelVersionArtifactApi: (...args: any[]) => any;
  parseMlModelFile: (...args: any[]) => any;
  schema?: any;
  activities?: Record<string, unknown>[];
  intl?: any;
};

type ModelVersionPageImplState = any;

export class ModelVersionPageImpl extends React.Component<ModelVersionPageImplProps, ModelVersionPageImplState> {
  listTransitionRequestId: any;
  pollIntervalId: any;

  initGetModelVersionDetailsRequestId = getUUID();
  getRunRequestId = getUUID();
  updateModelVersionRequestId = getUUID();
  transitionModelVersionStageRequestId = getUUID();
  getModelVersionDetailsRequestId = getUUID();
  initGetMlModelFileRequestId = getUUID();
  state = {
    criticalInitialRequestIds: [this.initGetModelVersionDetailsRequestId, this.initGetMlModelFileRequestId],
  };

  pollingRelatedRequestIds = [this.getModelVersionDetailsRequestId, this.getRunRequestId];

  hasPendingPollingRequest = () =>
    this.pollingRelatedRequestIds.every((requestId) => {
      const request = this.props.apis[requestId];
      return Boolean(request && request.active);
    });

  loadData = (isInitialLoading: any) => {
    const promises = [this.getModelVersionDetailAndRunInfo(isInitialLoading)];
    return Promise.all(promises);
  };

  pollData = () => {
    const { modelName, version, navigate } = this.props;
    if (!this.hasPendingPollingRequest() && Utils.isBrowserTabVisible()) {
      // @ts-expect-error TS(2554): Expected 1 arguments, but got 0.
      return this.loadData().catch((e) => {
        if (e.getErrorCode() === 'RESOURCE_DOES_NOT_EXIST') {
          Utils.logErrorAndNotifyUser(e);
          this.props.deleteModelVersionApi(modelName, version, undefined, true);
          navigate(ModelRegistryRoutes.getModelPageRoute(modelName));
        } else {
          // eslint-disable-next-line no-console -- TODO(FEINF-3587)
          console.error(e);
        }
      });
    }
    return Promise.resolve();
  };

  // We need to do this because currently the ModelVersionDetailed we got does not contain
  // experimentId. We need experimentId to construct a link to the source run. This workaround can
  // be removed after the availability of experimentId.
  getModelVersionDetailAndRunInfo(isInitialLoading: any) {
    const { modelName, version } = this.props;
    return this.props
      .getModelVersionApi(
        modelName,
        version,
        isInitialLoading === true ? this.initGetModelVersionDetailsRequestId : this.getModelVersionDetailsRequestId,
      )
      .then(({ value }: any) => {
        // Do not fetch run info if there is no run_id (e.g. model version created directly from a logged model)
        if (value && !value[getProtoField('model_version')].run_link && value[getProtoField('model_version')]?.run_id) {
          this.props.getRunApi(value[getProtoField('model_version')].run_id, this.getRunRequestId);
        }
      });
  }
  // We need this for getting mlModel artifact file,
  // this will be replaced with a single backend call in the future when supported
  getModelVersionMlModelFile() {
    const { modelName, version } = this.props;
    this.props
      .getModelVersionArtifactApi(modelName, version)
      .then((content: any) =>
        this.props.parseMlModelFile(modelName, version, content.value, this.initGetMlModelFileRequestId),
      )
      .catch(() => {
        // Failure of this call chain should not block the page. Here we remove
        // `initGetMlModelFileRequestId` from `criticalInitialRequestIds`
        // to unblock RequestStateWrapper from rendering its content
        this.setState((prevState: any) => ({
          criticalInitialRequestIds: without(prevState.criticalInitialRequestIds, this.initGetMlModelFileRequestId),
        }));
      });
  }

  // prettier-ignore
  handleStageTransitionDropdownSelect = (
    activity: PendingModelVersionActivity,
    archiveExistingVersions?: boolean,
  ) => {
    const { modelName, version } = this.props;
    const toStage = activity.to_stage;
    if (activity.type === ActivityTypes.APPLIED_TRANSITION) {
      this.props
        .transitionModelVersionStageApi(
          modelName,
          version.toString(),
          toStage,
          archiveExistingVersions,
          this.transitionModelVersionStageRequestId,
        )
        .then(this.loadData)
        .catch(Utils.logErrorAndNotifyUser);
    }
  };

  handleEditDescription = (description: any) => {
    const { modelName, version } = this.props;
    return (
      this.props
        .updateModelVersionApi(modelName, version, description, this.updateModelVersionRequestId)
        .then(this.loadData)
        // eslint-disable-next-line no-console -- TODO(FEINF-3587)
        .catch(console.error)
    );
  };

  componentDidMount() {
    // eslint-disable-next-line no-console -- TODO(FEINF-3587)
    this.loadData(true).catch(console.error);
    this.loadModelDataWithAliases();
    this.pollIntervalId = setInterval(this.pollData, POLL_INTERVAL);
    this.getModelVersionMlModelFile();
  }

  loadModelDataWithAliases = () => {
    this.props.getRegisteredModelApi(this.props.modelName);
  };

  // Make a new initial load if model version or name has changed
  componentDidUpdate(prevProps: ModelVersionPageImplProps) {
    if (this.props.version !== prevProps.version || this.props.modelName !== prevProps.modelName) {
      // eslint-disable-next-line no-console -- TODO(FEINF-3587)
      this.loadData(true).catch(console.error);
      this.getModelVersionMlModelFile();
    }
  }

  componentWillUnmount() {
    clearInterval(this.pollIntervalId);
  }

  render() {
    const { modelName, version, modelVersion, runInfo, runDisplayName, navigate, schema, modelEntity } = this.props;

    return (
      <PageContainer>
        <RequestStateWrapper
          requestIds={this.state.criticalInitialRequestIds}
          // eslint-disable-next-line no-trailing-spaces
        >
          {(loading: any, hasError: any, requests: any) => {
            if (hasError) {
              clearInterval(this.pollIntervalId);
              const resourceConflictError = Utils.getResourceConflictError(
                requests,
                this.state.criticalInitialRequestIds,
              );
              if (resourceConflictError) {
                return (
                  <ErrorView
                    statusCode={409}
                    subMessage={resourceConflictError.error.getMessageField()}
                    fallbackHomePageReactRoute={ModelRegistryRoutes.modelListPageRoute}
                  />
                );
              }
              if (Utils.shouldRender404(requests, this.state.criticalInitialRequestIds)) {
                return (
                  <ErrorView
                    statusCode={404}
                    subMessage={`Model ${modelName} v${version} does not exist`}
                    fallbackHomePageReactRoute={ModelRegistryRoutes.modelListPageRoute}
                  />
                );
              }
              // TODO(Zangr) Have a more generic boundary to handle all errors, not just 404.
              const permissionDeniedErrors = requests.filter((request: any) => {
                return (
                  this.state.criticalInitialRequestIds.includes(request.id) &&
                  request.error?.getErrorCode() === ErrorCodes.PERMISSION_DENIED
                );
              });
              if (permissionDeniedErrors && permissionDeniedErrors[0]) {
                return (
                  <ErrorView
                    statusCode={403}
                    subMessage={this.props.intl.formatMessage(
                      {
                        defaultMessage: 'Permission denied for {modelName} version {version}. Error: "{errorMsg}"',
                        description: 'Permission denied error message on model version detail page',
                      },
                      {
                        modelName: modelName,
                        version: version,
                        errorMsg: permissionDeniedErrors[0].error?.getMessageField(),
                      },
                    )}
                    fallbackHomePageReactRoute={ModelRegistryRoutes.modelListPageRoute}
                  />
                );
              }
              triggerError(requests);
            } else if (loading) {
              return <Spinner />;
            } else if (modelVersion) {
              // Null check to prevent NPE after delete operation
              return (
                <ModelVersionView
                  modelName={modelName}
                  modelVersion={modelVersion}
                  modelEntity={modelEntity}
                  runInfo={runInfo}
                  runDisplayName={runDisplayName}
                  handleEditDescription={this.handleEditDescription}
                  deleteModelVersionApi={this.props.deleteModelVersionApi}
                  navigate={navigate}
                  handleStageTransitionDropdownSelect={this.handleStageTransitionDropdownSelect}
                  schema={schema}
                  onAliasesModified={this.loadModelDataWithAliases}
                />
              );
            }
            return null;
          }}
        </RequestStateWrapper>
      </PageContainer>
    );
  }
}

const mapStateToProps = (state: ReduxState, ownProps: WithRouterNextProps<{ modelName: string; version: string }>) => {
  const modelName = decodeURIComponent(ownProps.params.modelName);
  const { version } = ownProps.params;
  const modelVersion = getModelVersion(state, modelName, version);
  const schema = getModelVersionSchemas(state, modelName, version);
  let runInfo: RunInfoEntity | null = null;
  if (modelVersion && !modelVersion.run_link) {
    runInfo = getRunInfo(modelVersion && modelVersion.run_id, state);
  }
  const tags = runInfo && getRunTags(runInfo.runUuid, state);
  const runDisplayName = tags && runInfo && Utils.getRunDisplayName(runInfo, runInfo.runUuid);
  const modelEntity = state.entities.modelByName[modelName];
  const { apis } = state;
  return {
    modelName,
    version,
    modelVersion,
    schema,
    runInfo,
    runDisplayName,
    apis,
    modelEntity,
  };
};

const mapDispatchToProps = {
  getModelVersionApi,
  getRegisteredModelApi,
  updateModelVersionApi,
  transitionModelVersionStageApi,
  getModelVersionArtifactApi,
  parseMlModelFile,
  deleteModelVersionApi,
  getRunApi,
};

const ModelVersionPageWithRouter = withRouterNext(
  // @ts-expect-error TS(2769): No overload matches this call.
  connect(mapStateToProps, mapDispatchToProps)(injectIntl(ModelVersionPageImpl)),
);

export const ModelVersionPage = withErrorBoundary(ErrorUtils.mlflowServices.MODEL_REGISTRY, ModelVersionPageWithRouter);

export default ModelVersionPage;
