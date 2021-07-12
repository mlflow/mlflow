import React from 'react';
import { connect } from 'react-redux';
import {
  getModelVersionApi,
  updateModelVersionApi,
  deleteModelVersionApi,
  transitionModelVersionStageApi,
  getModelVersionArtifactApi,
  parseMlModelFile,
} from '../actions';
import { getRunApi } from '../../experiment-tracking/actions';
import PropTypes from 'prop-types';
import { getModelVersion, getModelVersionSchemas } from '../reducers';
import { ModelVersionView } from './ModelVersionView';
import { ActivityTypes, MODEL_VERSION_STATUS_POLL_INTERVAL as POLL_INTERVAL } from '../constants';
import Utils from '../../common/utils/Utils';
import { getRunInfo, getRunTags } from '../../experiment-tracking/reducers/Reducers';
import RequestStateWrapper, { triggerError } from '../../common/components/RequestStateWrapper';
import { ErrorView } from '../../common/components/ErrorView';
import { Spinner } from '../../common/components/Spinner';
import { getModelPageRoute, modelListPageRoute } from '../routes';
import { getProtoField } from '../utils';
import { getUUID } from '../../common/utils/ActionUtils';
import _ from 'lodash';

export class ModelVersionPageImpl extends React.Component {
  static propTypes = {
    // own props
    history: PropTypes.object.isRequired,
    match: PropTypes.object.isRequired,
    // connected props
    modelName: PropTypes.string.isRequired,
    version: PropTypes.string.isRequired,
    modelVersion: PropTypes.object,
    runInfo: PropTypes.object,
    runDisplayName: PropTypes.string,
    getModelVersionApi: PropTypes.func.isRequired,
    updateModelVersionApi: PropTypes.func.isRequired,
    transitionModelVersionStageApi: PropTypes.func.isRequired,
    deleteModelVersionApi: PropTypes.func.isRequired,
    getRunApi: PropTypes.func.isRequired,
    apis: PropTypes.object.isRequired,
    getModelVersionArtifactApi: PropTypes.func.isRequired,
    parseMlModelFile: PropTypes.func.isRequired,
    schema: PropTypes.object,
  };

  initGetModelVersionDetailsRequestId = getUUID();
  getRunRequestId = getUUID();
  updateModelVersionRequestId = getUUID();
  transitionModelVersionStageRequestId = getUUID();
  getModelVersionDetailsRequestId = getUUID();
  initGetMlModelFileRequestId = getUUID();
  state = {
    criticalInitialRequestIds: [
      this.initGetModelVersionDetailsRequestId,
      this.initGetMlModelFileRequestId,
    ],
  };

  pollingRelatedRequestIds = [this.getModelVersionDetailsRequestId, this.getRunRequestId];

  hasPendingPollingRequest = () =>
    this.pollingRelatedRequestIds.every((requestId) => {
      const request = this.props.apis[requestId];
      return Boolean(request && request.active);
    });

  loadData = (isInitialLoading) => {
    const promises = [this.getModelVersionDetailAndRunInfo(isInitialLoading)];
    return Promise.all([promises]);
  };

  pollData = () => {
    const { modelName, version, history } = this.props;
    if (!this.hasPendingPollingRequest() && Utils.isBrowserTabVisible()) {
      return this.loadData().catch((e) => {
        if (e.getErrorCode() === 'RESOURCE_DOES_NOT_EXIST') {
          Utils.logErrorAndNotifyUser(e);
          this.props.deleteModelVersionApi(modelName, version, undefined, true);
          history.push(getModelPageRoute(modelName));
        } else {
          console.error(e);
        }
      });
    }
    return Promise.resolve();
  };

  // We need to do this because currently the ModelVersionDetailed we got does not contain
  // experimentId. We need experimentId to construct a link to the source run. This workaround can
  // be removed after the availability of experimentId.
  getModelVersionDetailAndRunInfo(isInitialLoading) {
    const { modelName, version } = this.props;
    return this.props
      .getModelVersionApi(
        modelName,
        version,
        isInitialLoading === true
          ? this.initGetModelVersionDetailsRequestId
          : this.getModelVersionDetailsRequestId,
      )
      .then(({ value }) => {
        if (value && !value[getProtoField('model_version')].run_link) {
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
      .then((content) =>
        this.props.parseMlModelFile(
          modelName,
          version,
          content.value,
          this.initGetMlModelFileRequestId,
        ),
      )
      .catch(() => {
        // Failure of this call chain should not block the page. Here we remove
        // `initGetMlModelFileRequestId` from `criticalInitialRequestIds`
        // to unblock RequestStateWrapper from rendering its content
        this.setState((prevState) => ({
          criticalInitialRequestIds: _.without(
            prevState.criticalInitialRequestIds,
            this.initGetMlModelFileRequestId,
          ),
        }));
      });
  }

  handleStageTransitionDropdownSelect = (activity, archiveExistingVersions) => {
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

  handleEditDescription = (description) => {
    const { modelName, version } = this.props;
    return this.props
      .updateModelVersionApi(modelName, version, description, this.updateModelVersionRequestId)
      .then(this.loadData)
      .catch(console.error);
  };

  componentDidMount() {
    this.loadData(true).catch(console.error);
    this.pollIntervalId = setInterval(this.pollData, POLL_INTERVAL);
    this.getModelVersionMlModelFile();
  }

  componentWillUnmount() {
    clearTimeout(this.pollIntervalId);
  }

  render() {
    const {
      modelName,
      version,
      modelVersion,
      runInfo,
      runDisplayName,
      history,
      schema,
    } = this.props;

    return (
      <div className='App-content'>
        <RequestStateWrapper requestIds={this.state.criticalInitialRequestIds}>
          {(loading, hasError, requests) => {
            if (hasError) {
              clearInterval(this.pollIntervalId);
              if (Utils.shouldRender404(requests, this.state.criticalInitialRequestIds)) {
                return (
                  <ErrorView
                    statusCode={404}
                    subMessage={`Model ${modelName} v${version} does not exist`}
                    fallbackHomePageReactRoute={modelListPageRoute}
                  />
                );
              }
              // TODO(Zangr) Have a more generic boundary to handle all errors, not just 404.
              triggerError(requests);
            } else if (loading) {
              return <Spinner />;
            } else if (modelVersion) {
              // Null check to prevent NPE after delete operation
              return (
                <ModelVersionView
                  modelName={modelName}
                  modelVersion={modelVersion}
                  runInfo={runInfo}
                  runDisplayName={runDisplayName}
                  handleEditDescription={this.handleEditDescription}
                  deleteModelVersionApi={this.props.deleteModelVersionApi}
                  history={history}
                  handleStageTransitionDropdownSelect={this.handleStageTransitionDropdownSelect}
                  schema={schema}
                />
              );
            }
            return null;
          }}
        </RequestStateWrapper>
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const modelName = decodeURIComponent(ownProps.match.params.modelName);
  const { version } = ownProps.match.params;
  const modelVersion = getModelVersion(state, modelName, version);
  const schema = getModelVersionSchemas(state, modelName, version);
  let runInfo = null;
  if (modelVersion && !modelVersion.run_link) {
    runInfo = getRunInfo(modelVersion && modelVersion.run_id, state);
  }
  const tags = runInfo && getRunTags(runInfo.getRunUuid(), state);
  const runDisplayName = tags && Utils.getRunDisplayName(tags, runInfo.getRunUuid());
  const { apis } = state;
  return {
    modelName,
    version,
    modelVersion,
    schema,
    runInfo,
    runDisplayName,
    apis,
  };
};

const mapDispatchToProps = {
  getModelVersionApi,
  updateModelVersionApi,
  transitionModelVersionStageApi,
  getModelVersionArtifactApi,
  parseMlModelFile,
  deleteModelVersionApi,
  getRunApi,
};

export const ModelVersionPage = connect(mapStateToProps, mapDispatchToProps)(ModelVersionPageImpl);
