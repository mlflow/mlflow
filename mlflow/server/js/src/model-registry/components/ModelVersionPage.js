import React from 'react';
import { connect } from 'react-redux';
import {
  getModelVersionApi,
  updateModelVersionApi,
  deleteModelVersionApi,
} from '../actions';
import { getRunApi, getUUID } from '../../Actions';
import PropTypes from 'prop-types';
import { getModelVersion } from '../reducers';
import { ModelVersionView } from './ModelVersionView';
import { ActivityTypes, MODEL_VERSION_STATUS_POLL_INTERVAL as POLL_INTERVAL } from '../constants';
import Utils from '../../utils/Utils';
import { getRunInfo, getRunTags } from '../../reducers/Reducers';
import RequestStateWrapper, { triggerError } from '../../components/RequestStateWrapper';
import { shouldRender404 } from '../../common/utils';
import { Error404View } from '../../common/components/Error404View';
import { Spinner } from '../../components/Spinner';
import { modelListPageRoute } from '../routes';
import { ComponentOverrides } from '../overrides/component-overrides';

class ModelVersionPage extends React.Component {
  static propTypes = {
    modelName: PropTypes.string.isRequired,
    version: PropTypes.number.isRequired,
    modelVersion: PropTypes.object,
    runInfo: PropTypes.object,
    runDisplayName: PropTypes.string,
    getModelVersionApi: PropTypes.func.isRequired,
    updateModelVersionApi: PropTypes.func.isRequired,
    deleteModelVersionApi: PropTypes.func.isRequired,
    getRunApi: PropTypes.func.isRequired,
    history: PropTypes.object.isRequired,
    apis: PropTypes.object.isRequired,
  };

  initGetModelVersionDetailsRequestId = getUUID();
  getRunRequestId = getUUID();
  updateModelVersionRequestId = getUUID();
  getModelVersionDetailsRequestId = getUUID();

  criticalInitialRequestIds = [this.initGetModelVersionDetailsRequestId];

  loadData = (isInitialLoading) => {
    this.getModelVersionDetailAndRunInfo(isInitialLoading).catch(console.error);
  };

  // We need to do this because currently the ModelVersion we got does not contain
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
        if (value) {
          this.props.getRunApi(value.model_version.run_id, this.getRunRequestId);
        }
      });
  }

  handleStageTransitionDropdownSelect = (activity) => {
    const { modelName, version } = this.props;
    const toStage = activity.model_registry_data.to_stage;
    if (activity.type === ActivityTypes.APPLIED_TRANSITION) {
      this.props
        .updateModelVersionApi(
          modelName,
          version,
          toStage,
          undefined,
          this.updateModelVersionRequestId,
        )
        .then(this.loadData);
    }
  };

  handleEditDescription = (description) => {
    const { modelName, version } = this.props;
    return this.props
      .updateModelVersionApi(
        modelName,
        version,
        undefined,
        description,
        undefined,
        this.updateModelVersionRequestId,
      )
      .then(this.loadData);
  };

  pollModelVersionDetails = () => {
    const { modelName, version, apis } = this.props;
    const pollRequest = apis[this.getModelVersionDetailsRequestId];
    if (!(pollRequest && pollRequest.active)) {
      this.props
        .getModelVersionApi(modelName, version, this.getModelVersionDetailsRequestId)
        .catch(console.error);
    }
  };

  componentDidMount() {
    this.loadData(true);
    this.pollIntervalId = setInterval(this.pollModelVersionDetails, POLL_INTERVAL);
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
    } = this.props;

    return (
      <div className='App-content'>
        <RequestStateWrapper requestIds={this.criticalInitialRequestIds}>
          {(loading, hasError, requests) => {
            if (hasError) {
              clearInterval(this.pollIntervalId);
              if (shouldRender404(requests, this.criticalInitialRequestIds)) {
                return (
                  <Error404View
                    resourceName={`Model ${modelName} v${version}`}
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
                  handleStageTransitionDropdownSelect={this.handleStageTransitionDropdownSelect}
                  handleEditDescription={this.handleEditDescription}
                  deleteModelVersionApi={this.props.deleteModelVersionApi}
                  history={history}
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
  const { modelName, version } = ownProps.match.params;
  const modelVersion = getModelVersion(state, modelName, version);
  const runInfo = getRunInfo(modelVersion && modelVersion.run_id, state);
  const tags = runInfo && getRunTags(runInfo.getRunUuid(), state);
  const runDisplayName = tags && Utils.getRunDisplayName(tags, runInfo.getRunUuid());
  const { apis } = state;
  return {
    modelName,
    version: Number(version),
    modelVersion,
    runInfo,
    runDisplayName,
    apis,
  };
};

const mapDispatchToProps = {
  getModelVersionApi,
  updateModelVersionApi,
  deleteModelVersionApi,
  getRunApi,
};

export default ComponentOverrides.ModelVersionPage ||
connect(
  mapStateToProps,
  mapDispatchToProps,
)(ModelVersionPage);
