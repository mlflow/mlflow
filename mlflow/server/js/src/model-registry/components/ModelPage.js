import React from 'react';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import {
  searchModelVersionsApi,
  getRegisteredModelApi,
  updateRegisteredModelApi,
  deleteRegisteredModelApi,
} from '../actions';
import { ModelView } from './ModelView';
import { getModelVersions } from '../reducers';
import { MODEL_VERSION_STATUS_POLL_INTERVAL as POLL_INTERVAL } from '../constants';
import RequestStateWrapper, { triggerError } from '../../common/components/RequestStateWrapper';
import { Spinner } from '../../common/components/Spinner';
import { ErrorView } from '../../common/components/ErrorView';
import { modelListPageRoute } from '../routes';
import Utils from '../../common/utils/Utils';
import { getUUID } from '../../common/utils/ActionUtils';

export class ModelPageImpl extends React.Component {
  static propTypes = {
    // own props
    history: PropTypes.object.isRequired,
    match: PropTypes.object.isRequired,
    // connected props
    modelName: PropTypes.string.isRequired,
    model: PropTypes.object,
    modelVersions: PropTypes.array,
    searchModelVersionsApi: PropTypes.func.isRequired,
    getRegisteredModelApi: PropTypes.func.isRequired,
    updateRegisteredModelApi: PropTypes.func.isRequired,
    deleteRegisteredModelApi: PropTypes.func.isRequired,
    apis: PropTypes.object.isRequired,
  };

  initSearchModelVersionsApiId = getUUID();
  initgetRegisteredModelApiId = getUUID();
  searchModelVersionsApiId = getUUID();
  getRegisteredModelApiId = getUUID();
  updateRegisteredModelApiId = getUUID();
  deleteRegisteredModelApiId = getUUID();

  criticalInitialRequestIds = [this.initSearchModelVersionsApiId, this.initgetRegisteredModelApiId];

  pollingRelatedRequestIds = [this.getRegisteredModelApiId, this.searchModelVersionsApiId];

  hasPendingPollingRequest = () =>
    this.pollingRelatedRequestIds.every((requestId) => {
      const request = this.props.apis[requestId];
      return Boolean(request && request.active);
    });

  handleEditDescription = (description) => {
    const { model } = this.props;
    return this.props
      .updateRegisteredModelApi(model.name, description, this.updateRegisteredModelApiId)
      .then(this.loadData);
  };

  handleDelete = () => {
    const { model } = this.props;
    return this.props.deleteRegisteredModelApi(model.name, this.deleteRegisteredModelApiId);
  };

  loadData = (isInitialLoading) => {
    const { modelName } = this.props;
    return Promise.all([
      this.props.getRegisteredModelApi(
        modelName,
        isInitialLoading === true ? this.initgetRegisteredModelApiId : this.getRegisteredModelApiId,
      ),
      this.props.searchModelVersionsApi(
        { name: modelName },
        isInitialLoading === true
          ? this.initSearchModelVersionsApiId
          : this.searchModelVersionsApiId,
      ),
    ]);
  };

  pollData = () => {
    const { modelName, history } = this.props;
    if (!this.hasPendingPollingRequest() && Utils.isBrowserTabVisible()) {
      return this.loadData().catch((e) => {
        if (e.getErrorCode() === 'RESOURCE_DOES_NOT_EXIST') {
          Utils.logErrorAndNotifyUser(e);
          this.props.deleteRegisteredModelApi(modelName, undefined, true);
          history.push(modelListPageRoute);
        } else {
          console.error(e);
        }
      });
    }
    return Promise.resolve();
  };

  componentDidMount() {
    this.loadData(true).catch(console.error);
    this.pollIntervalId = setInterval(this.pollData, POLL_INTERVAL);
  }

  componentWillUnmount() {
    clearInterval(this.pollIntervalId);
  }

  render() {
    const { model, modelVersions, history, modelName } = this.props;
    return (
      <div className='App-content'>
        <RequestStateWrapper requestIds={this.criticalInitialRequestIds}>
          {(loading, hasError, requests) => {
            if (hasError) {
              clearInterval(this.pollIntervalId);
              if (Utils.shouldRender404(requests, [this.initgetRegisteredModelApiId])) {
                return (
                  <ErrorView
                    statusCode={404}
                    subMessage={`Model ${modelName} does not exist`}
                    fallbackHomePageReactRoute={modelListPageRoute}
                  />
                );
              }
              // TODO(Zangr) Have a more generic boundary to handle all errors, not just 404.
              triggerError(requests);
            } else if (loading) {
              return <Spinner />;
            } else if (model) {
              // Null check to prevent NPE after delete operation
              return (
                <ModelView
                  model={model}
                  modelVersions={modelVersions}
                  handleEditDescription={this.handleEditDescription}
                  handleDelete={this.handleDelete}
                  showEditPermissionModal={this.showEditPermissionModal}
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
  const modelName = decodeURIComponent(ownProps.match.params.modelName);
  const model = state.entities.modelByName[modelName];
  const modelVersions = getModelVersions(state, modelName);
  const { apis } = state;
  return { modelName, model, modelVersions, apis };
};

const mapDispatchToProps = {
  searchModelVersionsApi,
  getRegisteredModelApi,
  updateRegisteredModelApi,
  deleteRegisteredModelApi,
};

export const ModelPage = connect(mapStateToProps, mapDispatchToProps)(ModelPageImpl);
