import React from 'react';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import { getUUID } from '../../Actions';
import {
  searchModelVersionsApi,
  getRegisteredModelDetailsApi,
  updateRegisteredModelApi,
  deleteRegisteredModelApi,
} from '../actions';
import { ModelView } from './ModelView';
import { getModelVersions } from '../reducers';
import { MODEL_VERSION_STATUS_POLL_INTERVAL as POLL_INTERVAL } from '../constants';
import RequestStateWrapper, { triggerError } from '../../components/RequestStateWrapper';
import { Spinner } from '../../components/Spinner';
import { Error404View } from '../../common/components/Error404View';
import { shouldRender404 } from '../../common/utils';
import { modelListPageRoute } from '../routes';

export class ModelPage extends React.Component {
  static propTypes = {
    modelName: PropTypes.string.isRequired,
    model: PropTypes.object,
    modelVersions: PropTypes.array,
    searchModelVersionsApi: PropTypes.func.isRequired,
    getRegisteredModelDetailsApi: PropTypes.func.isRequired,
    updateRegisteredModelApi: PropTypes.func.isRequired,
    deleteRegisteredModelApi: PropTypes.func.isRequired,
    history: PropTypes.object.isRequired,
    apis: PropTypes.object.isRequired,
  };

  initSearchModelVersionsApiId = getUUID();
  initGetRegisteredModelDetailsApiId = getUUID();
  searchModelVersionsApiId = getUUID();
  getRegisteredModelDetailsApiId = getUUID();
  updateRegisteredModelApiId = getUUID();
  deleteRegisteredModelApiId = getUUID();

  criticalInitialRequestIds = [
    this.initSearchModelVersionsApiId,
    this.initGetRegisteredModelDetailsApiId,
  ];

  handleEditDescription = (description) => {
    const { model } = this.props;
    return this.props
      .updateRegisteredModelApi(
        model.registered_model,
        undefined,
        description,
        this.updateRegisteredModelApiId,
      )
      .then(this.loadData);
  };

  handleDelete = () => {
    const { model } = this.props;
    return this.props.deleteRegisteredModelApi(
      model.registered_model,
      this.deleteRegisteredModelApiId
    );
  };

  loadData = (isInitialLoading) => {
    const { modelName } = this.props;
    return Promise.all([
      this.props.getRegisteredModelDetailsApi(
        modelName,
        isInitialLoading === true
          ? this.initGetRegisteredModelDetailsApiId
          : this.getRegisteredModelDetailsApiId,
      ),
      this.props.searchModelVersionsApi(
        { name: modelName },
        isInitialLoading === true
          ? this.initSearchModelVersionsApiId
          : this.searchModelVersionsApiId,
      ),
    ]).catch(console.error);
  };

  pollModelVersions = () => {
    const { modelName, apis } = this.props;
    const pollRequest = apis[this.searchModelVersionsApiId];
    if (!(pollRequest && pollRequest.active)) {
      this.props
        .searchModelVersionsApi({ name: modelName }, this.searchModelVersionsApiId)
        .catch(console.error);
    }
  };

  componentDidMount() {
    this.loadData(true);
    this.pollIntervalId = setInterval(this.pollModelVersions, POLL_INTERVAL);
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
              if (shouldRender404(requests, [this.initGetRegisteredModelDetailsApiId])) {
                return (
                  <Error404View
                    resourceName={`Model ${modelName}`}
                    fallbackHomePageReactRoute={modelListPageRoute}
                  />
                );
              }
              // TODO(Zangr) Have a more generic boundary to handle all errors, not just 404.
              triggerError(requests);
            } else if (loading) {
              return <Spinner />;
            } else if (model) { // Null check to prevent NPE after delete operation
              return (
                <ModelView
                  model={model}
                  modelVersions={modelVersions}
                  handleEditDescription={this.handleEditDescription}
                  handleDelete={this.handleDelete}
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
  const { modelName } = ownProps.match.params;
  const model = state.entities.modelByName[modelName];
  const modelVersions = getModelVersions(state, modelName);
  const { apis } = state;
  return { modelName, model, modelVersions, apis };
};

const mapDispatchToProps = {
  searchModelVersionsApi,
  getRegisteredModelDetailsApi,
  updateRegisteredModelApi,
  deleteRegisteredModelApi,
};

export default connect(mapStateToProps, mapDispatchToProps)(ModelPage);
