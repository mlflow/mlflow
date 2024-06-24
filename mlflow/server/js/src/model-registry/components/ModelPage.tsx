/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { connect } from 'react-redux';
import {
  searchModelVersionsApi,
  getRegisteredModelApi,
  updateRegisteredModelApi,
  deleteRegisteredModelApi,
} from '../actions';
import { ModelView } from './ModelView';
import { getModelVersions } from '../reducers';
import { MODEL_VERSION_STATUS_POLL_INTERVAL as POLL_INTERVAL } from '../constants';
import { PageContainer } from '../../common/components/PageContainer';
import RequestStateWrapper, { triggerError } from '../../common/components/RequestStateWrapper';
import { Spinner } from '../../common/components/Spinner';
import { ErrorView } from '../../common/components/ErrorView';
import { ModelRegistryRoutes } from '../routes';
import Utils from '../../common/utils/Utils';
import { getUUID } from '../../common/utils/ActionUtils';
import { injectIntl } from 'react-intl';
import { ErrorWrapper } from './../../common/utils/ErrorWrapper';
import { withRouterNext } from '../../common/utils/withRouterNext';
import type { WithRouterNextProps } from '../../common/utils/withRouterNext';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { ErrorCodes } from '../../common/constants';

type ModelPageImplProps = WithRouterNextProps<{ subpage: string }> & {
  modelName: string;
  model?: any;
  modelVersions?: any[];
  emailSubscriptionStatus?: string;
  userLevelEmailSubscriptionStatus?: string;
  searchModelVersionsApi: (...args: any[]) => any;
  getRegisteredModelApi: (...args: any[]) => any;
  updateRegisteredModelApi: (...args: any[]) => any;
  deleteRegisteredModelApi: (...args: any[]) => any;
  setEmailSubscriptionStatusApi: (...args: any[]) => any;
  getEmailSubscriptionStatusApi: (...args: any[]) => any;
  getUserLevelEmailSubscriptionStatusApi: (...args: any[]) => any;
  searchEndpointsByModelNameApi: (...args: any[]) => any;
  intl?: any;
};

export class ModelPageImpl extends React.Component<ModelPageImplProps> {
  hasUnfilledRequests: any;
  pollIntervalId: any;

  initSearchModelVersionsApiRequestId = getUUID();
  initgetRegisteredModelApiRequestId = getUUID();
  updateRegisteredModelApiId = getUUID();
  deleteRegisteredModelApiId = getUUID();

  criticalInitialRequestIds = [this.initSearchModelVersionsApiRequestId, this.initgetRegisteredModelApiRequestId];

  handleEditDescription = (description: any) => {
    const { model } = this.props;
    return this.props
      .updateRegisteredModelApi(model.name, description, this.updateRegisteredModelApiId)
      .then(this.loadData);
  };

  handleDelete = () => {
    const { model } = this.props;
    return this.props.deleteRegisteredModelApi(model.name, this.deleteRegisteredModelApiId);
  };

  loadData = (isInitialLoading: any) => {
    const { modelName } = this.props;
    this.hasUnfilledRequests = true;
    const promiseValues = [
      this.props.getRegisteredModelApi(
        modelName,
        isInitialLoading === true ? this.initgetRegisteredModelApiRequestId : null,
      ),
      this.props.searchModelVersionsApi(
        { name: modelName },
        isInitialLoading === true ? this.initSearchModelVersionsApiRequestId : null,
      ),
    ];
    return Promise.all(promiseValues).then(() => {
      this.hasUnfilledRequests = false;
    });
  };

  pollData = () => {
    const { modelName, navigate } = this.props;
    if (!this.hasUnfilledRequests && Utils.isBrowserTabVisible()) {
      // @ts-expect-error TS(2554): Expected 1 arguments, but got 0.
      return this.loadData().catch((e) => {
        if (e instanceof ErrorWrapper && e.getErrorCode() === 'RESOURCE_DOES_NOT_EXIST') {
          Utils.logErrorAndNotifyUser(e);
          this.props.deleteRegisteredModelApi(modelName, undefined, true);
          navigate(ModelRegistryRoutes.modelListPageRoute);
        } else {
          console.error(e);
        }
        this.hasUnfilledRequests = false;
      });
    }
    return Promise.resolve();
  };

  componentDidMount() {
    this.loadData(true).catch(console.error);
    this.hasUnfilledRequests = false;
    this.pollIntervalId = setInterval(this.pollData, POLL_INTERVAL);
  }

  componentWillUnmount() {
    clearInterval(this.pollIntervalId);
  }

  render() {
    const { model, modelVersions, navigate, modelName } = this.props;
    return (
      <PageContainer>
        <RequestStateWrapper
          requestIds={this.criticalInitialRequestIds}
          // eslint-disable-next-line no-trailing-spaces
        >
          {(loading: any, hasError: any, requests: any) => {
            if (hasError) {
              clearInterval(this.pollIntervalId);
              if (Utils.shouldRender404(requests, [this.initgetRegisteredModelApiRequestId])) {
                return (
                  <ErrorView
                    statusCode={404}
                    subMessage={this.props.intl.formatMessage(
                      {
                        defaultMessage: 'Model {modelName} does not exist',
                        description: 'Sub-message text for error message on overall model page',
                      },
                      {
                        modelName: modelName,
                      },
                    )}
                    fallbackHomePageReactRoute={ModelRegistryRoutes.modelListPageRoute}
                  />
                );
              }
              const permissionDeniedErrors = requests.filter((request: any) => {
                return (
                  this.criticalInitialRequestIds.includes(request.id) &&
                  request.error?.getErrorCode() === ErrorCodes.PERMISSION_DENIED
                );
              });
              if (permissionDeniedErrors && permissionDeniedErrors[0]) {
                return (
                  <ErrorView
                    statusCode={403}
                    subMessage={this.props.intl.formatMessage(
                      {
                        defaultMessage: 'Permission denied for {modelName}. Error: "{errorMsg}"',
                        description: 'Permission denied error message on registered model detail page',
                      },
                      {
                        modelName: modelName,
                        errorMsg: permissionDeniedErrors[0].error?.getMessageField(),
                      },
                    )}
                    fallbackHomePageReactRoute={ModelRegistryRoutes.modelListPageRoute}
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
                  navigate={navigate}
                  onMetadataUpdated={this.loadData}
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

const mapStateToProps = (state: any, ownProps: WithRouterNextProps<{ modelName: string }>) => {
  const modelName = decodeURIComponent(ownProps.params.modelName);
  const model = state.entities.modelByName[modelName];
  const modelVersions = getModelVersions(state, modelName);
  return {
    modelName,
    model,
    modelVersions,
  };
};

const mapDispatchToProps = {
  searchModelVersionsApi,
  getRegisteredModelApi,
  updateRegisteredModelApi,
  deleteRegisteredModelApi,
};

const ModelPageWithRouter = withRouterNext(
  // @ts-expect-error TS(2769): No overload matches this call.
  connect(mapStateToProps, mapDispatchToProps)(injectIntl(ModelPageImpl)),
);

export const ModelPage = withErrorBoundary(ErrorUtils.mlflowServices.MODEL_REGISTRY, ModelPageWithRouter);

export default ModelPage;
