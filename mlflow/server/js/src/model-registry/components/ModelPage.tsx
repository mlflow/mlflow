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
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { withRouterNext } from '../../common/utils/withRouterNext';
import type { WithRouterNextProps } from '../../common/utils/withRouterNext';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { ErrorCodes } from '../../common/constants';

type PaginationState = {
  currentPage: number;
  maxResultValue: number;
  currentPageToken: string | null;
  nextPageToken: string | null;
  previousPageTokens: Array<string | null>;
};

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

export class ModelPageImpl extends React.Component<ModelPageImplProps, PaginationState> {
  hasUnfilledRequests: any;
  pollIntervalId: any;

  initSearchModelVersionsApiRequestId = getUUID();
  initgetRegisteredModelApiRequestId = getUUID();
  updateRegisteredModelApiId = getUUID();
  deleteRegisteredModelApiId = getUUID();

  criticalInitialRequestIds = [this.initSearchModelVersionsApiRequestId, this.initgetRegisteredModelApiRequestId];

  state: PaginationState = {
    currentPage: 1,
    maxResultValue: 25,
    currentPageToken: null,
    nextPageToken: null,
    previousPageTokens: [],
  };

  handleClickNext = () => {
    const { nextPageToken, currentPageToken, previousPageTokens, currentPage } = this.state;
    if (!nextPageToken) return;

    this.setState(
      {
        previousPageTokens: [...previousPageTokens, currentPageToken],
        currentPageToken: nextPageToken,
        nextPageToken: null,
        currentPage: currentPage + 1,
      },
      () => this.fetchData(this.state.currentPageToken),
    );
  };

  handleClickPrev = () => {
    const { previousPageTokens, currentPage } = this.state;
    if (previousPageTokens.length === 0) return;

    const tokens = [...previousPageTokens];
    const prevToken = tokens.pop() ?? null;

    this.setState(
      {
        previousPageTokens: tokens,
        currentPageToken: prevToken,
        nextPageToken: null,
        currentPage: currentPage - 1,
      },
      () => this.fetchData(this.state.currentPageToken),
    );
  };

  handleSetMaxResult = ({ key: pageSize }: { key: number }) => {
    this.setState(
      {
        maxResultValue: pageSize,
        currentPage: 1,
        currentPageToken: null,
        nextPageToken: null,
        previousPageTokens: [],
      },
      () => this.fetchData(null),
    );
  };

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
  fetchData = (pageToken: string | null) => {
    const { modelName } = this.props;
    this.hasUnfilledRequests = true;

    return this.props
      .searchModelVersionsApi(
        {
          filterObj: { name: modelName },
          maxResults: this.state.maxResultValue,
          pageToken: pageToken ?? undefined,
        },
        this.initSearchModelVersionsApiRequestId,
      )
      .then((resp: any) => {
        const nextToken = resp?.next_page_token ?? resp?.value?.next_page_token ?? null;
        this.setState({ nextPageToken: nextToken });
      })
      .finally(() => (this.hasUnfilledRequests = false));
  };

  loadData = (isInitial = false) => {
    const { modelName } = this.props;

    const apiCalls = isInitial
      ? [this.props.getRegisteredModelApi(modelName, this.initgetRegisteredModelApiRequestId), this.fetchData(null)]
      : [this.fetchData(null)];

    return Promise.all(apiCalls);
  };

  pollData = () => {
    const { modelName, navigate } = this.props;
    if (!this.hasUnfilledRequests && Utils.isBrowserTabVisible()) {
      return this.loadData().catch((e) => {
        if (e instanceof ErrorWrapper && e.getErrorCode() === 'RESOURCE_DOES_NOT_EXIST') {
          Utils.logErrorAndNotifyUser(e);
          this.props.deleteRegisteredModelApi(modelName, undefined, true);
          navigate(ModelRegistryRoutes.modelListPageRoute);
        } else {
          // eslint-disable-next-line no-console -- TODO(FEINF-3587)
          console.error(e);
        }
        this.hasUnfilledRequests = false;
      });
    }
    return Promise.resolve();
  };

  componentDidMount() {
    this.loadData(true)
      .then(() => {
        const { modelVersions } = this.props;
        const hasPending = (modelVersions || []).some(({ status }) => status !== 'READY');
        if (hasPending) {
          this.pollIntervalId = setInterval(this.pollData, POLL_INTERVAL);
        }
      })
      .catch(console.error);
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
                  paginationProps={{
                    currentPage: this.state.currentPage,
                    hasPreviousPage: this.state.currentPage > 1,
                    hasNextPage: Boolean(this.state.nextPageToken),
                    maxResultValue: this.state.maxResultValue,
                    onNext: this.handleClickNext,
                    onPrev: this.handleClickPrev,
                    onPageSizeChange: this.handleSetMaxResult,
                  }}
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
