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
import LocalStorageUtils from '../../common/utils/LocalStorageUtils';
import { MODEL_VERSIONS_SEARCH_TIMESTAMP_FIELD, MODEL_VERSIONS_PER_PAGE_COMPACT } from '../constants';
import { PageContainer } from '../../common/components/PageContainer';
import RequestStateWrapper, { triggerError } from '../../common/components/RequestStateWrapper';
import { Spinner } from '../../common/components/Spinner';
import { ErrorView } from '../../common/components/ErrorView';
import { ModelRegistryRoutes } from '../routes';
import Utils from '../../common/utils/Utils';
import { getUUID } from '../../common/utils/ActionUtils';
import { injectIntl } from 'react-intl';
import { withRouterNext } from '../../common/utils/withRouterNext';
import type { WithRouterNextProps } from '../../common/utils/withRouterNext';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { ErrorCodes } from '../../common/constants';

type ModelPageImplState = {
  maxResultsSelection: number;
  pageTokens: Record<number, string | null>;
  loading: boolean;
  error: Error | undefined;
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
  searchParams: URLSearchParams;
  setSearchParams: (params: URLSearchParams | ((prev: URLSearchParams) => URLSearchParams)) => void;
  intl?: any;
};

/**
 * Returns a LocalStorageStore instance that can be used to persist data associated with the
 * ModelRegistry component.
 */
function getModelPageSessionStore(key: any) {
  return LocalStorageUtils.getSessionScopedStoreForComponent('ModelPage', key);
}

export function getOrderByExpr(orderByKey: string, orderByAsc: boolean): string {
  return orderByKey ? `${orderByKey} ${orderByAsc ? 'ASC' : 'DESC'}` : '';
}

export class ModelPageImpl extends React.Component<ModelPageImplProps, ModelPageImplState> {
  constructor(props: ModelPageImplProps) {
    super(props);
    const persistedPageTokens = this.getPersistedPageTokens();
    const maxResultsForTokens = this.getPersistedMaxResults();

    this.state = {
      maxResultsSelection: maxResultsForTokens,
      pageTokens: persistedPageTokens,
      loading: true,
      error: undefined,
    };
  }
  modelPageStoreKey = 'ModelPageStore';
  defaultPersistedPageTokens = { 1: null };

  initSearchModelVersionsApiRequestId = getUUID();
  initgetRegisteredModelApiRequestId = getUUID();
  updateRegisteredModelApiId = getUUID();
  deleteRegisteredModelApiId = getUUID();

  criticalInitialRequestIds = [this.initSearchModelVersionsApiRequestId, this.initgetRegisteredModelApiRequestId];

  componentDidMount() {
    this.loadModelVersions(true);
  }

  updateUrlWithState(orderByKey: string, orderByAsc: boolean, page: number): Promise<void> {
    return new Promise((resolve) => {
      const newParams = new URLSearchParams(this.props.searchParams);
      if (orderByKey) newParams.set('orderByKey', orderByKey);
      if (orderByAsc !== undefined) newParams.set('orderByAsc', String(orderByAsc));
      if (page) newParams.set('page', String(page));
      this.props.setSearchParams(newParams);
      resolve();
    });
  }

  resetHistoryState() {
    this.setState((prevState: any) => ({
      pageTokens: this.defaultPersistedPageTokens,
    }));
    this.setPersistedPageTokens(this.defaultPersistedPageTokens);
  }

  isEmptyPageResponse = (value: any) => {
    return !value || !value.model_versions || !value.next_page_token;
  };

  // Loads the initial set of model versions.
  loadModelVersions(isInitialLoading = true) {
    this.loadPage(this.currentPage, isInitialLoading, true);
  }

  get currentPage() {
    const urlPage = parseInt(this.props.searchParams.get('page') || '1', 10);
    return isNaN(urlPage) ? 1 : urlPage;
  }

  get orderByKey() {
    return this.props.searchParams.get('orderByKey') ?? MODEL_VERSIONS_SEARCH_TIMESTAMP_FIELD;
  }

  get orderByAsc() {
    return this.props.searchParams.get('orderByAsc') === 'true';
  }

  getPersistedPageTokens() {
    const store = getModelPageSessionStore(this.modelPageStoreKey);
    if (store && store.getItem('page_tokens')) {
      return JSON.parse(store.getItem('page_tokens'));
    } else {
      return this.defaultPersistedPageTokens;
    }
  }

  setPersistedPageTokens(page_tokens: any) {
    const store = getModelPageSessionStore(this.modelPageStoreKey);
    if (store) {
      store.setItem('page_tokens', JSON.stringify(page_tokens));
    }
  }

  getPersistedMaxResults() {
    const store = getModelPageSessionStore(this.modelPageStoreKey);
    if (store && store.getItem('max_results')) {
      return parseInt(store.getItem('max_results'), 10);
    } else {
      return MODEL_VERSIONS_PER_PAGE_COMPACT;
    }
  }

  setMaxResultsInStore(max_results: any) {
    const store = getModelPageSessionStore(this.modelPageStoreKey);
    store.setItem('max_results', max_results.toString());
  }

  handleEditDescription = (description: any) => {
    const { model } = this.props;
    return this.props
      .updateRegisteredModelApi(model.name, description, this.updateRegisteredModelApiId)
      .then(() => this.loadPage(1, false, true));
  };

  handleDelete = () => {
    const { model } = this.props;
    return this.props.deleteRegisteredModelApi(model.name, this.deleteRegisteredModelApiId).then(() => {
      this.props.navigate(ModelRegistryRoutes.modelListPageRoute);
    });
  };

  loadPage = (page: number, isInitialLoading: boolean, loadModelMetadata = false) => {
    const { modelName } = this.props;
    const { pageTokens } = this.state;
    this.setState({ loading: true, error: undefined });
    const filters_obj = { name: modelName };
    const promiseValues = [
      this.props
        .searchModelVersionsApi(
          filters_obj,
          isInitialLoading ? this.initSearchModelVersionsApiRequestId : null,
          this.state.maxResultsSelection,
          getOrderByExpr(this.orderByKey, this.orderByAsc),
          pageTokens[page],
        )
        .then((response: any) => {
          this.updatePageState(page, response);
        }),
    ];
    if (loadModelMetadata) {
      promiseValues.push(
        this.props.getRegisteredModelApi(
          modelName,
          isInitialLoading === true ? this.initgetRegisteredModelApiRequestId : null,
        ),
      );
    }
    return Promise.all(promiseValues)
      .then(() => {
        this.setState({ loading: false });
      })
      .catch((error) => {
        this.setState({ loading: false, error });
        this.resetHistoryState();
      });
  };

  getNextPageTokenFromResponse(response: any) {
    const { value } = response;
    if (this.isEmptyPageResponse(value)) {
      // Why we could be here:
      // 1. There are no models returned: we went to the previous page but all models after that
      //    page's token has been deleted.
      // 2. If `next_page_token` is not returned, assume there is no next page.
      return null;
    } else {
      return value.next_page_token;
    }
  }

  updatePageState = (page: any, response = {}) => {
    const nextPageToken = this.getNextPageTokenFromResponse(response);
    this.setState(
      (prevState: any) => ({
        pageTokens: {
          ...prevState.pageTokens,
          [page + 1]: nextPageToken,
        },
      }),
      () => {
        this.setPersistedPageTokens(this.state.pageTokens);
      },
    );
  };

  handleMaxResultsChange = (key: string) => {
    this.setState({ maxResultsSelection: parseInt(key, 10) }, () => {
      this.resetHistoryState();
      const { maxResultsSelection } = this.state;
      this.setMaxResultsInStore(maxResultsSelection);
      this.loadPage(1, false);
    });
  };

  handleClickNext = () => {
    const nextPage = this.currentPage + 1;
    this.updateUrlWithState(this.orderByKey, this.orderByAsc, nextPage).then(() => {
      this.loadPage(nextPage, false);
    });
  };

  handleClickPrev = () => {
    const prevPage = this.currentPage - 1;
    this.updateUrlWithState(this.orderByKey, this.orderByAsc, prevPage).then(() => {
      this.loadPage(prevPage, false);
    });
  };

  handleClickSortableColumn = (orderByKey: string, orderByDesc: boolean) => {
    const orderByAsc = !orderByDesc;
    this.resetHistoryState();
    this.updateUrlWithState(orderByKey, orderByAsc, 1).then(() => {
      this.loadPage(1, false);
    });
  };

  getMaxResultsSelection = () => {
    return this.state.maxResultsSelection;
  };

  render() {
    const { model, modelVersions, navigate, modelName } = this.props;
    const { pageTokens } = this.state;
    return (
      <PageContainer>
        <RequestStateWrapper requestIds={this.criticalInitialRequestIds}>
          {(loading: any, hasError: any, requests: any) => {
            if (hasError) {
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
                  onMetadataUpdated={this.loadModelVersions}
                  orderByKey={this.orderByKey}
                  orderByAsc={this.orderByAsc}
                  currentPage={this.currentPage}
                  nextPageToken={pageTokens[this.currentPage + 1]}
                  onClickNext={this.handleClickNext}
                  onClickPrev={this.handleClickPrev}
                  onClickSortableColumn={this.handleClickSortableColumn}
                  onSetMaxResult={this.handleMaxResultsChange}
                  maxResultValue={this.getMaxResultsSelection()}
                  loading={this.state.loading}
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
