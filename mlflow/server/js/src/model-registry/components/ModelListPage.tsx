/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { ModelListView } from './ModelListView';
import { connect } from 'react-redux';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import { getUUID } from '../../common/utils/ActionUtils';
import Utils from '../../common/utils/Utils';
import { getCombinedSearchFilter, constructSearchInputFromURLState } from '../utils/SearchUtils';
import {
  AntdTableSortOrder,
  REGISTERED_MODELS_PER_PAGE_COMPACT,
  REGISTERED_MODELS_SEARCH_NAME_FIELD,
} from '../constants';
import { searchRegisteredModelsApi } from '../actions';
import LocalStorageUtils from '../../common/utils/LocalStorageUtils';
import { withRouterNext } from '../../common/utils/withRouterNext';
import type { WithRouterNextProps } from '../../common/utils/withRouterNext';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';

type ModelListPageImplProps = WithRouterNextProps & {
  models?: any[];
  searchRegisteredModelsApi: (...args: any[]) => any;
};

type ModelListPageImplState = any;

export class ModelListPageImpl extends React.Component<
  ModelListPageImplProps,
  ModelListPageImplState
> {
  constructor(props: ModelListPageImplProps) {
    super(props);
    this.state = {
      orderByKey: REGISTERED_MODELS_SEARCH_NAME_FIELD,
      orderByAsc: true,
      currentPage: 1,
      maxResultsSelection: REGISTERED_MODELS_PER_PAGE_COMPACT,
      pageTokens: {},
      loading: false,
      searchInput: constructSearchInputFromURLState(this.getUrlState()),
    };
  }
  modelListPageStoreKey = 'ModelListPageStore';
  defaultPersistedPageTokens = { 1: null };
  initialSearchRegisteredModelsApiId = getUUID();
  searchRegisteredModelsApiId = getUUID();
  criticalInitialRequestIds = [this.initialSearchRegisteredModelsApiId];

  getUrlState() {
    return this.props.location ? Utils.getSearchParamsFromUrl(this.props.location.search) : {};
  }

  componentDidMount() {
    const urlState = this.getUrlState();
    const persistedPageTokens = this.getPersistedPageTokens();
    const maxResultsForTokens = this.getPersistedMaxResults();
    // eslint-disable-next-line react/no-did-mount-set-state
    this.setState(
      {
        // @ts-expect-error TS(4111): Property 'orderByKey' comes from an index signatur... Remove this comment to see the full error message
        orderByKey: urlState.orderByKey === undefined ? this.state.orderByKey : urlState.orderByKey,
        orderByAsc:
          // @ts-expect-error TS(4111): Property 'orderByAsc' comes from an index signatur... Remove this comment to see the full error message
          urlState.orderByAsc === undefined
            ? // @ts-expect-error TS(4111): Property 'orderByAsc' comes from an index signatur... Remove this comment to see the full error message
              this.state.orderByAsc
            : // @ts-expect-error TS(4111): Property 'orderByAsc' comes from an index signatur... Remove this comment to see the full error message
              urlState.orderByAsc === 'true',
        currentPage:
          // @ts-expect-error TS(4111): Property 'page' comes from an index signature, so ... Remove this comment to see the full error message
          urlState.page !== undefined && (urlState as any).page in persistedPageTokens
            ? // @ts-expect-error TS(2345): Argument of type 'unknown' is not assignable to pa... Remove this comment to see the full error message
              parseInt(urlState.page, 10)
            : // @ts-expect-error TS(4111): Property 'currentPage' comes from an index signatu... Remove this comment to see the full error message
              this.state.currentPage,
        maxResultsSelection: maxResultsForTokens,
        pageTokens: persistedPageTokens,
      },
      () => {
        this.loadModels(true);
      },
    );
  }

  getPersistedPageTokens() {
    const store = ModelListPageImpl.getLocalStore(this.modelListPageStoreKey);
    if (store && store.getItem('page_tokens')) {
      return JSON.parse(store.getItem('page_tokens'));
    } else {
      return this.defaultPersistedPageTokens;
    }
  }

  setPersistedPageTokens(page_tokens: any) {
    const store = ModelListPageImpl.getLocalStore(this.modelListPageStoreKey);
    if (store) {
      store.setItem('page_tokens', JSON.stringify(page_tokens));
    }
  }

  getPersistedMaxResults() {
    const store = ModelListPageImpl.getLocalStore(this.modelListPageStoreKey);
    if (store && store.getItem('max_results')) {
      return parseInt(store.getItem('max_results'), 10);
    } else {
      return REGISTERED_MODELS_PER_PAGE_COMPACT;
    }
  }

  setMaxResultsInStore(max_results: any) {
    const store = ModelListPageImpl.getLocalStore(this.modelListPageStoreKey);
    store.setItem('max_results', max_results.toString());
  }

  /**
   * Returns a LocalStorageStore instance that can be used to persist data associated with the
   * ModelRegistry component.
   */
  static getLocalStore(key: any) {
    return LocalStorageUtils.getSessionScopedStoreForComponent('ModelListPage', key);
  }

  // Loads the initial set of models.
  loadModels(isInitialLoading = false) {
    // @ts-expect-error TS(4111): Property 'currentPage' comes from an index signatu... Remove this comment to see the full error message
    this.loadPage(this.state.currentPage, undefined, undefined, isInitialLoading);
  }

  resetHistoryState() {
    this.setState((prevState: any) => ({
      currentPage: 1,
      pageTokens: this.defaultPersistedPageTokens,
    }));
    this.setPersistedPageTokens(this.defaultPersistedPageTokens);
  }

  /**
   *
   * @param orderByKey column key to sort by
   * @param orderByAsc is sort by ascending order
   * @returns {string} ex. 'name ASC'
   */
  static getOrderByExpr = (orderByKey: any, orderByAsc: any) =>
    orderByKey ? `${orderByKey} ${orderByAsc ? 'ASC' : 'DESC'}` : '';

  pollIntervalId: any;

  isEmptyPageResponse = (value: any) => {
    return !value || !value.registered_models || !value.next_page_token;
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
        currentPage: page,

        pageTokens: {
          ...prevState.pageTokens,
          [page + 1]: nextPageToken,
        },
      }),
      () => {
        // @ts-expect-error TS(4111): Property 'pageTokens' comes from an index signatur... Remove this comment to see the full error message
        this.setPersistedPageTokens(this.state.pageTokens);
      },
    );
  };

  handleSearch = (callback: any, errorCallback: any, searchInput: any) => {
    this.resetHistoryState();
    this.setState({ searchInput: searchInput }, () => {
      // @ts-expect-error TS(2554): Expected 4 arguments, but got 3.
      this.loadPage(1, callback, errorCallback);
    });
  };

  handleClear = (callback: any, errorCallback: any) => {
    this.setState(
      {
        orderByKey: REGISTERED_MODELS_SEARCH_NAME_FIELD,
        orderByAsc: true,
        searchInput: '',
        // eslint-disable-nextline
      },
      () => {
        this.updateUrlWithSearchFilter('', REGISTERED_MODELS_SEARCH_NAME_FIELD, true, 1);
        // @ts-expect-error TS(2554): Expected 4 arguments, but got 3.
        this.loadPage(1, callback, errorCallback);
      },
    );
  };

  // Note: this method is no longer used by the UI but is used in tests. Probably best to refactor at some point.
  handleSearchInputChange = (searchInput: any) => {
    this.setState({ searchInput: searchInput });
  };

  updateUrlWithSearchFilter = (searchInput: any, orderByKey: any, orderByAsc: any, page: any) => {
    const urlParams = {};
    if (searchInput) {
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      urlParams['searchInput'] = searchInput;
    }
    if (orderByKey && orderByKey !== REGISTERED_MODELS_SEARCH_NAME_FIELD) {
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      urlParams['orderByKey'] = orderByKey;
    }
    if (orderByAsc === false) {
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      urlParams['orderByAsc'] = orderByAsc;
    }
    if (page && page !== 1) {
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      urlParams['page'] = page;
    }
    const newUrl = `/models?${Utils.getSearchUrlFromState(urlParams)}`;
    if (newUrl !== this.props.location.pathname + this.props.location.search) {
      this.props.navigate(newUrl);
    }
  };

  handleMaxResultsChange = (key: any, callback: any, errorCallback: any) => {
    this.setState({ maxResultsSelection: parseInt(key, 10) }, () => {
      this.resetHistoryState();
      const { maxResultsSelection } = this.state;
      this.setMaxResultsInStore(maxResultsSelection);
      // @ts-expect-error TS(2554): Expected 4 arguments, but got 3.
      this.loadPage(1, callback, errorCallback);
    });
  };

  handleClickNext = (callback: any, errorCallback: any) => {
    const { currentPage } = this.state;
    // @ts-expect-error TS(2554): Expected 4 arguments, but got 3.
    this.loadPage(currentPage + 1, callback, errorCallback);
  };

  handleClickPrev = (callback: any, errorCallback: any) => {
    const { currentPage } = this.state;
    // @ts-expect-error TS(2554): Expected 4 arguments, but got 3.
    this.loadPage(currentPage - 1, callback, errorCallback);
  };

  handleClickSortableColumn = (
    orderByKey: any,
    sortOrder: any,
    callback: any,
    errorCallback: any,
  ) => {
    const orderByAsc = sortOrder !== AntdTableSortOrder.DESC; // default to true
    this.setState({ orderByKey, orderByAsc }, () => {
      this.resetHistoryState();
      // @ts-expect-error TS(2554): Expected 4 arguments, but got 3.
      this.loadPage(1, callback, errorCallback);
    });
  };

  getMaxResultsSelection = () => {
    // @ts-expect-error TS(4111): Property 'maxResultsSelection' comes from an index... Remove this comment to see the full error message
    return this.state.maxResultsSelection;
  };

  loadPage(page: any, callback: any, errorCallback: any, isInitialLoading: any) {
    const {
      searchInput,
      pageTokens,
      orderByKey,
      orderByAsc,
      // eslint-disable-nextline
    } = this.state;
    this.setState({ loading: true });
    this.updateUrlWithSearchFilter(searchInput, orderByKey, orderByAsc, page);
    this.props
      .searchRegisteredModelsApi(
        getCombinedSearchFilter({
          query: searchInput,
          // eslint-disable-nextline
        }),
        // @ts-expect-error TS(4111): Property 'maxResultsSelection' comes from an index... Remove this comment to see the full error message
        this.state.maxResultsSelection,
        ModelListPageImpl.getOrderByExpr(orderByKey, orderByAsc),
        pageTokens[page],
        isInitialLoading
          ? this.initialSearchRegisteredModelsApiId
          : this.searchRegisteredModelsApiId,
      )
      .then((r: any) => {
        this.updatePageState(page, r);
        this.setState({ loading: false });
        callback && callback();
      })
      .catch((e: any) => {
        Utils.logErrorAndNotifyUser(e);
        this.setState({ currentPage: 1 });
        this.resetHistoryState();
        errorCallback && errorCallback();
      });
  }

  render() {
    const {
      orderByKey,
      orderByAsc,
      currentPage,
      pageTokens,
      searchInput,
      // eslint-disable-nextline
    } = this.state;
    const { models } = this.props;
    return (
      <ScrollablePageWrapper>
        <RequestStateWrapper
          requestIds={[this.criticalInitialRequestIds]}
          // eslint-disable-next-line no-trailing-spaces
        >
          <ModelListView
            // @ts-expect-error TS(2322): Type '{ models: any[] | undefined; loading: any; e... Remove this comment to see the full error message
            models={models}
            // @ts-expect-error TS(4111): Property 'loading' comes from an index signature, ... Remove this comment to see the full error message
            loading={this.state.loading}
            searchInput={searchInput}
            orderByKey={orderByKey}
            orderByAsc={orderByAsc}
            currentPage={currentPage}
            nextPageToken={pageTokens[currentPage + 1]}
            onSearch={this.handleSearch}
            onClickNext={this.handleClickNext}
            onClickPrev={this.handleClickPrev}
            onClickSortableColumn={this.handleClickSortableColumn}
            onSetMaxResult={this.handleMaxResultsChange}
            getMaxResultValue={this.getMaxResultsSelection}
          />
        </RequestStateWrapper>
      </ScrollablePageWrapper>
    );
  }
}

const mapStateToProps = (state: any) => {
  const models = Object.values(state.entities.modelByName);
  return {
    models,
  };
};

const mapDispatchToProps = {
  searchRegisteredModelsApi,
};

const ModelListPageWithRouter = withRouterNext(
  connect(mapStateToProps, mapDispatchToProps)(ModelListPageImpl),
);

export const ModelListPage = withErrorBoundary(
  ErrorUtils.mlflowServices.MODEL_REGISTRY,
  ModelListPageWithRouter,
);
