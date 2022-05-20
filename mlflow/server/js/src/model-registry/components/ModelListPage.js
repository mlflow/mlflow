import React from 'react';
import { ModelListView } from './ModelListView';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import { getUUID } from '../../common/utils/ActionUtils';
import Utils from '../../common/utils/Utils';
import { appendTagsFilter, getModelNameFilter } from '../utils/SearchUtils';
import {
  AntdTableSortOrder,
  REGISTERED_MODELS_PER_PAGE,
  REGISTERED_MODELS_SEARCH_NAME_FIELD,
} from '../constants';
import { searchRegisteredModelsApi, listModelStagesApi } from '../actions';
import LocalStorageUtils from '../../common/utils/LocalStorageUtils';

export class ModelListPageImpl extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      orderByKey: REGISTERED_MODELS_SEARCH_NAME_FIELD,
      orderByAsc: true,
      currentPage: 1,
      maxResultsSelection: REGISTERED_MODELS_PER_PAGE,
      pageTokens: {},
      loading: false,
    };
  }
  static propTypes = {
    models: PropTypes.arrayOf(PropTypes.object),
    searchRegisteredModelsApi: PropTypes.func.isRequired,
    listModelStagesApi: PropTypes.func.isRequired,
    stageTagComponents: PropTypes.object,
    modelStageNames: PropTypes.array,
    // react-router props
    history: PropTypes.object.isRequired,
    location: PropTypes.object,
  };
  modelListPageStoreKey = 'ModelListPageStore';
  defaultPersistedPageTokens = { 1: null };
  initialSearchRegisteredModelsApiId = getUUID();
  searchRegisteredModelsApiId = getUUID();
  criticalInitialRequestIds = [this.initialSearchRegisteredModelsApiId];

  getUrlState() {
    return this.props.location ? Utils.getSearchParamsFromUrl(this.props.location.search) : {};
  }

  componentDidMount() {
    this.props.listModelStagesApi();
    const urlState = this.getUrlState();
    const persistedPageTokens = this.getPersistedPageTokens();
    const maxResultsForTokens = this.getPersistedMaxResults();
    // eslint-disable-next-line react/no-did-mount-set-state
    this.setState(
      {
        orderByKey: urlState.orderByKey === undefined ? this.state.orderByKey : urlState.orderByKey,
        orderByAsc:
          urlState.orderByAsc === undefined
            ? this.state.orderByAsc
            : urlState.orderByAsc === 'true',
        currentPage:
          urlState.page !== undefined && urlState.page in persistedPageTokens
            ? parseInt(urlState.page, 10)
            : this.state.currentPage,
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

  setPersistedPageTokens(page_tokens) {
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
      return REGISTERED_MODELS_PER_PAGE;
    }
  }

  setMaxResultsInStore(max_results) {
    const store = ModelListPageImpl.getLocalStore(this.modelListPageStoreKey);
    store.setItem('max_results', max_results.toString());
  }

  /**
   * Returns a LocalStorageStore instance that can be used to persist data associated with the
   * ModelRegistry component.
   */
  static getLocalStore(key) {
    return LocalStorageUtils.getSessionScopedStoreForComponent('ModelListPage', key);
  }

  // Loads the initial set of models.
  loadModels(isInitialLoading = false) {
    const { orderByKey, orderByAsc } = this.state;
    const urlState = this.getUrlState();
    this.loadPage(
      this.state.currentPage,
      urlState.nameSearchInput,
      urlState.tagSearchInput,
      orderByKey,
      orderByAsc,
      undefined,
      undefined,
      isInitialLoading,
    );
  }

  resetHistoryState() {
    this.setState((prevState) => ({
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
  static getOrderByExpr = (orderByKey, orderByAsc) =>
    orderByKey ? `${orderByKey} ${orderByAsc ? 'ASC' : 'DESC'}` : '';

  isEmptyPageResponse = (value) => {
    return !value || !value.registered_models || !value.next_page_token;
  };

  getNextPageTokenFromResponse(response) {
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

  updatePageState = (page, response = {}) => {
    const nextPageToken = this.getNextPageTokenFromResponse(response);
    this.setState(
      (prevState) => ({
        currentPage: page,
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

  handleSearch = (nameSearchInput, tagSearchInput, callback, errorCallback) => {
    this.resetHistoryState();
    const { orderByKey, orderByAsc } = this.state;
    this.loadPage(
      1,
      nameSearchInput,
      tagSearchInput,
      orderByKey,
      orderByAsc,
      callback,
      errorCallback,
    );
  };

  handleClear = (callback, errorCallback) => {
    this.setState({
      orderByKey: REGISTERED_MODELS_SEARCH_NAME_FIELD,
      orderByAsc: true,
    });
    this.updateUrlWithSearchFilter('', '', REGISTERED_MODELS_SEARCH_NAME_FIELD, true, 1);
    this.loadPage(1, '', '', REGISTERED_MODELS_SEARCH_NAME_FIELD, true, callback, errorCallback);
  };

  updateUrlWithSearchFilter = (nameSearchInput, tagSearchInput, orderByKey, orderByAsc, page) => {
    const urlParams = {};
    if (nameSearchInput) {
      urlParams['nameSearchInput'] = nameSearchInput;
    }
    if (tagSearchInput) {
      urlParams['tagSearchInput'] = tagSearchInput;
    }
    if (orderByKey && orderByKey !== REGISTERED_MODELS_SEARCH_NAME_FIELD) {
      urlParams['orderByKey'] = orderByKey;
    }
    if (orderByAsc === false) {
      urlParams['orderByAsc'] = orderByAsc;
    }
    if (page && page !== 1) {
      urlParams['page'] = page;
    }
    const newUrl = `/models?${Utils.getSearchUrlFromState(urlParams)}`;
    if (newUrl !== this.props.history.location.pathname + this.props.history.location.search) {
      this.props.history.push(newUrl);
    }
  };

  handleMaxResultsChange = (key, callback, errorCallback) => {
    this.setState({ maxResultsSelection: parseInt(key, 10) }, () => {
      this.resetHistoryState();
      const urlState = this.getUrlState();
      const { orderByKey, orderByAsc, maxResultsSelection } = this.state;
      this.setMaxResultsInStore(maxResultsSelection);
      this.loadPage(
        1,
        urlState.nameSearchInput,
        urlState.tagSearchInput,
        orderByKey,
        orderByAsc,
        callback,
        errorCallback,
      );
    });
  };

  handleClickNext = (callback, errorCallback) => {
    const urlState = this.getUrlState();
    const { orderByKey, orderByAsc, currentPage } = this.state;
    this.loadPage(
      currentPage + 1,
      urlState.nameSearchInput,
      urlState.tagSearchInput,
      orderByKey,
      orderByAsc,
      callback,
      errorCallback,
    );
  };

  handleClickPrev = (callback, errorCallback) => {
    const urlState = this.getUrlState();
    const { orderByKey, orderByAsc, currentPage } = this.state;
    this.loadPage(
      currentPage - 1,
      urlState.nameSearchInput,
      urlState.tagSearchInput,
      orderByKey,
      orderByAsc,
      callback,
      errorCallback,
    );
  };

  handleClickSortableColumn = (orderByKey, sortOrder, callback, errorCallback) => {
    const orderByAsc = sortOrder !== AntdTableSortOrder.DESC; // default to true
    this.setState({ orderByKey, orderByAsc });
    this.resetHistoryState();
    const urlState = this.getUrlState();
    this.loadPage(
      1,
      urlState.nameSearchInput,
      urlState.tagSearchInput,
      orderByKey,
      orderByAsc,
      callback,
      errorCallback,
    );
  };

  getMaxResultsSelection = () => {
    return this.state.maxResultsSelection;
  };

  loadPage(
    page,
    nameSearchInput = '',
    tagSearchInput = '',
    orderByKey,
    orderByAsc,
    callback,
    errorCallback,
    isInitialLoading,
  ) {
    const { pageTokens } = this.state;
    this.setState({ loading: true });
    this.updateUrlWithSearchFilter(nameSearchInput, tagSearchInput, orderByKey, orderByAsc, page);
    this.props
      .searchRegisteredModelsApi(
        appendTagsFilter(getModelNameFilter(nameSearchInput), tagSearchInput),
        this.state.maxResultsSelection,
        ModelListPageImpl.getOrderByExpr(orderByKey, orderByAsc),
        pageTokens[page],
        isInitialLoading
          ? this.initialSearchRegisteredModelsApiId
          : this.searchRegisteredModelsApiId,
      )
      .then((r) => {
        this.updatePageState(page, r);
        this.setState({ loading: false });
        callback && callback();
      })
      .catch((e) => {
        Utils.logErrorAndNotifyUser(e);
        this.setState({ currentPage: 1 });
        this.resetHistoryState();
        errorCallback && errorCallback();
      });
  }

  render() {
    const { orderByKey, orderByAsc, currentPage, pageTokens } = this.state;
    const { models, stageTagComponents, modelStageNames } = this.props;
    const urlState = this.getUrlState();
    return (
      <RequestStateWrapper requestIds={[this.criticalInitialRequestIds]}>
        <ModelListView
          models={models}
          loading={this.state.loading}
          nameSearchInput={urlState.nameSearchInput}
          tagSearchInput={urlState.tagSearchInput}
          orderByKey={orderByKey}
          orderByAsc={orderByAsc}
          currentPage={currentPage}
          nextPageToken={pageTokens[currentPage + 1]}
          onSearch={this.handleSearch}
          onClear={this.handleClear}
          onClickNext={this.handleClickNext}
          onClickPrev={this.handleClickPrev}
          onClickSortableColumn={this.handleClickSortableColumn}
          onSetMaxResult={this.handleMaxResultsChange}
          getMaxResultValue={this.getMaxResultsSelection}
          stageTagComponents={stageTagComponents}
          modelStageNames={modelStageNames}
        />
      </RequestStateWrapper>
    );
  }
}

const mapStateToProps = (state) => {
  const models = Object.values(state.entities.modelByName);

  const stageTagComponents = state.entities.listModelStages["stageTagComponents"] || {}
  const modelStageNames = state.entities.listModelStages["modelStageNames"] || []
  return {
    models, stageTagComponents, modelStageNames
  };
};

const mapDispatchToProps = {
  searchRegisteredModelsApi,
  listModelStagesApi,
};

export const ModelListPage = connect(mapStateToProps, mapDispatchToProps)(ModelListPageImpl);
