import React from 'react';
import { ModelListView } from './ModelListView';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import { getUUID } from '../../common/utils/ActionUtils';
import Utils from '../../common/utils/Utils';
import { getModelNameFilter } from '../utils/SearchUtils';
import {
  AntdTableSortOrder,
  REGISTERED_MODELS_PER_PAGE,
  REGISTERED_MODELS_SEARCH_NAME_FIELD,
} from '../constants';
import { searchRegisteredModelsApi } from '../actions';
import LocalStorageUtils from '../../common/utils/LocalStorageUtils';

class ModelListPage extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      searchInput: '',
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
    endpoints: PropTypes.object,
    searchRegisteredModelsApi: PropTypes.func.isRequired,
    // react-router props
    history: PropTypes.object.isRequired,
    location: PropTypes.object,
  };
  modelListPageStoreKey = 'ModelListPageStore';
  defaultPersistedPageTokens = { 1: null };
  initialSearchRegisteredModelsApiId = getUUID();
  searchRegisteredModelsApiId = getUUID();
  criticalInitialRequestIds = [this.initialSearchRegisteredModelsApiId];

  componentDidMount() {
    const urlState = this.props.location
      ? Utils.getSearchParamsFromUrl(this.props.location.search)
      : {};
    const persistedPageTokens = this.getPersistedPageTokens();
    const maxResultsForTokens = this.getPersistedMaxResults();
    // eslint-disable-next-line react/no-did-mount-set-state
    this.setState(
      {
        searchInput:
          urlState.searchInput === undefined ? this.state.searchInput : urlState.searchInput,
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
    const store = ModelListPage.getLocalStore(this.modelListPageStoreKey);
    if (store && store.getItem('page_tokens')) {
      return JSON.parse(store.getItem('page_tokens'));
    } else {
      return this.defaultPersistedPageTokens;
    }
  }

  setPersistedPageTokens(page_tokens) {
    const store = ModelListPage.getLocalStore(this.modelListPageStoreKey);
    if (store) {
      store.setItem('page_tokens', JSON.stringify(page_tokens));
    }
  }

  getPersistedMaxResults() {
    const store = ModelListPage.getLocalStore(this.modelListPageStoreKey);
    if (store && store.getItem('max_results')) {
      return parseInt(store.getItem('max_results'), 10);
    } else {
      return REGISTERED_MODELS_PER_PAGE;
    }
  }

  setMaxResultsInStore(max_results) {
    const store = ModelListPage.getLocalStore(this.modelListPageStoreKey);
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
    const { orderByKey, orderByAsc, searchInput } = this.state;
    this.loadPage(
      this.state.currentPage,
      searchInput,
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

  handleSearch = (searchInput, callback, errorCallback) => {
    this.setState({ searchInput });
    this.resetHistoryState();
    const { orderByKey, orderByAsc } = this.state;
    this.loadPage(1, searchInput, orderByKey, orderByAsc, callback, errorCallback);
  };

  updateUrlWithSearchFilter(searchInput, orderByKey, orderByAsc, page) {
    const state = {};
    if (searchInput) {
      state['searchInput'] = searchInput;
    }
    if (orderByKey && orderByKey !== REGISTERED_MODELS_SEARCH_NAME_FIELD) {
      state['orderByKey'] = orderByKey;
    }
    if (orderByAsc === false) {
      state['orderByAsc'] = orderByAsc;
    }
    if (page && page !== 1) {
      state['page'] = page;
    }
    const newUrl = `/models?${Utils.getSearchUrlFromState(state)}`;
    if (newUrl !== this.props.history.location.pathname + this.props.history.location.search) {
      this.props.history.push(newUrl);
    }
  }

  handleMaxResultsChange = (key, callback, errorCallback) => {
    this.setState({ maxResultsSelection: parseInt(key, 10) }, () => {
      this.resetHistoryState();
      const { searchInput, orderByKey, orderByAsc, maxResultsSelection } = this.state;
      this.setMaxResultsInStore(maxResultsSelection);
      this.loadPage(1, searchInput, orderByKey, orderByAsc, callback, errorCallback);
    });
  };

  handleClickNext = (callback, errorCallback) => {
    const { searchInput, orderByKey, orderByAsc, currentPage } = this.state;
    this.loadPage(currentPage + 1, searchInput, orderByKey, orderByAsc, callback, errorCallback);
  };

  handleClickPrev = (callback, errorCallback) => {
    const { searchInput, orderByKey, orderByAsc, currentPage } = this.state;
    this.loadPage(currentPage - 1, searchInput, orderByKey, orderByAsc, callback, errorCallback);
  };

  handleClickSortableColumn = (orderByKey, sortOrder, callback, errorCallback) => {
    const orderByAsc = sortOrder !== AntdTableSortOrder.DESC; // default to true
    this.setState({ orderByKey, orderByAsc });
    this.resetHistoryState();
    const { searchInput } = this.state;
    this.loadPage(1, searchInput, orderByKey, orderByAsc, callback, errorCallback);
  };

  getMaxResultsSelection = () => {
    return this.state.maxResultsSelection;
  };

  loadPage(page, searchInput, orderByKey, orderByAsc, callback, errorCallback, isInitialLoading) {
    const { pageTokens } = this.state;
    this.setState({ loading: true });
    this.updateUrlWithSearchFilter(searchInput, orderByKey, orderByAsc, page);
    this.props
      .searchRegisteredModelsApi(
        getModelNameFilter(searchInput),
        this.state.maxResultsSelection,
        ModelListPage.getOrderByExpr(orderByKey, orderByAsc),
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
    const { searchInput, orderByKey, orderByAsc, currentPage, pageTokens } = this.state;
    const { models, endpoints } = this.props;
    return (
      <div className='App-content'>
        <RequestStateWrapper requestIds={[this.criticalInitialRequestIds]}>
          <ModelListView
            models={models}
            loading={this.state.loading}
            endpoints={endpoints}
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
      </div>
    );
  }
}

const mapStateToProps = (state) => {
  const models = Object.values(state.entities.modelByName);
  return {
    models,
  };
};

const mapDispatchToProps = {
  searchRegisteredModelsApi,
};

export default connect(mapStateToProps, mapDispatchToProps)(ModelListPage);
