import React from 'react';
import { ModelListView } from './ModelListView';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import { getUUID } from '../../common/utils/ActionUtils';
import Utils from '../../common/utils/Utils';
import { AntdTableSortOrder, REGISTERED_MODELS_SEARCH_NAME_FIELD } from '../constants';
import { searchRegisteredModelsApi } from '../actions';

class ModelListPage extends React.Component {
  static propTypes = {
    models: PropTypes.arrayOf(PropTypes.object),
    searchRegisteredModelsApi: PropTypes.func.isRequired,
  };

  state = {
    searchInput: '',
    orderByKey: REGISTERED_MODELS_SEARCH_NAME_FIELD,
    orderByAsc: true,
    currentPage: 1,
    pageTokens: { 1: null },
  };

  initialSearchRegisteredModelsApiId = getUUID();
  searchRegisteredModelsApiId = getUUID();
  criticalInitialRequestIds = [this.initialSearchRegisteredModelsApiId];

  componentDidMount() {
    this.loadModels(true);
  }

  // Loads the initial set of models.
  loadModels(isInitialLoading = false) {
    const { orderByKey, orderByAsc, searchInput } = this.state;
    this.loadPage(1, searchInput, orderByKey, orderByAsc, undefined, undefined, isInitialLoading);
  }

  resetHistoryState() {
    this.setState((prevState) => ({
      currentPage: 1,
      pageTokens: { 1: null },
    }));
  }

  static getModelNameFilter = (query) =>
    `${REGISTERED_MODELS_SEARCH_NAME_FIELD} ilike '%${query}%'`;

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

  updatePageState = (page, response = {}) => {
    const { value } = response;
    let nextPageToken;
    if (this.isEmptyPageResponse(value)) {
      // Why we could be here:
      // 1. There are no models returned: we went to the previous page but all models after that
      //    page's token has been deleted.
      // 2. If `next_page_token` is not returned, assume there is no next page.
      nextPageToken = null;
    } else {
      nextPageToken = value.next_page_token;
    }
    this.setState((prevState) => {
      return {
        currentPage: page,
        pageTokens: {
          ...prevState.pageTokens,
          [page + 1]: nextPageToken,
        },
      };
    });
  };

  handleSearch = (searchInput, callback, errorCallback) => {
    this.setState({ searchInput });
    this.resetHistoryState();
    const { orderByKey, orderByAsc } = this.state;
    this.loadPage(1, searchInput, orderByKey, orderByAsc, callback, errorCallback);
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

  loadPage(page, searchInput, orderByKey, orderByAsc, callback, errorCallback, isInitialLoading) {
    const { pageTokens } = this.state;
    this.props
      .searchRegisteredModelsApi(
        ModelListPage.getModelNameFilter(searchInput),
        ModelListPage.getOrderByExpr(orderByKey, orderByAsc),
        pageTokens[page],
        isInitialLoading
          ? this.initialSearchRegisteredModelsApiId
          : this.searchRegisteredModelsApiId,
      )
      .then((r) => {
        this.updatePageState(page, r);
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
    const { models } = this.props;
    return (
      <div className='App-content'>
        <RequestStateWrapper requestIds={[this.criticalInitialRequestIds]}>
          <ModelListView
            models={models}
            searchInput={searchInput}
            orderByKey={orderByKey}
            orderByAsc={orderByAsc}
            currentPage={currentPage}
            nextPageToken={pageTokens[currentPage + 1]}
            onSearch={this.handleSearch}
            onClickNext={this.handleClickNext}
            onClickPrev={this.handleClickPrev}
            onClickSortableColumn={this.handleClickSortableColumn}
          />
        </RequestStateWrapper>
      </div>
    );
  }
}

const mapStateToProps = (state) => {
  const models = Object.values(state.entities.modelByName);
  return { models };
};

const mapDispatchToProps = {
  searchRegisteredModelsApi,
};

export default connect(mapStateToProps, mapDispatchToProps)(ModelListPage);
