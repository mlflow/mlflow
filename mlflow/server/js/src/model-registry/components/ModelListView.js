import React from 'react';
import PropTypes from 'prop-types';
import { Table, Input, Alert } from 'antd';
import { Link } from 'react-router-dom';
import './ModelListView.css';
import { getModelPageRoute, getModelVersionPageRoute } from '../routes';
import Utils from '../../common/utils/Utils';
import {
  AntdTableSortOrder,
  Stages,
  StageTagComponents,
  EMPTY_CELL_PLACEHOLDER,
  REGISTERED_MODELS_PER_PAGE,
  REGISTERED_MODELS_SEARCH_NAME_FIELD,
  REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD,
} from '../constants';
import { ModelRegistryDocUrl, onboarding } from '../../common/constants';
import { SimplePagination } from './SimplePagination';
import { Spinner } from '../../common/components/Spinner';
import { CreateModelButton } from './CreateModelButton';
import LocalStorageUtils from '../../common/utils/LocalStorageUtils';
import { css } from 'emotion';
import { spacingMedium } from '../../common/styles/spacing';

const NAME_COLUMN = 'Name';
const NAME_COLUMN_INDEX = 'name';
const LATEST_VERSION_COLUMN = 'Latest Version';
const LAST_MODIFIED_COLUMN = 'Last Modified';
const LAST_MODIFIED_COLUMN_INDEX = 'last_updated_timestamp';

const getOverallLatestVersionNumber = (latest_versions) =>
  latest_versions && Math.max(...latest_versions.map((v) => v.version));

const getLatestVersionNumberByStage = (latest_versions, stage) => {
  const modelVersion = latest_versions && latest_versions.find((v) => v.current_stage === stage);
  return modelVersion && modelVersion.version;
};

const { Search } = Input;

export class ModelListView extends React.Component {
  static propTypes = {
    models: PropTypes.array.isRequired,
    searchInput: PropTypes.string.isRequired,
    orderByKey: PropTypes.string.isRequired,
    orderByAsc: PropTypes.bool.isRequired,
    currentPage: PropTypes.number.isRequired,
    // To know if there is a next page. If null, there is no next page. If undefined, we haven't
    // gotten an answer from the backend yet.
    nextPageToken: PropTypes.string,
    loading: PropTypes.bool,
    onSearch: PropTypes.func.isRequired,
    onClickNext: PropTypes.func.isRequired,
    onClickPrev: PropTypes.func.isRequired,
    onClickSortableColumn: PropTypes.func.isRequired,
    onSetMaxResult: PropTypes.func.isRequired,
    getMaxResultValue: PropTypes.func.isRequired,
  };

  static defaultProps = {
    models: [],
  };

  state = {
    loading: false,
    lastNavigationActionWasClickPrev: false,
    maxResultsSelection: REGISTERED_MODELS_PER_PAGE,
    showOnboardingHelper: this.showOnboardingHelper(),
  };

  showOnboardingHelper() {
    const onboardingInformationStore = ModelListView.getLocalStore(onboarding);
    return onboardingInformationStore.getItem('showRegistryHelper') === null;
  }

  disableOnboardingHelper() {
    const onboardingInformationStore = ModelListView.getLocalStore(onboarding);
    onboardingInformationStore.setItem('showRegistryHelper', 'false');
  }

  /**
   * Returns a LocalStorageStore instance that can be used to persist data associated with the
   * ModelRegistry component.
   */
  static getLocalStore(key) {
    return LocalStorageUtils.getStoreForComponent('ModelListView', key);
  }

  componentDidMount() {
    const pageTitle = 'MLflow Models';
    Utils.updatePageTitle(pageTitle);
  }

  getSortOrder = (key) => {
    const { orderByKey, orderByAsc } = this.props;
    if (key !== orderByKey) {
      return null;
    }
    return { sortOrder: orderByAsc ? AntdTableSortOrder.ASC : AntdTableSortOrder.DESC };
  };

  getColumns = () => {
    const columns = [
      {
        title: NAME_COLUMN,
        className: 'model-name',
        dataIndex: NAME_COLUMN_INDEX,
        render: (text, row) => {
          return <Link to={getModelPageRoute(row.name)}>{text}</Link>;
        },
        sorter: true,
        ...this.getSortOrder(REGISTERED_MODELS_SEARCH_NAME_FIELD),
      },
      {
        title: LATEST_VERSION_COLUMN,
        className: 'latest-version',
        render: ({ name, latest_versions }) => {
          const versionNumber = getOverallLatestVersionNumber(latest_versions);
          return versionNumber ? (
            <Link to={getModelVersionPageRoute(name, versionNumber)}>
              {`Version ${versionNumber}`}
            </Link>
          ) : (
            EMPTY_CELL_PLACEHOLDER
          );
        },
      },
      {
        title: StageTagComponents[Stages.STAGING],
        className: 'latest-staging',
        render: ({ name, latest_versions }) => {
          const versionNumber = getLatestVersionNumberByStage(latest_versions, Stages.STAGING);
          return versionNumber ? (
            <Link to={getModelVersionPageRoute(name, versionNumber)}>
              {`Version ${versionNumber}`}
            </Link>
          ) : (
            EMPTY_CELL_PLACEHOLDER
          );
        },
      },
      {
        title: StageTagComponents[Stages.PRODUCTION],
        className: 'latest-production',
        render: ({ name, latest_versions }) => {
          const versionNumber = getLatestVersionNumberByStage(latest_versions, Stages.PRODUCTION);
          return versionNumber ? (
            <Link to={getModelVersionPageRoute(name, versionNumber)}>
              {`Version ${versionNumber}`}
            </Link>
          ) : (
            EMPTY_CELL_PLACEHOLDER
          );
        },
      },
      {
        title: LAST_MODIFIED_COLUMN,
        className: 'last-modified',
        dataIndex: LAST_MODIFIED_COLUMN_INDEX,
        render: (text, row) => <span>{Utils.formatTimestamp(row.last_updated_timestamp)}</span>,
        sorter: true,
        ...this.getSortOrder(REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD),
      },
    ];
    return columns;
  };

  getRowKey = (record) => record.name;

  setLoadingFalse = () => {
    this.setState({ loading: false });
  };

  handleSearch = (value) => {
    this.setState({ loading: true, lastNavigationActionWasClickPrev: false });
    this.props.onSearch(value, this.setLoadingFalse, this.setLoadingFalse);
  };

  static getSortFieldName = (column) => {
    switch (column) {
      case NAME_COLUMN_INDEX:
        return REGISTERED_MODELS_SEARCH_NAME_FIELD;
      case LAST_MODIFIED_COLUMN_INDEX:
        return REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD;
      default:
        return null;
    }
  };

  handleTableChange = (pagination, filters, sorter) => {
    this.setState({ loading: true, lastNavigationActionWasClickPrev: false });
    this.props.onClickSortableColumn(
      ModelListView.getSortFieldName(sorter.field),
      sorter.order,
      this.setLoadingFalse,
      this.setLoadingFalse,
    );
  };

  renderOnboardingContent() {
    const learnMoreLinkUrl = ModelListView.getLearnMoreLinkUrl();
    const content = (
      <div>
        Share and serve machine learning models.{' '}
        <a href={learnMoreLinkUrl} target='_blank' rel='noopener noreferrer' className='LinkColor'>
          Learn more
        </a>
      </div>
    );

    return this.state.showOnboardingHelper ? (
      <Alert
        className='onboarding-information'
        description={content}
        type='info'
        showIcon
        closable
        onClose={() => this.disableOnboardingHelper()}
      />
    ) : null;
  }

  getEmptyTextComponent() {
    const { searchInput } = this.props;
    const { lastNavigationActionWasClickPrev } = this.state;
    // Handle the case when emptiness is caused by search filter
    if (searchInput) {
      if (lastNavigationActionWasClickPrev) {
        return (
          'No models found for the page. Please refresh the page as the underlying data may ' +
          'have changed significantly.'
        );
      } else {
        return 'No models found.';
      }
    }
    return (
      <div>
        <span>
          No models yet. <CreateModelButton buttonType='link' buttonText='Create a model' /> to get
          started.
        </span>
      </div>
    );
  }

  static getLearnMoreLinkUrl = () => ModelRegistryDocUrl;

  handleClickNext = () => {
    this.setState({ loading: true, lastNavigationActionWasClickPrev: false });
    this.props.onClickNext(this.setLoadingFalse, this.setLoadingFalse);
  };

  handleClickPrev = () => {
    this.setState({ loading: true, lastNavigationActionWasClickPrev: true });
    this.props.onClickPrev(this.setLoadingFalse, this.setLoadingFalse);
  };

  handleSetMaxResult = ({ item, key, keyPath, domEvent }) => {
    this.setState({ loading: true });
    this.props.onSetMaxResult(key, this.setLoadingFalse, this.setLoadingFalse);
  };

  render() {
    const { models, searchInput, currentPage, nextPageToken } = this.props;
    const { loading } = this.state;

    return (
      <div className='model-list-view'>
        <h1 className={`breadcrumb-header ${classNames.wrapper}`}>Registered Models</h1>
        {this.renderOnboardingContent()}
        <div style={{ display: 'flex' }}>
          <CreateModelButton />
          <Search
            className='model-list-search'
            aria-label='search model name'
            placeholder='search model name'
            defaultValue={searchInput}
            onSearch={this.handleSearch}
            style={{ width: 210, height: 32, marginLeft: 'auto' }}
            enterButton
            allowClear
          />
        </div>
        <Table
          size='middle'
          rowKey={this.getRowKey}
          className='model-version-table'
          dataSource={models}
          columns={this.getColumns()}
          locale={{ emptyText: this.getEmptyTextComponent() }}
          pagination={{
            hideOnSinglePage: true,
            pageSize: this.props.getMaxResultValue(),
          }}
          loading={loading && { indicator: <Spinner /> }}
          onChange={this.handleTableChange}
        />
        <div>
          <SimplePagination
            currentPage={currentPage}
            loading={this.props.loading}
            isLastPage={nextPageToken === null}
            onClickNext={this.handleClickNext}
            onClickPrev={this.handleClickPrev}
            handleSetMaxResult={this.handleSetMaxResult}
            maxResultOptions={[REGISTERED_MODELS_PER_PAGE, 25, 50, 100]}
            getSelectedPerPageSelection={this.props.getMaxResultValue}
          />
        </div>
      </div>
    );
  }
}

const classNames = {
  wrapper: css({
    marginTop: spacingMedium,
  }),
};
