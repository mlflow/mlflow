import React from 'react';
import PropTypes from 'prop-types';
import { Table, Input } from 'antd';
import { Link } from 'react-router-dom';
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
import { ModelRegistryDocUrl } from '../../common/constants';
import { SimplePagination } from './SimplePagination';
import { Spinner } from '../../common/components/Spinner';

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
    onSearch: PropTypes.func.isRequired,
    onClickNext: PropTypes.func.isRequired,
    onClickPrev: PropTypes.func.isRequired,
    onClickSortableColumn: PropTypes.func.isRequired,
  };

  static defaultProps = {
    models: [],
  };

  state = {
    loading: false,
    lastNavigationActionWasClickPrev: false,
  };

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
    return [
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
        dataIndex: LAST_MODIFIED_COLUMN_INDEX,
        render: (text, row) => <span>{Utils.formatTimestamp(row.last_updated_timestamp)}</span>,
        sorter: true,
        ...this.getSortOrder(REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD),
      },
    ];
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
    // Handle the case when emptiness is caused by no registered model
    const learnMoreLinkUrl = ModelListView.getLearnMoreLinkUrl();
    return (
      <div>
        <div>No models yet.</div>
        <div>
          MLflow Model Registry is a centralized model store that enables you to manage the full
          lifecycle of MLflow Models.{' '}
          <a target='_blank' href={learnMoreLinkUrl}>
            Learn more
          </a>
          {'.'}
        </div>
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

  render() {
    const { models, searchInput, currentPage, nextPageToken } = this.props;
    const { loading } = this.state;
    const emptyText = this.getEmptyTextComponent();

    return (
      <div>
        <div style={{ display: 'flex' }}>
          <h1>Registered Models</h1>
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
          locale={{ emptyText }}
          pagination={{ hideOnSinglePage: true, defaultPageSize: REGISTERED_MODELS_PER_PAGE }}
          loading={loading && { indicator: <Spinner /> }}
          onChange={this.handleTableChange}
        />
        <SimplePagination
          currentPage={currentPage}
          isLastPage={nextPageToken === null}
          onClickNext={this.handleClickNext}
          onClickPrev={this.handleClickPrev}
        />
      </div>
    );
  }
}
