import React from 'react';
import _ from 'lodash';
import PropTypes from 'prop-types';
import { Table, Input } from 'antd';
import { Link } from 'react-router-dom';
import { getModelPageRoute, getModelVersionPageRoute } from '../routes';
import Utils from '../../common/utils/Utils';
import { Stages, StageTagComponents, EMPTY_CELL_PLACEHOLDER } from '../constants';
import { ModelRegistryDocUrl } from '../../common/constants';

const NAME_COLUMN = 'Name';
const LATEST_VERSION_COLUMN = 'Latest Version';
const LAST_MODIFIED_COLUMN = 'Last Modified';

const getOverallLatestVersionNumber = (latest_versions) =>
  latest_versions && Math.max(...latest_versions.map((v) => v.version));

const getLatestVersionNumberByStage = (latest_versions, stage) => {
  const modelVersion = latest_versions && latest_versions.find((v) => v.current_stage === stage);
  return modelVersion && modelVersion.version;
};

const { Search } = Input;
export const SEARCH_DEBOUNCE_INTERVAL = 200;

export class ModelListView extends React.Component {
  static propTypes = {
    models: PropTypes.array.isRequired,
  };

  static defaultProps = {
    models: [],
  };

  state = {
    nameFilter: '',
  };

  componentDidMount() {
    const pageTitle = 'MLflow Models';
    Utils.updatePageTitle(pageTitle);
  }

  getColumns = () => {
    return [
      {
        title: NAME_COLUMN,
        className: 'model-name',
        dataIndex: 'name',
        render: (text, row) => {
          return <Link to={getModelPageRoute(row.name)}>{text}</Link>;
        },
        sorter: (a, b) => a.name.localeCompare(b.name),
        defaultSortOrder: 'ascend',
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
        dataIndex: 'last_updated_timestamp',
        render: (text, row) => <span>{Utils.formatTimestamp(row.last_updated_timestamp)}</span>,
        sorter: (a, b) => a.last_updated_timestamp - b.last_updated_timestamp,
      },
    ];
  };

  getRowKey = (record) => record.name;

  getFilteredModels() {
    const { models } = this.props;
    const { nameFilter } = this.state;
    return models.filter((model) => model.name.toLowerCase().includes(nameFilter.toLowerCase()));
  }

  handleSearchByName = (e) => {
    // SyntheticEvent is pooled in React, to access the event properties in an asynchronous way like
    // debounce & throttling, we need to call event.persist() on the event.
    // https://reactjs.org/docs/events.html#event-pooling
    e.persist();
    this.emitNameFilterChangeDebounced(e);
  };

  emitNameFilterChangeDebounced = _.debounce((e) => {
    this.setState({ nameFilter: e.target.value });
  }, SEARCH_DEBOUNCE_INTERVAL);

  static getEmptyTextComponent(nameFilter) {
    // Handle the case when emptiness is caused by search filter
    if (nameFilter) {
      return 'No models found.';
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

  render() {
    const { nameFilter } = this.state;
    const sortedModels = this.getFilteredModels();
    const emptyText = ModelListView.getEmptyTextComponent(nameFilter);
    return (
      <div>
        <div style={{ display: 'flex' }}>
          <h1>Registered Models</h1>
          <Search
            aria-label='search model name'
            placeholder='Search Model Name'
            onChange={this.handleSearchByName}
            style={{ width: 200, height: 32, marginLeft: 'auto' }}
          />
        </div>
        <Table
          size='middle'
          rowKey={this.getRowKey}
          className='model-version-table'
          dataSource={sortedModels}
          columns={this.getColumns()}
          locale={{ emptyText }}
        />
      </div>
    );
  }
}
