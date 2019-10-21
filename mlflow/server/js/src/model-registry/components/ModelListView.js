import React from 'react';
import _ from 'lodash';
import PropTypes from 'prop-types';
import { Table, Input } from 'antd';
import { Link } from 'react-router-dom';
import { getModelPageRoute, getModelVersionPageRoute } from '../routes';
import Utils from '../../utils/Utils';
import { Stages, StageTagComponents, EMPTY_CELL_PLACEHOLDER } from '../constants';

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

  getColumns = () => {
    return [
      {
        title: NAME_COLUMN,
        dataIndex: 'name',
        render: (text, row) => {
          return (
            <Link to={getModelPageRoute(row.name)}>
              {text}
            </Link>
          );
        },
        sorter: (a, b) => a.name.localeCompare(b.name),
        defaultSortOrder: 'ascend',
      },
      {
        title: LATEST_VERSION_COLUMN,
        render: ({ name, latest_versions }) => {
          const versionNumber = getOverallLatestVersionNumber(latest_versions);
          return versionNumber ? (
            <Link to={getModelVersionPageRoute(name, versionNumber)}>
              {`Version ${versionNumber}`}
            </Link>
          ) : EMPTY_CELL_PLACEHOLDER;
        }
      },
      {
        title: StageTagComponents[Stages.STAGING],
        render: ({ name, latest_versions }) => {
          const versionNumber = getLatestVersionNumberByStage(latest_versions, Stages.STAGING);
          return versionNumber ? (
            <Link to={getModelVersionPageRoute(name, versionNumber)}>
              {`Version ${versionNumber}`}
            </Link>
          ) : EMPTY_CELL_PLACEHOLDER;
        },
      },
      {
        title: StageTagComponents[Stages.PRODUCTION],
        render: ({ name, latest_versions }) => {
          const versionNumber = getLatestVersionNumberByStage(latest_versions, Stages.PRODUCTION);
          return versionNumber ? (
            <Link to={getModelVersionPageRoute(name, versionNumber)}>
              {`Version ${versionNumber}`}
            </Link>
          ) : EMPTY_CELL_PLACEHOLDER;
        },
      },
      {
        title: LAST_MODIFIED_COLUMN,
        dataIndex: 'last_updated_timestamp',
        render: (text, row) => (
          <span>{Utils.formatTimestamp(row.last_updated_timestamp)}</span>
        ),
        sorter: (a, b) => a.last_updated_timestamp - b.last_updated_timestamp,
      },
    ];
  };

  getRowKey = (record) => record.name;

  getFilteredModels() {
    const { models } = this.props;
    const { nameFilter } = this.state;
    return models.filter((model) =>
      model.registered_model.name.toLowerCase().includes(nameFilter.toLowerCase()),
    );
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
  }, 200);

  render() {
    const { nameFilter } = this.state;
    const sortedModels = this.getFilteredModels();
    const emptyText = `No model${nameFilter ? ' found' : ''}.`;
    return (
      <div>
        <div style={{ display: 'flex' }}>
          <h1>Registered Models</h1>
          <Search
            placeholder='search model name'
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
