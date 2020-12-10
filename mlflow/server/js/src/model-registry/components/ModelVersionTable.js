import React from 'react';
import PropTypes from 'prop-types';
import { Table, Tooltip } from 'antd';
import { Link } from 'react-router-dom';
import Utils from '../../common/utils/Utils';
import { truncateToFirstLineWithMaxLength } from '../../common/utils/StringUtils';
import {
  ACTIVE_STAGES,
  StageTagComponents,
  ModelVersionStatus,
  ModelVersionStatusIcons,
  modelVersionStatusIconTooltips,
} from '../constants';
import { getModelVersionPageRoute } from '../routes';
import { RegisteringModelDocUrl } from '../../common/constants';

const VERSION_COLUMN = 'Version';
const CREATED_AT_COLUMN = 'Registered at';
const CREATED_BY_COLUMN = 'Created by';
const STAGE_COLUMN = 'Stage';
const DESCRIPTION_COLUMN = 'Description';

export class ModelVersionTable extends React.Component {
  static propTypes = {
    modelName: PropTypes.string.isRequired,
    modelVersions: PropTypes.array.isRequired,
    activeStageOnly: PropTypes.bool,
    onChange: PropTypes.func.isRequired,
  };

  static defaultProps = {
    modelVersions: [],
    activeStageOnly: false,
  };

  getColumns = () => {
    const { modelName } = this.props;
    return [
      {
        key: 'status',
        title: '', // Status column does not have title
        render: ({ status, status_message }) => (
          <Tooltip title={status_message || modelVersionStatusIconTooltips[status]}>
            {ModelVersionStatusIcons[status]}
          </Tooltip>
        ),
        align: 'right',
        width: 40,
      },
      {
        title: VERSION_COLUMN,
        className: 'model-version',
        dataIndex: 'version',
        render: (version) => (
          <Link to={getModelVersionPageRoute(modelName, version)}>{`Version ${version}`}</Link>
        ),
      },
      {
        title: CREATED_AT_COLUMN,
        dataIndex: 'creation_timestamp',
        render: (creationTimestamp) => <span>{Utils.formatTimestamp(creationTimestamp)}</span>,
        sorter: (a, b) => a.creation_timestamp - b.creation_timestamp,
        defaultSortOrder: 'descend',
        sortDirections: ['descend'],
      },
      {
        title: CREATED_BY_COLUMN,
        dataIndex: 'user_id',
      },
      {
        title: STAGE_COLUMN,
        dataIndex: 'current_stage',
        render: (currentStage) => {
          return StageTagComponents[currentStage];
        },
      },
      {
        title: DESCRIPTION_COLUMN,
        dataIndex: 'description',
        render: (description) => truncateToFirstLineWithMaxLength(description, 32),
      },
    ];
  };

  getRowKey = (record) => record.creation_timestamp;

  emptyTablePlaceholder = () => {
    const learnMoreLinkUrl = ModelVersionTable.getLearnMoreLinkUrl();
    return (
      <span>
        No models are registered yet.{' '}
        <a target='_blank' href={learnMoreLinkUrl}>
          Learn more
        </a>{' '}
        about how to register <br />a model.
      </span>
    );
  };
  static getLearnMoreLinkUrl = () => RegisteringModelDocUrl;

  render() {
    const { modelVersions, activeStageOnly } = this.props;
    const versions = activeStageOnly
      ? modelVersions.filter((v) => ACTIVE_STAGES.includes(v.current_stage))
      : modelVersions;
    return (
      <Table
        size='middle'
        rowKey={this.getRowKey}
        className='model-version-table'
        dataSource={versions}
        columns={this.getColumns()}
        locale={{ emptyText: this.emptyTablePlaceholder() }}
        rowSelection={{
          onChange: this.props.onChange,
          getCheckboxProps: (record) => ({
            disabled: record.status !== ModelVersionStatus.READY,
          }),
        }}
      />
    );
  }
}
