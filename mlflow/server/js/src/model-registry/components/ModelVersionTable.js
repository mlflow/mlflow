import React from 'react';
import PropTypes from 'prop-types';
import { Table, Tooltip } from 'antd';
import { Link } from 'react-router-dom';
import { Typography } from '@databricks/design-system';
import Utils from '../../common/utils/Utils';
import { truncateToFirstLineWithMaxLength } from '../../common/utils/StringUtils';
import {
  ModelVersionStatus,
  ModelVersionStatusIcons,
  modelVersionStatusIconTooltips,
} from '../constants';
import { getModelVersionPageRoute } from '../routes';
import { RegisteringModelDocUrl } from '../../common/constants';
import { FormattedMessage, injectIntl } from 'react-intl';

const { Text } = Typography;

export class ModelVersionTableImpl extends React.Component {
  static propTypes = {
    modelName: PropTypes.string.isRequired,
    modelVersions: PropTypes.array.isRequired,
    activeStageOnly: PropTypes.bool,
    onChange: PropTypes.func.isRequired,
    allStagesAvailable: PropTypes.array.isRequired, // Nico
    stageTagComponents: PropTypes.object.isRequired,
    intl: PropTypes.any,
  };

  static defaultProps = {
    modelVersions: [],
    activeStageOnly: false,
  };

  getColumns = () => {
    const { modelName } = this.props;
    const columns = [
      {
        key: 'status',
        title: '', // Status column does not have title
        render: ({ status, status_message }) => (
          <Tooltip title={status_message || modelVersionStatusIconTooltips[status]}>
            <Text size='lg'>{ModelVersionStatusIcons[status]}</Text>
          </Tooltip>
        ),
        align: 'right',
        width: 40,
      },
      {
        title: this.props.intl.formatMessage({
          defaultMessage: 'Version',
          description: 'Column title text for model version in model version table',
        }),
        className: 'model-version',
        dataIndex: 'version',
        render: (version) => (
          <FormattedMessage
            defaultMessage='<link>Version {versionNumber}</link>'
            description='Link to model version in the model version table'
            values={{
              link: (chunks) => (
                <Link to={getModelVersionPageRoute(modelName, version)}>{chunks}</Link>
              ),
              versionNumber: version,
            }}
          />
        ),
      },
      {
        title: this.props.intl.formatMessage({
          defaultMessage: 'Registered at',
          description: 'Column title text for created at timestamp in model version table',
        }),
        dataIndex: 'creation_timestamp',
        render: (creationTimestamp) => <span>{Utils.formatTimestamp(creationTimestamp)}</span>,
        sorter: (a, b) => a.creation_timestamp - b.creation_timestamp,
        defaultSortOrder: 'descend',
        sortDirections: ['descend'],
      },
      {
        title: this.props.intl.formatMessage({
          defaultMessage: 'Created by',
          description: 'Column title text for creator username in model version table',
        }),
        dataIndex: 'user_id',
      },
      {
        title: this.props.intl.formatMessage({
          defaultMessage: 'Stage',
          description: 'Column title text for model version stage in model version table',
        }),
        dataIndex: 'current_stage',
        render: (currentStage) => {
          return this.props.stageTagComponents[currentStage];
        },
      },
      {
        title: this.props.intl.formatMessage({
          defaultMessage: 'Description',
          description: 'Column title text for description in model version table',
        }),
        dataIndex: 'description',
        render: (description) => truncateToFirstLineWithMaxLength(description, 32),
      },
    ];
    return columns;
  };

  getRowKey = (record) => record.creation_timestamp;

  emptyTablePlaceholder = () => {
    const learnMoreLinkUrl = ModelVersionTable.getLearnMoreLinkUrl();
    return (
      <span>
        <FormattedMessage
          defaultMessage='No models are registered yet. <link>Learn more</link> about how to
             register a model.'
          description='Message text when no model versions are registerd'
          values={{
            link: (chunks) => (
              // Reported during ESLint upgrade
              // eslint-disable-next-line react/jsx-no-target-blank
              <a target='_blank' href={learnMoreLinkUrl}>
                {chunks}
              </a>
            ),
          }}
        />
      </span>
    );
  };
  static getLearnMoreLinkUrl = () => RegisteringModelDocUrl;

  render() {
    const { modelVersions, activeStageOnly, allStagesAvailable } = this.props;
    const versions = activeStageOnly
      ? modelVersions.filter((v) => allStagesAvailable.includes(v.current_stage))
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
        pagination={{
          position: ['bottomRight'],
          size: 'default',
        }}
      />
    );
  }
}

export const ModelVersionTable = injectIntl(ModelVersionTableImpl);
