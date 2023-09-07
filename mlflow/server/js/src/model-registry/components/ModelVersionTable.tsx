/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { Table } from 'antd';
import { Link } from 'react-router-dom-v5-compat';
import { Tooltip, Typography } from '@databricks/design-system';
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
import { FormattedMessage, injectIntl } from 'react-intl';

const { Text } = Typography;

type OwnProps = {
  modelName: string;
  modelVersions: any[];
  activeStageOnly?: boolean;
  onChange: (...args: any[]) => any;
  intl?: any;
};

type Props = OwnProps & typeof ModelVersionTableImpl.defaultProps;

export class ModelVersionTableImpl extends React.Component<Props> {
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
        render: ({ status, status_message }: any) => (
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
        render: (version: any) => (
          <FormattedMessage
            defaultMessage='<link>Version {versionNumber}</link>'
            description='Link to model version in the model version table'
            values={{
              link: (chunks: any) => (
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
        render: (creationTimestamp: any) => <span>{Utils.formatTimestamp(creationTimestamp)}</span>,
        sorter: (a: any, b: any) => a.creation_timestamp - b.creation_timestamp,
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
        render: (currentStage: any) => {
          return StageTagComponents[currentStage];
        },
      },
      {
        title: this.props.intl.formatMessage({
          defaultMessage: 'Description',
          description: 'Column title text for description in model version table',
        }),
        dataIndex: 'description',
        render: (description: any) => truncateToFirstLineWithMaxLength(description, 32),
      },
    ];
    return columns;
  };

  getRowKey = (record: any) => record.creation_timestamp;

  emptyTablePlaceholder = () => {
    const learnMoreLinkUrl = (ModelVersionTable as any).getLearnMoreLinkUrl();
    return (
      <span>
        <FormattedMessage
          defaultMessage='No models are registered yet. <link>Learn more</link> about how to
             register a model.'
          description='Message text when no model versions are registerd'
          values={{
            link: (
              chunks: any, // Reported during ESLint upgrade
            ) => (
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
        // @ts-expect-error TS(2322): Type '({ key: string; title: string; render: ({ st... Remove this comment to see the full error message
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

// @ts-expect-error TS(2769): No overload matches this call.
export const ModelVersionTable = injectIntl(ModelVersionTableImpl);
