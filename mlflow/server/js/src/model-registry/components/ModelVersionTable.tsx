/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { Table, type TableColumnType } from 'antd';
import { Link } from '../../common/utils/RoutingUtils';
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
import { FormattedMessage, type IntlShape, injectIntl } from 'react-intl';
import type { ModelEntity, ModelVersionInfoEntity } from '../../experiment-tracking/types';
import { ModelVersionTableAliasesCell } from './aliases/ModelVersionTableAliasesCell';
import { useEditRegisteredModelAliasesModal } from '../hooks/useEditRegisteredModelAliasesModal';

const { Text } = Typography;

type ModelVersionTableImplProps = {
  modelName: string;
  modelVersions?: any[];
  activeStageOnly?: boolean;
  onChange: (...args: any[]) => any;
  intl: IntlShape;
  modelEntity?: ModelEntity;
  onAliasesModified: () => void;
  showEditAliasesModal?: (versionNumber: string) => void;
  usingNextModelsUI?: boolean;
};

export class ModelVersionTableImpl extends React.Component<ModelVersionTableImplProps> {
  static defaultProps = {
    modelVersions: [],
    activeStageOnly: false,
  };

  getColumns = () => {
    const { modelName, usingNextModelsUI } = this.props;

    const columns: TableColumnType<ModelVersionInfoEntity>[] = [
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
    ];

    if (usingNextModelsUI) {
      // Display aliases column only when "new models UI" is flipped
      columns.push({
        title: this.props.intl.formatMessage({
          defaultMessage: 'Aliases',
          description: 'Column title text for model version aliases in model version table',
        }),
        dataIndex: 'aliases',
        render: (aliases: string[], { version }: ModelVersionInfoEntity) => {
          return (
            <ModelVersionTableAliasesCell
              modelName={modelName}
              version={version}
              aliases={aliases}
              onAddEdit={() => {
                this.props.showEditAliasesModal?.(version);
              }}
            />
          );
        },
      });
    } else {
      // If not, display legacy "Stage" columns
      columns.push({
        title: this.props.intl.formatMessage({
          defaultMessage: 'Stage',
          description: 'Column title text for model version stage in model version table',
        }),
        dataIndex: 'current_stage',
        render: (currentStage: any) => {
          return StageTagComponents[currentStage];
        },
      });
    }
    columns.push(
      // Add remaining columns
      {
        title: this.props.intl.formatMessage({
          defaultMessage: 'Description',
          description: 'Column title text for description in model version table',
        }),
        dataIndex: 'description',
        render: (description: any) => truncateToFirstLineWithMaxLength(description, 32),
      },
    );
    return columns;
  };

  getRowKey = (record: any) => record.creation_timestamp;

  emptyTablePlaceholder = () => {
    const learnMoreLinkUrl = ModelVersionTableImpl.getLearnMoreLinkUrl();
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
    const { modelVersions = [], activeStageOnly } = this.props;
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
        pagination={{
          position: ['bottomRight'],
          size: 'default',
        }}
      />
    );
  }
}

const ModelVersionTableWithAliasEditor = (props: ModelVersionTableImplProps) => {
  const { EditAliasesModal, showEditAliasesModal } = useEditRegisteredModelAliasesModal({
    model: props.modelEntity || null,
    onSuccess: props.onAliasesModified,
  });
  return (
    <>
      <ModelVersionTableImpl {...props} showEditAliasesModal={showEditAliasesModal} />
      {EditAliasesModal}
    </>
  );
};
export const ModelVersionTable = injectIntl(ModelVersionTableWithAliasEditor);
