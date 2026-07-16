import {
  Button,
  PencilIcon,
  PlusIcon,
  Tag,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { ConnectionSource } from '../types';
import type { MCPAccessBinding, MCPServer } from '../types';
import {
  formatTransportType,
  formatBindingTarget,
  resolveBindingDisplayName,
  STATUS_TAG_COLOR,
} from '../utils';
import { useServerState } from '../hooks/useServerState';
import { ellipsisStyles, noShrinkStyles } from '../styles';
import Utils from '../../common/utils/Utils';
import { ViewDetailsDrawer, DetailField } from './ViewDetailsDrawer';
import { ConnectionInstructions } from './ConnectionInstructions';
import { SubsectionHelpHeading } from './SubsectionHelpHeading';
import { ExpandableListSection } from './ExpandableListSection';

export const AccessBindingsSubsection = ({
  bindings,
  derivedName,
  server,
  onAddBinding,
  onEditBinding,
  onDeleteBinding,
}: {
  bindings: MCPAccessBinding[];
  derivedName: string;
  server?: MCPServer;
  onAddBinding?: () => void;
  onEditBinding?: (binding: MCPAccessBinding) => void;
  onDeleteBinding?: (binding: MCPAccessBinding) => void;
}) => {
  const { canUpdate, canDelete, canManage } = useServerState(server);
  const intl = useIntl();

  return (
    <div>
      <SubsectionHelpHeading
        title={
          <FormattedMessage
            defaultMessage="Access endpoints"
            description="MCP server access endpoints subsection heading"
          />
        }
        componentId="mlflow.mcp_registry.detail.access_bindings_help"
        helpAriaLabel={intl.formatMessage({
          defaultMessage: 'About access endpoints',
          description: 'Aria label for access endpoints subsection help popover',
        })}
        helpText={
          <FormattedMessage
            defaultMessage="Live connections between this registered server and deployed endpoints."
            description="Help text for MCP server access endpoints subsection"
          />
        }
        actions={
          onAddBinding && canUpdate ? (
            <Button
              componentId="mlflow.mcp_registry.detail.add_binding"
              icon={<PlusIcon />}
              onClick={onAddBinding}
              css={{ marginLeft: 'auto' }}
            >
              <FormattedMessage defaultMessage="Add endpoint" description="Button to add an access endpoint" />
            </Button>
          ) : undefined
        }
      />
      {bindings.length === 0 ? (
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="No access endpoints configured for this server."
            description="Empty state message for access endpoints subsection"
          />
        </Typography.Text>
      ) : (
        <ExpandableListSection
          items={bindings}
          getKey={(b) => String(b.binding_id)}
          getAriaLabel={(b, expanded) =>
            intl.formatMessage(
              {
                defaultMessage: '{action} binding {url}',
                description: 'Aria label for expanding/collapsing an access endpoint row',
              },
              { action: expanded ? 'Collapse' : 'Expand', url: b.endpoint_url },
            )
          }
          renderRow={({ item: binding }) => (
            <AccessBindingRowContent
              binding={binding}
              onEdit={onEditBinding && canManage ? () => onEditBinding(binding) : undefined}
              onDelete={onDeleteBinding && canManage ? () => onDeleteBinding(binding) : undefined}
            />
          )}
          renderExpanded={(binding) => (
            <AccessBindingExpandedContent binding={binding} derivedName={derivedName} />
          )}
        />
      )}
    </div>
  );
};

const AccessBindingRowContent = ({
  binding,
  onEdit,
  onDelete,
}: {
  binding: MCPAccessBinding;
  onEdit?: () => void;
  onDelete?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const target = formatBindingTarget(binding);

  return (
    <>
      <Tag componentId="mlflow.mcp_registry.detail.binding_transport_tag" color="indigo" css={noShrinkStyles}>
        {formatTransportType(binding.transport_type)}
      </Tag>
      <Typography.Text css={ellipsisStyles(theme)}>{binding.endpoint_url}</Typography.Text>
      <Typography.Text color="secondary" size="sm" css={noShrinkStyles}>
        {target}
      </Typography.Text>
      {(onEdit || onDelete) && (
        <div
          css={{ ...noShrinkStyles, display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}
          onClick={(e) => e.stopPropagation()}
        >
          {onEdit && (
            <Button
              componentId="mlflow.mcp_registry.detail.binding.edit"
              type="tertiary"
              size="small"
              icon={<PencilIcon />}
              onClick={onEdit}
              aria-label={intl.formatMessage({
                defaultMessage: 'Edit access endpoint',
                description: 'Aria label for edit access endpoint button',
              })}
            />
          )}
          {onDelete && (
            <Button
              componentId="mlflow.mcp_registry.detail.binding.delete"
              type="tertiary"
              size="small"
              icon={<TrashIcon />}
              danger
              onClick={onDelete}
              aria-label={intl.formatMessage({
                defaultMessage: 'Delete access endpoint',
                description: 'Aria label for delete access endpoint button',
              })}
            />
          )}
        </div>
      )}
    </>
  );
};

const AccessBindingExpandedContent = ({
  binding,
  derivedName,
}: {
  binding: MCPAccessBinding;
  derivedName: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const target = formatBindingTarget(binding);

  return (
    <>
      <ConnectionInstructions
        source={ConnectionSource.BINDING}
        binding={binding}
        derivedName={derivedName}
        detailLink={<BindingDetailsDrawer binding={binding} />}
      />
      <div css={{ display: 'flex', gap: theme.spacing.lg }}>
        <Typography.Text size="sm">
          <Typography.Text bold size="sm">
            <FormattedMessage defaultMessage="Target:" description="Binding expanded target label" />
          </Typography.Text>{' '}
          {target}
        </Typography.Text>
        <Typography.Text size="sm">
          <Typography.Text bold size="sm">
            <FormattedMessage defaultMessage="Updated:" description="Binding expanded updated label" />
          </Typography.Text>{' '}
          {binding.last_updated_timestamp ? Utils.formatTimestamp(binding.last_updated_timestamp, intl) : '—'}
        </Typography.Text>
      </div>
    </>
  );
};

const BindingDetailsDrawer = ({ binding }: { binding: MCPAccessBinding }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const displayName = resolveBindingDisplayName(binding);
  const target = binding.server_alias || binding.server_version || '—';
  const description = binding.resolved_version?.server_json?.description || '—';
  const versionStatus = binding.resolved_version?.status;

  return (
    <ViewDetailsDrawer title={displayName}>
      <DetailField
        label={intl.formatMessage({ defaultMessage: 'Description', description: 'Binding drawer description label' })}
        value={description}
      />
      <DetailField
        label={intl.formatMessage({ defaultMessage: 'Endpoint URL', description: 'Binding drawer endpoint label' })}
        value={binding.endpoint_url}
        mono
      />
      <DetailField
        label={intl.formatMessage({ defaultMessage: 'Transport', description: 'Binding drawer transport label' })}
        value={formatTransportType(binding.transport_type)}
        tagColor="indigo"
      />
      <DetailField
        label={intl.formatMessage({ defaultMessage: 'MCP server', description: 'Binding drawer server label' })}
        value={binding.server_name}
        mono
      />
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <Typography.Text bold size="sm">
          <FormattedMessage defaultMessage="Version target:" description="Binding drawer version target label" />
        </Typography.Text>
        <Typography.Text size="sm">{target}</Typography.Text>
        {versionStatus && (
          <Tag
            componentId="mlflow.mcp_registry.detail.binding_drawer.version_status"
            color={STATUS_TAG_COLOR[versionStatus]}
          >
            {versionStatus}
          </Tag>
        )}
      </div>
      <DetailField
        label={intl.formatMessage({ defaultMessage: 'Last updated', description: 'Binding drawer last updated label' })}
        value={binding.last_updated_timestamp ? Utils.formatTimestamp(binding.last_updated_timestamp, intl) : '—'}
      />
      <DetailField
        label={intl.formatMessage({ defaultMessage: 'Updated by', description: 'Binding drawer updated by label' })}
        value={binding.last_updated_by || '—'}
      />
      <DetailField
        label={intl.formatMessage({ defaultMessage: 'Created at', description: 'Binding drawer created at label' })}
        value={binding.creation_timestamp ? Utils.formatTimestamp(binding.creation_timestamp, intl) : '—'}
      />
      <DetailField
        label={intl.formatMessage({ defaultMessage: 'Created by', description: 'Binding drawer created by label' })}
        value={binding.created_by || '—'}
      />
    </ViewDetailsDrawer>
  );
};
