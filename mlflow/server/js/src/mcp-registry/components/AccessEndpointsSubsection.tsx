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
import type { MCPAccessEndpoint, MCPServer } from '../types';
import { formatTransportType, formatEndpointTarget, resolveEndpointDisplayName, STATUS_TAG_COLOR } from '../utils';
import { useServerState } from '../hooks/useServerState';
import { ellipsisStyles, noShrinkStyles } from '../styles';
import Utils from '../../common/utils/Utils';
import { ViewDetailsDrawer, DetailField } from './ViewDetailsDrawer';
import { ConnectionInstructions } from './ConnectionInstructions';
import { SubsectionHelpHeading } from './SubsectionHelpHeading';
import { ExpandableListSection } from './ExpandableListSection';

export const AccessEndpointsSubsection = ({
  endpoints,
  derivedName,
  server,
  onAddEndpoint,
  onEditEndpoint,
  onDeleteEndpoint,
}: {
  endpoints: MCPAccessEndpoint[];
  derivedName: string;
  server?: MCPServer;
  onAddEndpoint?: () => void;
  onEditEndpoint?: (endpoint: MCPAccessEndpoint) => void;
  onDeleteEndpoint?: (endpoint: MCPAccessEndpoint) => void;
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
        componentId="mlflow.mcp_registry.detail.access_endpoints_help"
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
          onAddEndpoint && canUpdate ? (
            <Button
              componentId="mlflow.mcp_registry.detail.add_endpoint"
              icon={<PlusIcon />}
              onClick={onAddEndpoint}
              css={{ marginLeft: 'auto' }}
            >
              <FormattedMessage defaultMessage="Add endpoint" description="Button to add an access endpoint" />
            </Button>
          ) : undefined
        }
      />
      {endpoints.length === 0 ? (
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="No access endpoints configured for this server."
            description="Empty state message for access endpoints subsection"
          />
        </Typography.Text>
      ) : (
        <ExpandableListSection
          items={endpoints}
          getKey={(b) => String(b.id)}
          getAriaLabel={(b, expanded) =>
            intl.formatMessage(
              {
                defaultMessage: '{action} endpoint {url}',
                description: 'Aria label for expanding/collapsing an access endpoint row',
              },
              { action: expanded ? 'Collapse' : 'Expand', url: b.url },
            )
          }
          renderRow={({ item: endpoint }) => (
            <AccessEndpointRowContent
              endpoint={endpoint}
              onEdit={onEditEndpoint && canUpdate ? () => onEditEndpoint(endpoint) : undefined}
              onDelete={onDeleteEndpoint && canDelete ? () => onDeleteEndpoint(endpoint) : undefined}
            />
          )}
          renderExpanded={(endpoint) => <AccessEndpointExpandedContent endpoint={endpoint} derivedName={derivedName} />}
        />
      )}
    </div>
  );
};

const AccessEndpointRowContent = ({
  endpoint,
  onEdit,
  onDelete,
}: {
  endpoint: MCPAccessEndpoint;
  onEdit?: () => void;
  onDelete?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const target = formatEndpointTarget(endpoint);

  return (
    <>
      <Tag componentId="mlflow.mcp_registry.detail.endpoint_transport_tag" color="indigo" css={noShrinkStyles}>
        {formatTransportType(endpoint.transport_type)}
      </Tag>
      <Typography.Text css={ellipsisStyles(theme)}>{endpoint.url}</Typography.Text>
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
              componentId="mlflow.mcp_registry.detail.endpoint.edit"
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
              componentId="mlflow.mcp_registry.detail.endpoint.delete"
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

const AccessEndpointExpandedContent = ({
  endpoint,
  derivedName,
}: {
  endpoint: MCPAccessEndpoint;
  derivedName: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const target = formatEndpointTarget(endpoint);

  return (
    <>
      <ConnectionInstructions
        source={ConnectionSource.ENDPOINT}
        endpoint={endpoint}
        derivedName={derivedName}
        detailLink={<EndpointDetailsDrawer endpoint={endpoint} />}
      />
      <div css={{ display: 'flex', gap: theme.spacing.lg }}>
        <Typography.Text size="sm">
          <Typography.Text bold size="sm">
            <FormattedMessage defaultMessage="Target:" description="Endpoint expanded target label" />
          </Typography.Text>{' '}
          {target}
        </Typography.Text>
        <Typography.Text size="sm">
          <Typography.Text bold size="sm">
            {endpoint.last_updated_timestamp !== endpoint.creation_timestamp ? (
              <FormattedMessage defaultMessage="Updated:" description="Endpoint expanded updated label" />
            ) : (
              <FormattedMessage defaultMessage="Created:" description="Endpoint expanded created label" />
            )}
          </Typography.Text>{' '}
          {endpoint.last_updated_timestamp ? Utils.formatTimestamp(endpoint.last_updated_timestamp, intl) : '—'}
        </Typography.Text>
      </div>
    </>
  );
};

const EndpointDetailsDrawer = ({ endpoint }: { endpoint: MCPAccessEndpoint }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const displayName = resolveEndpointDisplayName(endpoint);
  const target = endpoint.server_alias || endpoint.server_version || '—';
  const versionStatus = endpoint.resolved_version?.status;

  const hasBeenUpdated =
    endpoint.last_updated_timestamp !== endpoint.creation_timestamp || endpoint.last_updated_by !== endpoint.created_by;

  return (
    <ViewDetailsDrawer title={displayName}>
      <DetailField
        label={intl.formatMessage({ defaultMessage: 'Endpoint URL', description: 'Endpoint drawer endpoint label' })}
        value={endpoint.url}
        mono
        copyable
      />
      <DetailField
        label={intl.formatMessage({ defaultMessage: 'Transport', description: 'Endpoint drawer transport label' })}
        value={formatTransportType(endpoint.transport_type)}
        tagColor="indigo"
      />
      <DetailField
        label={intl.formatMessage({ defaultMessage: 'MCP server', description: 'Endpoint drawer server label' })}
        value={endpoint.server_name}
        mono
      />
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <Typography.Text bold size="sm">
          <FormattedMessage defaultMessage="Version target:" description="Endpoint drawer version target label" />
        </Typography.Text>
        <Typography.Text size="sm">{target}</Typography.Text>
        {versionStatus && (
          <Tag
            componentId="mlflow.mcp_registry.detail.endpoint_drawer.version_status"
            color={STATUS_TAG_COLOR[versionStatus]}
          >
            {versionStatus}
          </Tag>
        )}
      </div>
      {hasBeenUpdated && (
        <>
          <DetailField
            label={intl.formatMessage({
              defaultMessage: 'Last updated',
              description: 'Endpoint drawer last updated label',
            })}
            value={endpoint.last_updated_timestamp ? Utils.formatTimestamp(endpoint.last_updated_timestamp, intl) : '—'}
          />
          <DetailField
            label={intl.formatMessage({
              defaultMessage: 'Updated by',
              description: 'Endpoint drawer updated by label',
            })}
            value={endpoint.last_updated_by || '—'}
          />
        </>
      )}
      <DetailField
        label={intl.formatMessage({ defaultMessage: 'Created at', description: 'Endpoint drawer created at label' })}
        value={endpoint.creation_timestamp ? Utils.formatTimestamp(endpoint.creation_timestamp, intl) : '—'}
      />
      <DetailField
        label={intl.formatMessage({ defaultMessage: 'Created by', description: 'Endpoint drawer created by label' })}
        value={endpoint.created_by || '—'}
      />
    </ViewDetailsDrawer>
  );
};
