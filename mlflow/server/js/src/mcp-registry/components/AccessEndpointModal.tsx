import type React from 'react';
import { useEffect, useState } from 'react';
import {
  Alert,
  Input,
  Modal,
  SimpleSelect,
  SimpleSelectOption,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { MCPAccessEndpoint, MCPRemoteTransportType } from '../types';
import { TransportType } from '../types';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useMCPServerQuery, useMCPServerVersionsQuery } from '../hooks/useMCPServerDetailQuery';
import { MCPRegistryApi } from '../api';
import { MCP_QUERY_KEYS } from '../utils';
import { flexColumnGapStyles } from '../styles';
import { useCreateAccessEndpointMutation, useUpdateAccessEndpointMutation } from '../hooks/useAccessEndpointMutation';
import { isValidEndpointUrl, resolveEndpointDisplayName } from '../utils';
import { FieldLabel } from '../../admin/components/FieldLabel';
import { EndpointTargetSelector, ALIAS_PREFIX, VERSION_PREFIX } from './EndpointTargetSelector';

function endpointToTarget(endpoint: MCPAccessEndpoint): string {
  if (endpoint.server_alias) return `${ALIAS_PREFIX}${endpoint.server_alias}`;
  if (endpoint.server_version) return `${VERSION_PREFIX}${endpoint.server_version}`;
  return `${ALIAS_PREFIX}latest`;
}

export const AccessEndpointModal = ({
  visible,
  onCancel,
  onSuccess,
  editEndpoint,
  lockedServer,
  scopedVersion,
  scopedAliases,
  createTitle,
}: {
  visible: boolean;
  onCancel: () => void;
  onSuccess?: () => void;
  editEndpoint?: MCPAccessEndpoint;
  lockedServer?: string;
  scopedVersion?: string;
  scopedAliases?: string[];
  createTitle?: React.ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isEditMode = Boolean(editEndpoint);
  const isServerLocked = isEditMode || Boolean(lockedServer);

  const [selectedServer, setSelectedServer] = useState('');
  const [endpointUrl, setEndpointUrl] = useState('');
  const [selectedTarget, setSelectedTarget] = useState(`${ALIAS_PREFIX}latest`);
  const [transportType, setTransportType] = useState<MCPRemoteTransportType>(TransportType.STREAMABLE_HTTP);

  const createMutation = useCreateAccessEndpointMutation();
  const updateMutation = useUpdateAccessEndpointMutation();
  const activeMutation = isEditMode ? updateMutation : createMutation;

  // TODO: Server list is capped at 1000 results. Switch to a searchable combobox
  // with scroll-based pagination for registries with more servers.
  const { data: allServersResponse } = useQuery({
    queryKey: [MCP_QUERY_KEYS.SERVERS_LIST, 'all'],
    queryFn: () => MCPRegistryApi.searchMCPServers({ max_results: 1000 }),
    enabled: !isServerLocked,
  });
  const servers = allServersResponse?.mcp_servers;
  const { data: server } = useMCPServerQuery(selectedServer);
  const { data: versions } = useMCPServerVersionsQuery(selectedServer);

  useEffect(() => {
    if (visible) {
      if (editEndpoint) {
        setSelectedServer(editEndpoint.server_name);
        setEndpointUrl(editEndpoint.url);
        setSelectedTarget(endpointToTarget(editEndpoint));
        setTransportType(editEndpoint.transport_type);
      } else {
        setSelectedServer(lockedServer || '');
        setEndpointUrl('');
        setSelectedTarget(scopedVersion ? `${VERSION_PREFIX}${scopedVersion}` : `${ALIAS_PREFIX}latest`);
        setTransportType(TransportType.STREAMABLE_HTTP);
      }
      createMutation.reset();
      updateMutation.reset();
    }
  }, [visible, editEndpoint, lockedServer, scopedVersion]); // eslint-disable-line react-hooks/exhaustive-deps -- reset() creates new ref

  const aliases = server?.aliases ?? [];
  const isSubmitting = activeMutation.isLoading;

  const isValidUrl = isValidEndpointUrl(endpointUrl);
  const isFormValid = Boolean(selectedServer && isValidUrl && selectedTarget);

  const handleSubmit = () => {
    if (!isFormValid) return;
    const isAlias = selectedTarget.startsWith(ALIAS_PREFIX);
    const targetValue = isAlias
      ? selectedTarget.slice(ALIAS_PREFIX.length)
      : selectedTarget.slice(VERSION_PREFIX.length);
    const onMutationSuccess = {
      onSuccess: () => {
        onCancel();
        onSuccess?.();
      },
    };

    if (isEditMode && editEndpoint) {
      updateMutation.mutate(
        {
          serverName: editEndpoint.server_name,
          endpointId: editEndpoint.id,
          request: {
            url: endpointUrl.trim(),
            server_alias: isAlias ? targetValue : null,
            server_version: isAlias ? null : targetValue,
            transport_type: transportType,
          },
        },
        onMutationSuccess,
      );
    } else {
      createMutation.mutate(
        {
          serverName: selectedServer,
          request: {
            url: endpointUrl.trim(),
            server_alias: isAlias ? targetValue : undefined,
            server_version: isAlias ? undefined : targetValue,
            transport_type: transportType,
          },
        },
        onMutationSuccess,
      );
    }
  };

  return (
    <Modal
      componentId="mlflow.mcp_registry.endpoint_modal"
      data-testid="mcp-endpoint-modal"
      title={
        isEditMode ? (
          <FormattedMessage
            defaultMessage="Edit access endpoint"
            description="MCP registry edit access endpoint modal title"
          />
        ) : (
          createTitle || (
            <FormattedMessage
              defaultMessage="Create access endpoint"
              description="MCP registry create access endpoint modal title"
            />
          )
        )
      }
      visible={visible}
      onCancel={onCancel}
      onOk={handleSubmit}
      okText={
        isEditMode
          ? intl.formatMessage({
              defaultMessage: 'Save',
              description: 'MCP registry edit access endpoint modal save button',
            })
          : intl.formatMessage({
              defaultMessage: 'Create',
              description: 'MCP registry create access endpoint modal create button',
            })
      }
      confirmLoading={isSubmitting}
      okButtonProps={{ disabled: !isFormValid || isSubmitting }}
    >
      <div css={flexColumnGapStyles(theme, theme.spacing.md)}>
        {activeMutation.error && (
          <Alert
            componentId="mlflow.mcp_registry.endpoint_modal.error"
            type="error"
            message={activeMutation.error?.message}
            closable={false}
          />
        )}

        <div>
          <FieldLabel>
            <FormattedMessage defaultMessage="MCP Server:" description="MCP registry endpoint modal server label" />
          </FieldLabel>
          {isServerLocked ? (
            <Typography.Text data-testid="mcp-endpoint-modal-locked-server">
              {editEndpoint ? resolveEndpointDisplayName(editEndpoint) : selectedServer}
            </Typography.Text>
          ) : (
            <SimpleSelect
              id="mcp-registry-endpoint-server"
              componentId="mlflow.mcp_registry.endpoint_modal.server"
              data-testid="mcp-endpoint-modal-server-selector"
              value={selectedServer}
              onChange={({ target }) => {
                setSelectedServer(target.value);
                setSelectedTarget(`${ALIAS_PREFIX}latest`);
              }}
              disabled={isSubmitting}
              placeholder={intl.formatMessage({
                defaultMessage: 'Select an MCP server',
                description: 'MCP registry endpoint modal server placeholder',
              })}
            >
              {servers?.map((s) => (
                <SimpleSelectOption key={s.name} value={s.name}>
                  {s.display_name || s.name}
                </SimpleSelectOption>
              ))}
            </SimpleSelect>
          )}
        </div>

        <div>
          <FieldLabel>
            <FormattedMessage defaultMessage="Endpoint URL:" description="MCP registry endpoint modal endpoint label" />
          </FieldLabel>
          <Input
            componentId="mlflow.mcp_registry.endpoint_modal.endpoint"
            value={endpointUrl}
            onChange={(e) => setEndpointUrl(e.target.value)}
            disabled={isSubmitting}
            placeholder={intl.formatMessage({
              defaultMessage: 'https://mcp.example.com/server',
              description: 'MCP registry endpoint modal endpoint placeholder',
            })}
            validationState={endpointUrl.trim() && !isValidUrl ? 'error' : undefined}
          />
          {endpointUrl.trim() && !isValidUrl && (
            <Typography.Text color="error" size="sm" data-testid="mcp-endpoint-modal-url-error">
              <FormattedMessage
                defaultMessage="Enter a valid HTTP or HTTPS URL"
                description="MCP registry endpoint modal endpoint URL validation error"
              />
            </Typography.Text>
          )}
        </div>

        <div>
          <FieldLabel>
            <FormattedMessage
              defaultMessage="Version/Alias:"
              description="MCP registry endpoint modal version/alias label"
            />
          </FieldLabel>
          <EndpointTargetSelector
            value={selectedTarget}
            onChange={setSelectedTarget}
            disabled={!selectedServer || isSubmitting}
            scopedVersion={scopedVersion}
            scopedAliases={scopedAliases}
            aliases={aliases}
            versions={versions ?? []}
          />
        </div>

        <div>
          <FieldLabel>
            <FormattedMessage
              defaultMessage="Transport Type:"
              description="MCP registry endpoint modal transport label"
            />
          </FieldLabel>
          <SimpleSelect
            id="mcp-registry-endpoint-transport"
            componentId="mlflow.mcp_registry.endpoint_modal.transport"
            value={transportType}
            onChange={({ target }) => setTransportType(target.value as MCPRemoteTransportType)}
            disabled={isSubmitting}
          >
            <SimpleSelectOption value="streamable-http">
              <FormattedMessage
                defaultMessage="Streamable HTTP"
                description="MCP registry streamable HTTP transport option"
              />
            </SimpleSelectOption>
            <SimpleSelectOption value="sse">
              <FormattedMessage
                defaultMessage="Server-Sent Events (SSE)"
                description="MCP registry SSE transport option"
              />
            </SimpleSelectOption>
          </SimpleSelect>
        </div>
      </div>
    </Modal>
  );
};
