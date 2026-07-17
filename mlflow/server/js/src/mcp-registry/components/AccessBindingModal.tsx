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

import type { MCPAccessBinding, MCPRemoteTransportType } from '../types';
import { TransportType } from '../types';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useMCPServerQuery, useMCPServerVersionsQuery } from '../hooks/useMCPServerDetailQuery';
import { MCPRegistryApi } from '../api';
import { MCP_QUERY_KEYS } from '../utils';
import { flexColumnGapStyles } from '../styles';
import { useCreateAccessBindingMutation, useUpdateAccessBindingMutation } from '../hooks/useAccessBindingMutation';
import { isValidEndpointUrl, resolveBindingDisplayName } from '../utils';
import { FieldLabel } from '../../admin/components/FieldLabel';
import { BindingTargetSelector, ALIAS_PREFIX, VERSION_PREFIX } from './BindingTargetSelector';

function bindingToTarget(binding: MCPAccessBinding): string {
  if (binding.server_alias) return `${ALIAS_PREFIX}${binding.server_alias}`;
  if (binding.server_version) return `${VERSION_PREFIX}${binding.server_version}`;
  return `${ALIAS_PREFIX}latest`;
}

export const AccessBindingModal = ({
  visible,
  onCancel,
  onSuccess,
  editBinding,
  lockedServer,
  scopedVersion,
  scopedAliases,
  createTitle,
}: {
  visible: boolean;
  onCancel: () => void;
  onSuccess?: () => void;
  editBinding?: MCPAccessBinding;
  lockedServer?: string;
  scopedVersion?: string;
  scopedAliases?: string[];
  createTitle?: React.ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const isEditMode = Boolean(editBinding);
  const isServerLocked = isEditMode || Boolean(lockedServer);

  const [selectedServer, setSelectedServer] = useState('');
  const [endpointUrl, setEndpointUrl] = useState('');
  const [selectedTarget, setSelectedTarget] = useState(`${ALIAS_PREFIX}latest`);
  const [transportType, setTransportType] = useState<MCPRemoteTransportType>(TransportType.STREAMABLE_HTTP);

  const createMutation = useCreateAccessBindingMutation();
  const updateMutation = useUpdateAccessBindingMutation();
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
      if (editBinding) {
        setSelectedServer(editBinding.server_name);
        setEndpointUrl(editBinding.endpoint_url);
        setSelectedTarget(bindingToTarget(editBinding));
        setTransportType(editBinding.transport_type);
      } else {
        setSelectedServer(lockedServer || '');
        setEndpointUrl('');
        setSelectedTarget(scopedVersion ? `${VERSION_PREFIX}${scopedVersion}` : `${ALIAS_PREFIX}latest`);
        setTransportType(TransportType.STREAMABLE_HTTP);
      }
      createMutation.reset();
      updateMutation.reset();
    }
  }, [visible, editBinding, lockedServer, scopedVersion]); // eslint-disable-line react-hooks/exhaustive-deps -- reset() creates new ref

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

    if (isEditMode && editBinding) {
      updateMutation.mutate(
        {
          serverName: editBinding.server_name,
          bindingId: editBinding.binding_id,
          request: {
            endpoint_url: endpointUrl.trim(),
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
            endpoint_url: endpointUrl.trim(),
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
      componentId="mlflow.mcp_registry.binding_modal"
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
            componentId="mlflow.mcp_registry.binding_modal.error"
            type="error"
            message={activeMutation.error?.message}
            closable={false}
          />
        )}

        <div>
          <FieldLabel>
            <FormattedMessage defaultMessage="MCP Server:" description="MCP registry binding modal server label" />
          </FieldLabel>
          {isServerLocked ? (
            <Typography.Text>{editBinding ? resolveBindingDisplayName(editBinding) : selectedServer}</Typography.Text>
          ) : (
            <SimpleSelect
              id="mcp-registry-binding-server"
              componentId="mlflow.mcp_registry.binding_modal.server"
              value={selectedServer}
              onChange={({ target }) => {
                setSelectedServer(target.value);
                setSelectedTarget(`${ALIAS_PREFIX}latest`);
              }}
              disabled={isSubmitting}
              placeholder={intl.formatMessage({
                defaultMessage: 'Select an MCP server',
                description: 'MCP registry binding modal server placeholder',
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
            <FormattedMessage defaultMessage="Endpoint URL:" description="MCP registry binding modal endpoint label" />
          </FieldLabel>
          <Input
            componentId="mlflow.mcp_registry.binding_modal.endpoint"
            value={endpointUrl}
            onChange={(e) => setEndpointUrl(e.target.value)}
            disabled={isSubmitting}
            placeholder={intl.formatMessage({
              defaultMessage: 'https://mcp.example.com/server',
              description: 'MCP registry binding modal endpoint placeholder',
            })}
            validationState={endpointUrl.trim() && !isValidUrl ? 'error' : undefined}
          />
          {endpointUrl.trim() && !isValidUrl && (
            <Typography.Text color="error" size="sm">
              <FormattedMessage
                defaultMessage="Enter a valid HTTP or HTTPS URL"
                description="MCP registry binding modal endpoint URL validation error"
              />
            </Typography.Text>
          )}
        </div>

        <div>
          <FieldLabel>
            <FormattedMessage
              defaultMessage="Version/Alias:"
              description="MCP registry binding modal version/alias label"
            />
          </FieldLabel>
          <BindingTargetSelector
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
              description="MCP registry binding modal transport label"
            />
          </FieldLabel>
          <SimpleSelect
            id="mcp-registry-binding-transport"
            componentId="mlflow.mcp_registry.binding_modal.transport"
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
