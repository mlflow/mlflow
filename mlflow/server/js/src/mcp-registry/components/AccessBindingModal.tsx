import type React from 'react';
import { useEffect, useState } from 'react';
import {
  Alert,
  Input,
  Modal,
  SimpleSelect,
  SimpleSelectOption,
  SimpleSelectOptionGroup,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import type { MCPAccessBinding, MCPRemoteTransportType } from '../types';
import { useMCPServerQuery, useMCPServerVersionsQuery } from '../hooks/useMCPServerDetailQuery';
import { useMCPServersListQuery } from '../hooks/useMCPServersListQuery';
import { useCreateAccessBindingMutation, useUpdateAccessBindingMutation } from '../hooks/useAccessBindingMutation';
import { isValidEndpointUrl, resolveBindingDisplayName } from '../utils';
import { FieldLabel } from '../../admin/components/FieldLabel';

const ALIAS_PREFIX = 'alias:';
const VERSION_PREFIX = 'version:';

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
  defaultVersion,
  scopedVersion,
  scopedAliases,
  createTitle,
}: {
  visible: boolean;
  onCancel: () => void;
  onSuccess?: () => void;
  editBinding?: MCPAccessBinding;
  lockedServer?: string;
  defaultVersion?: string;
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
  const [transportType, setTransportType] = useState<MCPRemoteTransportType>('streamable-http');

  const createMutation = useCreateAccessBindingMutation();
  const updateMutation = useUpdateAccessBindingMutation();
  const activeMutation = isEditMode ? updateMutation : createMutation;

  // TODO: useMCPServersListQuery paginates; the dropdown only shows the first page.
  // Switch to a searchable combobox with scroll-based pagination or fetch all pages.
  const { data: servers } = useMCPServersListQuery({ enabled: !isServerLocked });
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
        setSelectedTarget(
          scopedVersion
            ? `${VERSION_PREFIX}${scopedVersion}`
            : defaultVersion
              ? `${VERSION_PREFIX}${defaultVersion}`
              : `${ALIAS_PREFIX}latest`,
        );
        setTransportType('streamable-http');
      }
      createMutation.reset();
      updateMutation.reset();
    }
  }, [visible, editBinding, lockedServer, defaultVersion, scopedVersion]); // eslint-disable-line react-hooks/exhaustive-deps -- reset() creates new ref

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
        {
          onSuccess: () => {
            onCancel();
            onSuccess?.();
          },
        },
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
        {
          onSuccess: () => {
            onCancel();
            onSuccess?.();
          },
        },
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
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
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
          <SimpleSelect
            id="mcp-registry-binding-target"
            componentId="mlflow.mcp_registry.binding_modal.target"
            value={selectedTarget}
            onChange={({ target }) => setSelectedTarget(target.value)}
            disabled={!selectedServer || isSubmitting}
          >
            {scopedVersion ? (
              <>
                <SimpleSelectOptionGroup
                  label={intl.formatMessage({
                    defaultMessage: 'Version',
                    description: 'MCP registry binding modal version group label',
                  })}
                >
                  <SimpleSelectOption value={`${VERSION_PREFIX}${scopedVersion}`}>{scopedVersion}</SimpleSelectOption>
                </SimpleSelectOptionGroup>
                {scopedAliases && scopedAliases.length > 0 && (
                  <SimpleSelectOptionGroup
                    label={intl.formatMessage({
                      defaultMessage: 'Aliases',
                      description: 'MCP registry binding modal aliases group label',
                    })}
                  >
                    {scopedAliases.map((alias) => (
                      <SimpleSelectOption key={alias} value={`${ALIAS_PREFIX}${alias}`}>
                        @{alias}
                      </SimpleSelectOption>
                    ))}
                  </SimpleSelectOptionGroup>
                )}
              </>
            ) : (
              <>
                <SimpleSelectOptionGroup
                  label={intl.formatMessage({
                    defaultMessage: 'Aliases',
                    description: 'MCP registry binding modal aliases group label',
                  })}
                >
                  <SimpleSelectOption value={`${ALIAS_PREFIX}latest`}>
                    <FormattedMessage defaultMessage="@latest" description="MCP registry latest alias option" />
                  </SimpleSelectOption>
                  {aliases.map((a) => (
                    <SimpleSelectOption key={a.alias} value={`${ALIAS_PREFIX}${a.alias}`}>
                      @{a.alias}
                    </SimpleSelectOption>
                  ))}
                </SimpleSelectOptionGroup>
                {versions && versions.length > 0 && (
                  <SimpleSelectOptionGroup
                    label={intl.formatMessage({
                      defaultMessage: 'Versions',
                      description: 'MCP registry binding modal versions group label',
                    })}
                  >
                    {versions.map((v) => (
                      <SimpleSelectOption key={v.version} value={`${VERSION_PREFIX}${v.version}`}>
                        {v.version}
                      </SimpleSelectOption>
                    ))}
                  </SimpleSelectOptionGroup>
                )}
              </>
            )}
          </SimpleSelect>
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
