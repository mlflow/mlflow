import {
  Alert,
  Button,
  FormUI,
  Input,
  Modal,
  PlusIcon,
  RHFControlledComponents,
  SimpleSelect,
  SimpleSelectOption,
  Spacer,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { FormattedMessage, useIntl } from 'react-intl';
import type { MCPServerVersion, MCPStatus } from '../types';
import { useCreateMCPServerVersionMutation } from './useCreateMCPServerVersionMutation';
import { validateServerJson, validateToolsJson } from '../utils';
import { KeyValueTag } from '../../common/components/KeyValueTag';
import type { KeyValueEntity } from '../../common/types';
import { TagKeySelectDropdown } from '../../common/components/TagSelectDropdown';

interface CreateMCPServerVersionFormState {
  displayName: string;
  serverJsonText: string;
  status: MCPStatus;
  source: string;
  toolsText: string;
  tags: Record<string, string>;
}

const INITIAL_FORM_STATE: CreateMCPServerVersionFormState = {
  displayName: '',
  serverJsonText: '',
  status: 'draft',
  source: '',
  toolsText: '',
  tags: {},
};

export const useCreateMCPServerVersionModal = ({
  onSuccess,
  serverName,
  latestVersion,
}: {
  onSuccess?: (result: { name: string; version: string }) => void;
  serverName?: string;
  latestVersion?: MCPServerVersion;
} = {}) => {
  const isVersionMode = Boolean(serverName);
  const [open, setOpen] = useState(false);
  const [formState, setFormState] = useState<CreateMCPServerVersionFormState>(INITIAL_FORM_STATE);
  const [validationError, setValidationError] = useState<string | undefined>(undefined);
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const { mutate, error: mutationError, reset: resetMutation, isLoading } = useCreateMCPServerVersionMutation();

  const tagForm = useForm<KeyValueEntity>({ defaultValues: { key: undefined, value: '' } });
  const tagFormValues = tagForm.watch();

  const handleAddTag = () => {
    if (!tagFormValues.key?.trim()) return;
    setFormState((prev) => ({
      ...prev,
      tags: { ...prev.tags, [tagFormValues.key.trim()]: tagFormValues.value?.trim() || '' },
    }));
    tagForm.reset();
  };

  const handleRemoveTag = (key: string) => {
    setFormState((prev) => {
      const next = { ...prev.tags };
      delete next[key];
      return { ...prev, tags: next };
    });
  };

  const handleFieldChange = <K extends keyof CreateMCPServerVersionFormState>(
    field: K,
    value: CreateMCPServerVersionFormState[K],
  ) => {
    setFormState((prev) => ({ ...prev, [field]: value }));
    if (validationError) {
      setValidationError(undefined);
    }
  };

  const handleSubmit = () => {
    if (isLoading) return;
    const serverJsonResult = validateServerJson(formState.serverJsonText);
    if (!serverJsonResult.valid || !serverJsonResult.parsed) {
      setValidationError(serverJsonResult.error);
      return;
    }

    let parsedTools;
    if (formState.toolsText.trim()) {
      const toolsResult = validateToolsJson(formState.toolsText);
      if (!toolsResult.valid) {
        setValidationError(toolsResult.error);
        return;
      }
      parsedTools = toolsResult.parsed as { name: string; [key: string]: unknown }[];
    }

    setValidationError(undefined);

    const tagsToSet = Object.keys(formState.tags).length > 0 ? formState.tags : undefined;

    mutate(
      {
        serverJson: serverJsonResult.parsed,
        displayName: formState.displayName.trim() || undefined,
        isNewServer: !isVersionMode,
        status: formState.status,
        source: formState.source.trim() || undefined,
        tools: parsedTools,
        tags: tagsToSet,
      },
      {
        onSuccess: (data) => {
          onSuccess?.({ name: data.name, version: data.version });
          setOpen(false);
        },
      },
    );
  };

  const displayError = validationError || mutationError?.message;

  const modalElement = (
    <Modal
      componentId="mlflow.mcp_registry.create_server_version.modal"
      visible={open}
      onCancel={() => setOpen(false)}
      title={
        isVersionMode ? (
          <FormattedMessage
            defaultMessage="Create MCP server version"
            description="Title for the create MCP server version modal when adding a version to an existing server"
          />
        ) : (
          <FormattedMessage
            defaultMessage="Create MCP server"
            description="Title for the create MCP server version modal"
          />
        )
      }
      okText={
        <FormattedMessage
          defaultMessage="Create"
          description="Label for the confirm button in the create MCP server version modal"
        />
      }
      okButtonProps={{
        loading: isLoading,
        disabled: !formState.serverJsonText.trim(),
      }}
      onOk={handleSubmit}
      cancelText={
        <FormattedMessage
          defaultMessage="Cancel"
          description="Label for the cancel button in the create MCP server version modal"
        />
      }
      size="wide"
    >
      {displayError && (
        <>
          <Alert
            componentId="mlflow.mcp_registry.create_server_version.error"
            closable={false}
            message={displayError}
            type="error"
          />
          <Spacer />
        </>
      )}
      <FormUI.Label htmlFor="mlflow.mcp_registry.create.display_name">
        <FormattedMessage
          defaultMessage="Display name:"
          description="Label for display name field in create MCP server modal"
        />
      </FormUI.Label>
      <Input
        componentId="mlflow.mcp_registry.create.display_name"
        id="mlflow.mcp_registry.create.display_name"
        value={formState.displayName}
        onChange={(e) => handleFieldChange('displayName', e.target.value)}
        placeholder={intl.formatMessage({
          defaultMessage: 'Human-readable label for this server',
          description: 'Placeholder for display name in create MCP server modal',
        })}
      />
      <Spacer />
      <FormUI.Label htmlFor="mlflow.mcp_registry.create.server_json">
        <FormattedMessage
          defaultMessage="server.json:"
          description="Label for server.json field in create MCP server modal"
        />
        <span css={{ color: theme.colors.textValidationDanger, marginLeft: 2 }}>*</span>
      </FormUI.Label>
      <Input.TextArea
        componentId="mlflow.mcp_registry.create.server_json"
        id="mlflow.mcp_registry.create.server_json"
        value={formState.serverJsonText}
        onChange={(e) => handleFieldChange('serverJsonText', e.target.value)}
        placeholder={intl.formatMessage({
          defaultMessage: 'Enter your MCP server definition',
          description: 'Placeholder for server.json in create MCP server modal',
        })}
        autoSize={{ minRows: 6, maxRows: 14 }}
        css={{ fontFamily: 'monospace' }}
      />
      <Spacer />
      <FormUI.Label htmlFor="mlflow.mcp_registry.create.status">
        <FormattedMessage defaultMessage="Status:" description="Label for status field in create MCP server modal" />
        <span css={{ color: theme.colors.textValidationDanger, marginLeft: 2 }}>*</span>
      </FormUI.Label>
      <SimpleSelect
        componentId="mlflow.mcp_registry.create.status"
        id="mlflow.mcp_registry.create.status"
        value={formState.status}
        onChange={({ target }) => handleFieldChange('status', target.value as MCPStatus)}
      >
        <SimpleSelectOption value="draft">
          <FormattedMessage defaultMessage="Draft" description="Draft status option in create MCP server modal" />
        </SimpleSelectOption>
        <SimpleSelectOption value="active">
          <FormattedMessage defaultMessage="Active" description="Active status option in create MCP server modal" />
        </SimpleSelectOption>
        <SimpleSelectOption value="deprecated">
          <FormattedMessage
            defaultMessage="Deprecated"
            description="Deprecated status option in create MCP server modal"
          />
        </SimpleSelectOption>
      </SimpleSelect>
      <Spacer />
      <FormUI.Label htmlFor="mlflow.mcp_registry.create.source">
        <FormattedMessage defaultMessage="Source:" description="Label for source field in create MCP server modal" />
      </FormUI.Label>
      <Input
        componentId="mlflow.mcp_registry.create.source"
        id="mlflow.mcp_registry.create.source"
        value={formState.source}
        onChange={(e) => handleFieldChange('source', e.target.value)}
        placeholder={intl.formatMessage({
          defaultMessage: 'https://github.com/org/repo',
          description: 'Placeholder for source in create MCP server modal',
        })}
      />
      <Spacer />
      <FormUI.Label htmlFor="mlflow.mcp_registry.create.tools">
        <FormattedMessage defaultMessage="Tools:" description="Label for tools field in create MCP server modal" />
      </FormUI.Label>
      <Input.TextArea
        componentId="mlflow.mcp_registry.create.tools"
        id="mlflow.mcp_registry.create.tools"
        value={formState.toolsText}
        onChange={(e) => handleFieldChange('toolsText', e.target.value)}
        placeholder='[{"name": "search", "description": "Search the web"}]'
        autoSize={{ minRows: 3, maxRows: 8 }}
        css={{ fontFamily: 'monospace' }}
      />
      <Spacer />
      <FormUI.Label>
        {isVersionMode ? (
          <FormattedMessage
            defaultMessage="Metadata:"
            description="Label for metadata field in create MCP server version modal"
          />
        ) : (
          <FormattedMessage defaultMessage="Tags:" description="Label for tags field in create MCP server modal" />
        )}
      </FormUI.Label>
      <form
        onSubmit={tagForm.handleSubmit(handleAddTag)}
        css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md, marginTop: theme.spacing.xs }}
      >
        <div css={{ minWidth: 0, display: 'flex', gap: theme.spacing.md, flex: 1 }}>
          <div css={{ flex: 1 }}>
            <TagKeySelectDropdown allAvailableTags={[]} control={tagForm.control} />
          </div>
          <div css={{ flex: 1 }}>
            <RHFControlledComponents.Input
              componentId="mlflow.mcp_registry.create.tag.value"
              name="value"
              control={tagForm.control}
              placeholder={intl.formatMessage({
                defaultMessage: 'Type a value',
                description: 'Placeholder for tag value input in create MCP server modal',
              })}
            />
          </div>
        </div>
        <Tooltip
          content={intl.formatMessage({
            defaultMessage: 'Add tag',
            description: 'Tooltip for add tag button in create MCP server modal',
          })}
          componentId="mlflow.mcp_registry.create.tag.add.tooltip"
        >
          <Button
            componentId="mlflow.mcp_registry.create.tag.add"
            htmlType="submit"
            aria-label={intl.formatMessage({
              defaultMessage: 'Add tag',
              description: 'Aria label for add tag button in create MCP server modal',
            })}
          >
            <PlusIcon />
          </Button>
        </Tooltip>
      </form>
      {Object.keys(formState.tags).length > 0 && (
        <div
          css={{
            display: 'flex',
            rowGap: theme.spacing.xs,
            flexWrap: 'wrap',
            marginTop: theme.spacing.sm,
          }}
        >
          {Object.entries(formState.tags).map(([key, value]) => (
            <KeyValueTag isClosable tag={{ key, value }} onClose={() => handleRemoveTag(key)} key={key} />
          ))}
        </div>
      )}
    </Modal>
  );

  const openModal = () => {
    resetMutation();
    setValidationError(undefined);

    if (latestVersion) {
      setFormState({
        displayName: '',
        serverJsonText: JSON.stringify(latestVersion.server_json, null, 2),
        status: latestVersion.status === 'deleted' ? 'draft' : latestVersion.status,
        source: latestVersion.source || '',
        toolsText: latestVersion.tools?.length ? JSON.stringify(latestVersion.tools, null, 2) : '',
        tags: { ...latestVersion.tags },
      });
    } else {
      setFormState({
        ...INITIAL_FORM_STATE,
        serverJsonText: serverName ? JSON.stringify({ name: serverName }, null, 2) : '',
      });
    }
    tagForm.reset();
    setOpen(true);
  };

  return { CreateMCPServerVersionModal: modalElement, openModal };
};
