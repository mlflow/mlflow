import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  Drawer,
  FormUI,
  Input,
  LightningIcon,
  PencilIcon,
  PlusIcon,
  Radio,
  SimpleSelect,
  SimpleSelectOption,
  Spinner,
  Tag,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useState, useMemo, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import {
  TagAssignmentRoot,
  TagAssignmentRow,
  TagAssignmentLabel,
  TagAssignmentKey,
  TagAssignmentValue,
  TagAssignmentRemoveButton,
  useTagAssignmentForm,
} from '@databricks/web-shared/unified-tagging';
import type { Endpoint, EndpointModel } from '../types';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { Descriptions } from '@mlflow/mlflow/src/common/components/Descriptions';
import { KeyValueTag } from '@mlflow/mlflow/src/common/components/KeyValueTag';
import { ModelInfo } from './ModelInfo';
import { useListSecrets } from '../hooks/useListSecrets';
import { useListEndpointBindings } from '../hooks/useListEndpointBindings';
import { useUpdateEndpoint } from '../hooks/useUpdateEndpoint';
import { useUpdateEndpointModel } from '../hooks/useUpdateEndpointModel';
import { useRemoveEndpointModel } from '../hooks/useRemoveEndpointModel';
import { useAddEndpointModel } from '../hooks/useAddEndpointModel';
import { secretsApi } from '../api/secretsApi';
import { LIST_SECRETS_QUERY_KEY, LIST_ROUTES_QUERY_KEY } from '../constants';
import { MaskedApiKeyInput } from './MaskedApiKeyInput';
import { AuthConfigFields } from './AuthConfigFields';
import { PROVIDERS } from './routeConstants';

const EMPTY_TAG_ENTITY = { key: '', value: '' };
const EMPTY_TAG_ARRAY: { key: string; value: string }[] = [];

export interface RouteDetailDrawerProps {
  route: Endpoint | null;
  open: boolean;
  onClose: () => void;
  onUpdate?: (
    routeId: string,
    updateData: {
      secret_id?: string;
      secret_name?: string;
      secret_value?: string;
      provider?: string;
      auth_config?: string;
      route_description?: string;
      route_tags?: string;
    },
  ) => void;
  onDelete?: (route: Endpoint) => void;
}

type ManagementOperation = 'changeSecret' | 'deleteRoute' | null;
type SecretSource = 'existing' | 'new';
type EditingField = 'name' | 'description' | 'tags' | null;

export const RouteDetailDrawer = ({ route, open, onClose, onUpdate, onDelete }: RouteDetailDrawerProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const queryClient = useQueryClient();

  // Management operation state
  const [selectedOperation, setSelectedOperation] = useState<ManagementOperation>(null);
  const [showResourceUsage, setShowResourceUsage] = useState(false);

  // Delete route state
  const [routeDeleteConfirmation, setRouteDeleteConfirmation] = useState('');

  // Change secret state
  const [secretSource, setSecretSource] = useState<SecretSource>('existing');
  const [selectedSecretId, setSelectedSecretId] = useState<string>('');
  const [newSecretName, setNewSecretName] = useState<string>('');
  const [newSecretValue, setNewSecretValue] = useState<string>('');
  const [newSecretProvider, setNewSecretProvider] = useState<string>('');
  const [authConfig, setAuthConfig] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<{
    secretId?: string;
    secretName?: string;
    secretValue?: string;
    provider?: string;
  }>({});
  const [isLoading, setIsLoading] = useState(false);

  // Inline editing state
  const [editingField, setEditingField] = useState<EditingField>(null);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');
  type TagEntity = { key: string; value: string };
  const tagsForm = useForm<{ tags: TagEntity[] }>({ mode: 'onChange' });
  const tagsFieldArray = useTagAssignmentForm({
    name: 'tags',
    emptyValue: EMPTY_TAG_ENTITY,
    keyProperty: 'key',
    valueProperty: 'value',
    form: tagsForm,
    defaultValues: EMPTY_TAG_ARRAY,
  });

  // Model-level operations state
  const [selectedModelForEdit, setSelectedModelForEdit] = useState<EndpointModel | null>(null);
  const [selectedModelForRemoval, setSelectedModelForRemoval] = useState<EndpointModel | null>(null);
  const [showAddModelDialog, setShowAddModelDialog] = useState(false);
  const [selectedModelForModelChange, setSelectedModelForModelChange] = useState<EndpointModel | null>(null);
  const [newModelName, setNewModelName] = useState<string>('');

  // Per-model secret change form state
  const [modelEditSecretSource, setModelEditSecretSource] = useState<SecretSource>('existing');
  const [modelEditSelectedSecretId, setModelEditSelectedSecretId] = useState<string>('');
  const [modelEditNewSecretName, setModelEditNewSecretName] = useState<string>('');
  const [modelEditNewSecretValue, setModelEditNewSecretValue] = useState<string>('');
  const [modelEditAuthConfig, setModelEditAuthConfig] = useState<Record<string, string>>({});
  const [modelEditErrors, setModelEditErrors] = useState<{
    secretId?: string;
    secretName?: string;
    secretValue?: string;
  }>({});

  // Fetch data
  const { secrets = [] } = useListSecrets({ enabled: open });
  // Disabled: bindings API endpoint not yet available
  // const { bindings: endpointBindings = [] } = useListEndpointBindings({
  //   endpointId: route?.endpoint_id || '',
  //   enabled: open && !!route?.endpoint_id,
  // });
  const endpointBindings: any[] = [];

  // Model-level mutations
  const { updateEndpoint: updateEndpointMutation, isLoading: isUpdatingEndpoint } = useUpdateEndpoint();
  const { updateModel: updateModelMutation, isLoading: isUpdatingModel } = useUpdateEndpointModel();
  const { removeModel: removeModelMutation, isLoading: isRemovingModel } = useRemoveEndpointModel();
  const { addModel: addModelMutation, isLoading: isAddingModel } = useAddEndpointModel();

  // No longer need single currentSecret or routeBinding - each model has its own secret

  // Reset when route changes
  useEffect(() => {
    if (route && open) {
      setSelectedOperation(null);
      setEditingField(null);
      setShowResourceUsage(false);
      setRouteDeleteConfirmation('');
      setSecretSource('existing');
      setSelectedSecretId('');
      setNewSecretName('');
      setNewSecretValue('');
      // Get provider from first model if available
      const firstModelProvider = route.models?.[0]?.provider || '';
      setNewSecretProvider(firstModelProvider);
      setAuthConfig({});
      setErrors({});

      // Initialize edit details
      setEditDescription(route.description || '');
      const existingTags = route.tags
        ? Array.isArray(route.tags)
          ? route.tags
          : Object.entries(route.tags).map(([key, value]) => ({ key, value }))
        : [];
      // Always include an empty tag at the end for adding new tags
      const initialTags = existingTags.length > 0 ? [...existingTags, EMPTY_TAG_ENTITY] : [EMPTY_TAG_ENTITY];
      tagsForm.reset({ tags: initialTags });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [route?.endpoint_id, open, secrets]);

  // Convert tags to array format
  const tagEntities = route?.tags
    ? Array.isArray(route.tags)
      ? route.tags
      : Object.entries(route.tags).map(([key, value]) => ({ key, value }))
    : [];

  // Filter endpoint bindings to exclude self (this endpoint's own binding to the secret)
  const externalResourceBindings = useMemo(() => {
    if (!route) return [];
    // Filter out bindings that are just this endpoint's own secret binding
    // We want to show other resources (experiments, jobs, etc.) that are using this endpoint
    return endpointBindings.filter((binding) => {
      // Keep bindings that have a resource_type other than ROUTE
      // or if they have a resource_id that's different from this endpoint
      return binding.resource_type !== 'ROUTE' || binding.resource_id !== route.endpoint_id;
    });
  }, [endpointBindings, route]);

  // Format resource type for display
  const formatResourceType = (resourceType: string) => {
    const typeMap: Record<string, string> = {
      SCORER_JOB: 'Scorer Job',
      GLOBAL: 'Global Resource',
      ROUTE: 'Endpoint',
      EXPERIMENT: 'Experiment',
      RUN: 'Run',
    };
    return typeMap[resourceType] || resourceType.split('_').map((word: string) => word.charAt(0) + word.slice(1).toLowerCase()).join(' ');
  };

  // Get display name for resource
  const getResourceName = (binding: any) => {
    // If there's a route_name, use it (for endpoint bindings)
    if (binding.route_name) {
      return binding.route_name;
    }
    // Otherwise show the resource_id (which might be a job ID, experiment ID, etc.)
    return binding.resource_id;
  };

  // Filter secrets to only show those from compatible providers
  // When changing secrets, filter by the selected provider in the form
  const compatibleSecrets = useMemo(() => {
    if (!newSecretProvider) return secrets;

    return secrets.filter((secret) => {
      if (secret.provider) {
        return secret.provider.toLowerCase() === newSecretProvider.toLowerCase();
      }
      const secretNameLower = secret.secret_name.toLowerCase();
      const providerLower = newSecretProvider.toLowerCase();
      if (providerLower === 'openai') return secretNameLower.includes('openai');
      if (providerLower === 'anthropic')
        return secretNameLower.includes('anthropic') || secretNameLower.includes('claude');
      if (providerLower === 'bedrock') return secretNameLower.includes('bedrock') || secretNameLower.includes('aws');
      if (providerLower === 'vertex_ai')
        return (
          secretNameLower.includes('vertex') || secretNameLower.includes('gemini') || secretNameLower.includes('google')
        );
      if (providerLower === 'azure') return secretNameLower.includes('azure');
      if (providerLower === 'databricks') return secretNameLower.includes('databricks');
      return true;
    });
  }, [secrets, newSecretProvider]);

  // Get the selected provider info with auth config fields
  const selectedProviderInfo = useMemo(() => {
    return PROVIDERS.find((p) => p.value === newSecretProvider);
  }, [newSecretProvider]);

  // Filter secrets for the selected model being edited (by model's provider)
  const modelEditCompatibleSecrets = useMemo(() => {
    if (!selectedModelForEdit?.provider) return secrets;

    const modelProvider = selectedModelForEdit.provider;
    return secrets.filter((secret) => {
      if (secret.provider) {
        return secret.provider.toLowerCase() === modelProvider.toLowerCase();
      }
      const secretNameLower = secret.secret_name.toLowerCase();
      const providerLower = modelProvider.toLowerCase();
      if (providerLower === 'openai') return secretNameLower.includes('openai');
      if (providerLower === 'anthropic')
        return secretNameLower.includes('anthropic') || secretNameLower.includes('claude');
      if (providerLower === 'bedrock') return secretNameLower.includes('bedrock') || secretNameLower.includes('aws');
      if (providerLower === 'vertex_ai')
        return (
          secretNameLower.includes('vertex') || secretNameLower.includes('gemini') || secretNameLower.includes('google')
        );
      if (providerLower === 'azure') return secretNameLower.includes('azure');
      if (providerLower === 'databricks') return secretNameLower.includes('databricks');
      return true;
    });
  }, [secrets, selectedModelForEdit?.provider]);

  // Get provider info for the model being edited
  const modelEditProviderInfo = useMemo(() => {
    return PROVIDERS.find((p) => p.value === selectedModelForEdit?.provider);
  }, [selectedModelForEdit?.provider]);

  // Handle change secret update
  const handleUpdate = async () => {
    if (!route) return;

    const newErrors: { secretId?: string; secretName?: string; secretValue?: string; provider?: string } = {};

    if (secretSource === 'existing') {
      if (!selectedSecretId) {
        newErrors.secretId = intl.formatMessage({
          defaultMessage: 'Please select an API key',
          description: 'Secret selection required error',
        });
      }
    } else {
      if (!newSecretProvider) {
        newErrors.provider = intl.formatMessage({
          defaultMessage: 'Provider is required',
          description: 'Provider required error',
        });
      }
      if (!newSecretName.trim()) {
        newErrors.secretName = intl.formatMessage({
          defaultMessage: 'Secret name is required',
          description: 'Secret name required error',
        });
      }
      if (!newSecretValue.trim()) {
        newErrors.secretValue = intl.formatMessage({
          defaultMessage: 'API key is required',
          description: 'API key required error',
        });
      }
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    setIsLoading(true);
    try {
      if (secretSource === 'existing') {
        await onUpdate?.(route.endpoint_id, { secret_id: selectedSecretId });
      } else {
        await onUpdate?.(route.endpoint_id, {
          secret_name: newSecretName,
          secret_value: newSecretValue,
          provider: newSecretProvider || undefined,
          auth_config: Object.keys(authConfig).length > 0 ? JSON.stringify(authConfig) : undefined,
        });
      }
      setSelectedOperation(null);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle delete route
  const handleDelete = () => {
    if (!route) return;
    onDelete?.(route);
    onClose();
  };

  // Handle save description
  const handleSaveDescription = async () => {
    if (!route || !onUpdate) return;

    setIsLoading(true);
    try {
      await onUpdate(route.endpoint_id, {
        route_description: editDescription.trim() || undefined,
      });

      setEditingField(null);
    } catch (err) {
      console.error('Failed to update endpoint description:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle save tags
  const handleSaveTags = async () => {
    if (!route || !onUpdate) return;

    setIsLoading(true);
    try {
      const tags = tagsForm.getValues().tags.filter((tag) => tag.key.trim() || tag.value.trim());
      const route_tags = tags.length > 0 ? JSON.stringify(tags) : undefined;

      await onUpdate(route.endpoint_id, {
        route_tags,
      });

      setEditingField(null);
    } catch (err) {
      console.error('Failed to update endpoint tags:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle cancel editing
  const handleCancelEditing = () => {
    if (!route) return;

    if (editingField === 'description') {
      setEditDescription(route.description || '');
    } else if (editingField === 'tags') {
      const existingTags = route.tags
        ? Array.isArray(route.tags)
          ? route.tags
          : Object.entries(route.tags).map(([key, value]) => ({ key, value }))
        : [];
      const initialTags = existingTags.length > 0 ? [...existingTags, EMPTY_TAG_ENTITY] : [EMPTY_TAG_ENTITY];
      tagsForm.reset({ tags: initialTags });
    }

    setEditingField(null);
  };

  // Handle model secret update
  const handleModelSecretUpdate = async () => {
    if (!route || !selectedModelForEdit) return;

    const newErrors: { secretId?: string; secretName?: string; secretValue?: string } = {};

    if (modelEditSecretSource === 'existing') {
      if (!modelEditSelectedSecretId) {
        newErrors.secretId = intl.formatMessage({
          defaultMessage: 'Please select an API key',
          description: 'Secret selection required error',
        });
      }
    } else {
      if (!modelEditNewSecretName.trim()) {
        newErrors.secretName = intl.formatMessage({
          defaultMessage: 'Secret name is required',
          description: 'Secret name required error',
        });
      }
      if (!modelEditNewSecretValue.trim()) {
        newErrors.secretValue = intl.formatMessage({
          defaultMessage: 'API key is required',
          description: 'API key required error',
        });
      }
    }

    if (Object.keys(newErrors).length > 0) {
      setModelEditErrors(newErrors);
      return;
    }

    try {
      let secretIdToUse = modelEditSelectedSecretId;

      // If creating a new secret, first create it
      if (modelEditSecretSource === 'new') {
        const createSecretResponse = await secretsApi.createSecret({
          secret_name: modelEditNewSecretName,
          secret_value: modelEditNewSecretValue,
          provider: selectedModelForEdit.provider,
          auth_config: Object.keys(modelEditAuthConfig).length > 0 ? JSON.stringify(modelEditAuthConfig) : undefined,
        });
        secretIdToUse = createSecretResponse.secret.secret_id;
        // Invalidate secrets query to refetch the new secret
        await queryClient.invalidateQueries({ queryKey: [LIST_SECRETS_QUERY_KEY] });
      }

      // Now update the model with the secret_id
      await updateModelMutation({
        endpoint_id: route.endpoint_id,
        model_id: selectedModelForEdit.model_id,
        secret_id: secretIdToUse,
      });

      // Invalidate both queries to refetch updated data
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: [LIST_ROUTES_QUERY_KEY] }),
        queryClient.invalidateQueries({ queryKey: [LIST_SECRETS_QUERY_KEY] }),
      ]);

      // Reset form and close edit mode
      handleCancelModelEdit();
    } catch (err) {
      console.error('Failed to update model secret:', err);
    }
  };

  // Handle cancel model secret edit
  const handleCancelModelEdit = () => {
    setSelectedModelForEdit(null);
    setModelEditSecretSource('existing');
    setModelEditSelectedSecretId('');
    setModelEditNewSecretName('');
    setModelEditNewSecretValue('');
    setModelEditAuthConfig({});
    setModelEditErrors({});
  };

  // Reset model edit form when selectedModelForEdit changes
  useEffect(() => {
    if (selectedModelForEdit) {
      setModelEditSecretSource('existing');
      setModelEditSelectedSecretId('');
      setModelEditNewSecretName('');
      setModelEditNewSecretValue('');
      setModelEditAuthConfig({});
      setModelEditErrors({});
    }
  }, [selectedModelForEdit]);

  const hasChanges =
    secretSource === 'new'
      ? newSecretName.trim().length > 0 || newSecretValue.trim().length > 0
      : selectedSecretId !== '';

  return (
    <>
      <Drawer.Root modal open={open} onOpenChange={onClose}>
        <Drawer.Content
          componentId="mlflow.routes.detail_drawer"
          width="700px"
          title={
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <div
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: 32,
                  height: 32,
                  borderRadius: theme.borders.borderRadiusMd,
                  backgroundColor: theme.colors.backgroundSecondary,
                }}
              >
                <LightningIcon css={{ fontSize: 18 }} />
              </div>
              <FormattedMessage defaultMessage="Endpoint Details" description="Endpoint detail drawer > drawer title" />
            </div>
          }
        >
          {route && (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
              {/* Endpoint Metadata Section */}
              <div
                css={{
                  padding: theme.spacing.lg,
                  borderRadius: theme.borders.borderRadiusMd,
                  backgroundColor: theme.colors.backgroundSecondary,
                  border: `1px solid ${theme.colors.border}`,
                }}
              >
                {/* Header with name */}
                <div
                  css={{
                    marginBottom: theme.spacing.md,
                  }}
                >
                  {editingField === 'name' ? (
                    <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
                      <Input
                        componentId="mlflow.routes.detail_drawer.edit_name"
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        placeholder={intl.formatMessage({
                          defaultMessage: 'Enter endpoint name',
                          description: 'Endpoint name input placeholder',
                        })}
                        autoFocus
                      />
                      <Button
                        componentId="mlflow.routes.detail_drawer.save_name"
                        type="primary"
                        size="small"
                        onClick={async () => {
                          if (!route) return;
                          try {
                            await updateEndpointMutation({
                              endpoint_id: route.endpoint_id,
                              name: editName,
                            });
                            setEditingField(null);
                          } catch (error) {
                            console.error('Failed to update endpoint name:', error);
                          }
                        }}
                        loading={isUpdatingEndpoint}
                        disabled={!editName.trim()}
                      >
                        <FormattedMessage defaultMessage="Save" description="Save button" />
                      </Button>
                      <Button
                        componentId="mlflow.routes.detail_drawer.cancel_name_edit"
                        size="small"
                        onClick={() => setEditingField(null)}
                      >
                        <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
                      </Button>
                    </div>
                  ) : (
                    <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.sm }}>
                      <Typography.Title level={3} css={{ margin: 0 }}>
                        {route.name || route.endpoint_id}
                      </Typography.Title>
                      <Button
                        componentId="mlflow.routes.detail_drawer.edit_name_button"
                        icon={<PencilIcon />}
                        size="small"
                        onClick={() => {
                          setEditName(route.name || '');
                          setEditingField('name');
                        }}
                        css={{ marginTop: 2 }}
                      />
                    </div>
                  )}
                </div>

                {/* Timestamps */}
                <div
                  css={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, 1fr)',
                    gap: theme.spacing.md,
                    marginBottom: theme.spacing.md,
                  }}
                >
                  <div>
                    <Typography.Text color="secondary" size="sm">
                      <FormattedMessage defaultMessage="Created" description="Route detail drawer > created label" />
                    </Typography.Text>
                    <Typography.Paragraph css={{ marginTop: theme.spacing.xs, marginBottom: 0, fontSize: 13 }}>
                      {Utils.formatTimestamp(route.created_at)}
                      {route.created_by && (
                        <Typography.Text
                          color="secondary"
                          size="sm"
                          css={{ display: 'block', marginTop: theme.spacing.xs }}
                        >
                          by {route.created_by}
                        </Typography.Text>
                      )}
                    </Typography.Paragraph>
                  </div>

                  <div>
                    <Typography.Text color="secondary" size="sm">
                      <FormattedMessage
                        defaultMessage="Last Updated"
                        description="Route detail drawer > last updated label"
                      />
                    </Typography.Text>
                    <Typography.Paragraph css={{ marginTop: theme.spacing.xs, marginBottom: 0, fontSize: 13 }}>
                      {Utils.formatTimestamp(route.last_updated_at)}
                      {route.last_updated_by && (
                        <Typography.Text
                          color="secondary"
                          size="sm"
                          css={{ display: 'block', marginTop: theme.spacing.xs }}
                        >
                          by {route.last_updated_by}
                        </Typography.Text>
                      )}
                    </Typography.Paragraph>
                  </div>
                </div>

                {/* Description */}
                <div css={{ marginBottom: theme.spacing.md }}>
                  <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                    <Typography.Text color="secondary" size="sm">
                      <FormattedMessage
                        defaultMessage="Description"
                        description="Route detail drawer > description label"
                      />
                    </Typography.Text>
                    {editingField !== 'description' && (
                      <Button
                        componentId="mlflow.routes.detail_drawer.edit_description"
                        icon={<PencilIcon />}
                        size="small"
                        onClick={() => {
                          setEditDescription(route.description || '');
                          setEditingField('description');
                        }}
                        css={{ padding: `0 ${theme.spacing.xs}px`, minWidth: 'auto' }}
                      />
                    )}
                  </div>
                  {editingField === 'description' ? (
                    <div css={{ marginTop: theme.spacing.xs }}>
                      <Input
                        componentId="mlflow.routes.detail_drawer.description_input"
                        value={editDescription}
                        onChange={(e) => setEditDescription(e.target.value)}
                        placeholder={intl.formatMessage({
                          defaultMessage: 'Add a description for this endpoint',
                          description: 'Endpoint description placeholder',
                        })}
                        autoFocus
                      />
                      <div css={{ display: 'flex', gap: theme.spacing.sm, marginTop: theme.spacing.xs }}>
                        <Button
                          componentId="mlflow.routes.detail_drawer.save_description"
                          type="primary"
                          size="small"
                          onClick={handleSaveDescription}
                          loading={isLoading}
                        >
                          <FormattedMessage defaultMessage="Save" description="Save button" />
                        </Button>
                        <Button
                          componentId="mlflow.routes.detail_drawer.cancel_description"
                          size="small"
                          onClick={handleCancelEditing}
                        >
                          <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <Typography.Paragraph css={{ marginTop: theme.spacing.xs, marginBottom: 0, fontSize: 13 }}>
                      {route.description || (
                        <Typography.Text color="secondary" size="sm">
                          <FormattedMessage
                            defaultMessage="No description"
                            description="No description placeholder"
                          />
                        </Typography.Text>
                      )}
                    </Typography.Paragraph>
                  )}
                </div>

                {/* Tags */}
                <div>
                  <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                    <Typography.Text color="secondary" size="sm">
                      <FormattedMessage defaultMessage="Tags" description="Route detail drawer > tags label" />
                    </Typography.Text>
                    {editingField !== 'tags' && (
                      <Button
                        componentId="mlflow.routes.detail_drawer.edit_tags"
                        icon={<PencilIcon />}
                        size="small"
                        onClick={() => {
                          const existingTags = route.tags
                            ? Array.isArray(route.tags)
                              ? route.tags
                              : Object.entries(route.tags).map(([key, value]) => ({ key, value }))
                            : [];
                          const initialTags =
                            existingTags.length > 0 ? [...existingTags, EMPTY_TAG_ENTITY] : [EMPTY_TAG_ENTITY];
                          tagsForm.reset({ tags: initialTags });
                          setEditingField('tags');
                        }}
                        css={{ padding: `0 ${theme.spacing.xs}px`, minWidth: 'auto' }}
                      />
                    )}
                  </div>
                  {editingField === 'tags' ? (
                    <div css={{ marginTop: theme.spacing.xs }}>
                      <TagAssignmentRoot {...tagsFieldArray}>
                        <TagAssignmentRow>
                          <TagAssignmentLabel>
                            <FormattedMessage defaultMessage="Key" description="Tag key label" />
                          </TagAssignmentLabel>
                          <TagAssignmentLabel>
                            <FormattedMessage defaultMessage="Value" description="Tag value label" />
                          </TagAssignmentLabel>
                        </TagAssignmentRow>

                        {tagsFieldArray.fields.map((field, index) => (
                          <TagAssignmentRow key={field.id}>
                            <TagAssignmentKey index={index} />
                            <TagAssignmentValue index={index} />
                            <TagAssignmentRemoveButton
                              componentId="mlflow.routes.detail_drawer.remove_tag"
                              index={index}
                            />
                          </TagAssignmentRow>
                        ))}
                      </TagAssignmentRoot>
                      <div css={{ display: 'flex', gap: theme.spacing.sm, marginTop: theme.spacing.xs }}>
                        <Button
                          componentId="mlflow.routes.detail_drawer.save_tags"
                          type="primary"
                          size="small"
                          onClick={handleSaveTags}
                          loading={isLoading}
                        >
                          <FormattedMessage defaultMessage="Save" description="Save button" />
                        </Button>
                        <Button
                          componentId="mlflow.routes.detail_drawer.cancel_tags"
                          size="small"
                          onClick={handleCancelEditing}
                        >
                          <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div
                      css={{ display: 'flex', gap: theme.spacing.sm, flexWrap: 'wrap', marginTop: theme.spacing.xs }}
                    >
                      {tagEntities.length > 0 ? (
                        tagEntities.map((tag) => <KeyValueTag key={tag.key} tag={tag} />)
                      ) : (
                        <Typography.Text color="secondary" size="sm">
                          <FormattedMessage defaultMessage="No tags" description="No tags placeholder" />
                        </Typography.Text>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Models Section */}
              <div
                css={{
                  padding: theme.spacing.lg,
                  borderRadius: theme.borders.borderRadiusMd,
                  backgroundColor: theme.colors.backgroundSecondary,
                  border: `1px solid ${theme.colors.border}`,
                }}
              >
                <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: theme.spacing.md }}>
                  <Typography.Text css={{ fontWeight: 500 }}>
                    <FormattedMessage
                      defaultMessage="Models ({count})"
                      description="Route detail drawer > models header"
                      values={{ count: route.models?.length || 0 }}
                    />
                  </Typography.Text>
                </div>

                {route.models && route.models.length > 0 ? (
                    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                      {route.models.map((model) => {
                        const modelSecret = secrets.find((s) => s.secret_id === model.secret_id);
                        return (
                          <div
                            key={model.model_id}
                            css={{
                              padding: theme.spacing.md,
                              borderRadius: theme.borders.borderRadiusSm,
                              border: `1px solid ${theme.colors.border}`,
                              backgroundColor: theme.colors.actionDefaultBackgroundHover,
                            }}
                          >
                            <ModelInfo
                              modelName={model.model_name}
                              secretName={modelSecret?.secret_name || model.secret_name}
                              secretMaskedValue={modelSecret?.masked_value}
                              provider={model.provider}
                              showSecret={!!modelSecret}
                            />

                            {/* Per-model action buttons */}
                            {selectedModelForEdit?.model_id !== model.model_id &&
                              selectedModelForModelChange?.model_id !== model.model_id && (
                                <div css={{ display: 'flex', gap: theme.spacing.sm, marginTop: theme.spacing.md }}>
                                  <Button
                                    componentId={`mlflow.routes.detail_drawer.change_model_${model.model_id}`}
                                    icon={<PencilIcon />}
                                    size="small"
                                    onClick={() => {
                                      setSelectedModelForModelChange(model);
                                      setNewModelName(model.model_name);
                                    }}
                                  >
                                    <FormattedMessage defaultMessage="Change Model" description="Change model button" />
                                  </Button>
                                  <Button
                                    componentId={`mlflow.routes.detail_drawer.change_secret_${model.model_id}`}
                                    icon={<PencilIcon />}
                                    size="small"
                                    onClick={() => setSelectedModelForEdit(model)}
                                  >
                                    <FormattedMessage defaultMessage="Change Secret" description="Change secret button" />
                                  </Button>
                                {/* Disabled for current phase - Add/Remove Model not yet implemented */}
                                {/* <Button
                                  componentId={`mlflow.routes.detail_drawer.remove_model_${model.model_id}`}
                                  icon={<TrashIcon />}
                                  danger
                                  size="small"
                                  onClick={() => setSelectedModelForRemoval(model)}
                                >
                                  <FormattedMessage defaultMessage="Remove Model" description="Remove model button" />
                                </Button> */}
                              </div>
                            )}

                            {/* Inline Change Secret Form */}
                            {selectedModelForEdit?.model_id === model.model_id && (
                              <div
                                css={{
                                  marginTop: theme.spacing.md,
                                  padding: theme.spacing.md,
                                  borderRadius: theme.borders.borderRadiusMd,
                                  border: `2px solid ${theme.colors.actionDefaultBorderDefault}`,
                                  backgroundColor: theme.colors.backgroundSecondary,
                                }}
                              >
                                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                                  {/* Secret source toggle */}
                                  <div>
                                    <FormUI.Label>
                                      <FormattedMessage
                                        defaultMessage="API Key Source"
                                        description="API key source label"
                                      />
                                    </FormUI.Label>
                                    <Radio.Group
                                      componentId="mlflow.routes.model_edit.secret_source"
                                      name="model-secret-source"
                                      value={modelEditSecretSource}
                                      onChange={(e) => {
                                        setModelEditSecretSource(e.target.value as SecretSource);
                                        setModelEditErrors({});
                                      }}
                                    >
                                      <Radio value="existing">
                                        <FormattedMessage
                                          defaultMessage="Use Existing Key"
                                          description="Use existing secret option"
                                        />
                                      </Radio>
                                      <Radio value="new">
                                        <FormattedMessage
                                          defaultMessage="Create New Key"
                                          description="Create new secret option"
                                        />
                                      </Radio>
                                    </Radio.Group>
                                  </div>

                                  {/* Existing secret selection */}
                                  {modelEditSecretSource === 'existing' && (
                                    <div>
                                      <DialogCombobox
                                        componentId="mlflow.routes.model_edit.secret"
                                        label={intl.formatMessage({
                                          defaultMessage: 'Select API Key',
                                          description: 'Select API key label',
                                        })}
                                        value={
                                          modelEditSelectedSecretId
                                            ? [
                                                modelEditCompatibleSecrets.find(
                                                  (s) => s.secret_id === modelEditSelectedSecretId,
                                                )?.secret_name || '',
                                              ]
                                            : []
                                        }
                                      >
                                        <DialogComboboxTrigger
                                          allowClear={false}
                                          placeholder={intl.formatMessage({
                                            defaultMessage: 'Select Existing API Key',
                                            description: 'Select existing API key placeholder',
                                          })}
                                        />
                                        <DialogComboboxContent>
                                          <DialogComboboxOptionList>
                                            {modelEditCompatibleSecrets.map((secret) => (
                                              <DialogComboboxOptionListSelectItem
                                                key={secret.secret_id}
                                                checked={modelEditSelectedSecretId === secret.secret_id}
                                                value={secret.secret_name}
                                                onChange={() => {
                                                  setModelEditSelectedSecretId(secret.secret_id);
                                                  setModelEditErrors((prev) => ({ ...prev, secretId: undefined }));
                                                }}
                                              >
                                                <Typography.Text>{secret.secret_name}</Typography.Text>
                                              </DialogComboboxOptionListSelectItem>
                                            ))}
                                          </DialogComboboxOptionList>
                                        </DialogComboboxContent>
                                      </DialogCombobox>
                                      {modelEditErrors.secretId && (
                                        <FormUI.Message type="error" message={modelEditErrors.secretId} />
                                      )}
                                    </div>
                                  )}

                                  {/* New secret creation */}
                                  {modelEditSecretSource === 'new' && (
                                    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                                      {/* Provider display (locked to model's provider) */}
                                      <div>
                                        <FormUI.Label>
                                          <FormattedMessage defaultMessage="Provider" description="Provider label" />
                                        </FormUI.Label>
                                        <Input
                                          componentId="mlflow.routes.model_edit.provider"
                                          value={modelEditProviderInfo?.label || model.provider}
                                          disabled
                                          css={{ backgroundColor: theme.colors.backgroundSecondary }}
                                        />
                                        <Typography.Text
                                          color="secondary"
                                          size="sm"
                                          css={{ display: 'block', marginTop: theme.spacing.xs }}
                                        >
                                          <FormattedMessage
                                            defaultMessage="Provider is locked to match the model's configuration"
                                            description="Provider locked message"
                                          />
                                        </Typography.Text>
                                      </div>

                                      <div>
                                        <FormUI.Label htmlFor="model-edit-new-secret-name">
                                          <FormattedMessage
                                            defaultMessage="Secret Name"
                                            description="New secret name label"
                                          />
                                          <span css={{ color: theme.colors.textValidationDanger }}> *</span>
                                        </FormUI.Label>
                                        <Input
                                          componentId="mlflow.routes.model_edit.new_secret_name"
                                          id="model-edit-new-secret-name"
                                          placeholder={intl.formatMessage({
                                            defaultMessage: 'e.g., my-openai-key',
                                            description: 'Secret name placeholder',
                                          })}
                                          value={modelEditNewSecretName}
                                          onChange={(e) => {
                                            setModelEditNewSecretName(e.target.value);
                                            setModelEditErrors((prev) => ({ ...prev, secretName: undefined }));
                                          }}
                                          validationState={modelEditErrors.secretName ? 'error' : undefined}
                                        />
                                        {modelEditErrors.secretName && (
                                          <FormUI.Message type="error" message={modelEditErrors.secretName} />
                                        )}
                                      </div>

                                      <div>
                                        <FormUI.Label htmlFor="model-edit-new-secret-value">
                                          <FormattedMessage defaultMessage="API Key" description="New API key label" />
                                          <span css={{ color: theme.colors.textValidationDanger }}> *</span>
                                        </FormUI.Label>
                                        <MaskedApiKeyInput
                                          value={modelEditNewSecretValue}
                                          onChange={(value) => {
                                            setModelEditNewSecretValue(value);
                                            setModelEditErrors((prev) => ({ ...prev, secretValue: undefined }));
                                          }}
                                          placeholder={intl.formatMessage({
                                            defaultMessage: 'Enter your API key',
                                            description: 'API key placeholder',
                                          })}
                                          id="model-edit-new-secret-value"
                                          componentId="mlflow.routes.model_edit.new_secret_value"
                                        />
                                        {modelEditErrors.secretValue && (
                                          <FormUI.Message type="error" message={modelEditErrors.secretValue} />
                                        )}
                                      </div>

                                      {/* Auth configuration fields */}
                                      <AuthConfigFields
                                        fields={modelEditProviderInfo?.authConfigFields || []}
                                        values={modelEditAuthConfig}
                                        onChange={(name, value) => {
                                          setModelEditAuthConfig((prev) => ({ ...prev, [name]: value }));
                                        }}
                                        componentIdPrefix="mlflow.routes.model_edit.auth_config"
                                      />
                                    </div>
                                  )}

                                  {/* Action buttons */}
                                  <div
                                    css={{
                                      display: 'flex',
                                      gap: theme.spacing.sm,
                                      justifyContent: 'flex-end',
                                    }}
                                  >
                                    <Button
                                      componentId="mlflow.routes.model_edit.cancel"
                                      onClick={handleCancelModelEdit}
                                      disabled={isUpdatingModel}
                                    >
                                      <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
                                    </Button>
                                    <Button
                                      componentId="mlflow.routes.model_edit.save"
                                      type="primary"
                                      onClick={handleModelSecretUpdate}
                                      loading={isUpdatingModel}
                                    >
                                      <FormattedMessage defaultMessage="Update Secret" description="Update secret button" />
                                    </Button>
                                  </div>
                                </div>
                              </div>
                            )}

                            {/* Inline Change Model Form */}
                            {selectedModelForModelChange?.model_id === model.model_id && (
                              <div
                                css={{
                                  marginTop: theme.spacing.md,
                                  padding: theme.spacing.md,
                                  borderRadius: theme.borders.borderRadiusMd,
                                  border: `2px solid ${theme.colors.actionDefaultBorderDefault}`,
                                  backgroundColor: theme.colors.actionDefaultBackgroundHover,
                                }}
                              >
                                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                                  <div>
                                    <FormUI.Label htmlFor={`model-change-select-${model.model_id}`}>
                                      <FormattedMessage
                                        defaultMessage="Select Model"
                                        description="Select model label"
                                      />
                                    </FormUI.Label>
                                    <SimpleSelect
                                      componentId={`mlflow.routes.model_change.model_select_${model.model_id}`}
                                      id={`model-change-select-${model.model_id}`}
                                      label=""
                                      value={newModelName}
                                      onChange={(e) => setNewModelName(e.target.value)}
                                      placeholder={intl.formatMessage({
                                        defaultMessage: 'Select a model from the list',
                                        description: 'Select model placeholder',
                                      })}
                                    >
                                      {PROVIDERS.find((p) => p.value === model.provider)?.commonModels?.map((m) => (
                                        <SimpleSelectOption key={m.id} value={m.id}>
                                          {m.name}
                                        </SimpleSelectOption>
                                      ))}
                                    </SimpleSelect>
                                  </div>

                                  <div css={{ textAlign: 'center', color: theme.colors.textSecondary }}>
                                    <Typography.Text size="sm">
                                      <FormattedMessage
                                        defaultMessage="— or —"
                                        description="Model selection separator"
                                      />
                                    </Typography.Text>
                                  </div>

                                  <div>
                                    <FormUI.Label htmlFor={`model-change-input-${model.model_id}`}>
                                      <FormattedMessage
                                        defaultMessage="Custom Model Name"
                                        description="Custom model name label"
                                      />
                                    </FormUI.Label>
                                    <Input
                                      componentId={`mlflow.routes.model_change.model_input_${model.model_id}`}
                                      id={`model-change-input-${model.model_id}`}
                                      placeholder={intl.formatMessage({
                                        defaultMessage: 'e.g., gpt-4o-2024-11-20',
                                        description: 'Custom model name placeholder',
                                      })}
                                      value={newModelName}
                                      onChange={(e) => setNewModelName(e.target.value)}
                                    />
                                    <Typography.Text
                                      color="secondary"
                                      size="sm"
                                      css={{ display: 'block', marginTop: theme.spacing.xs }}
                                    >
                                      <FormattedMessage
                                        defaultMessage="Enter a custom model name or select from the list above"
                                        description="Custom model name help text"
                                      />
                                    </Typography.Text>
                                  </div>

                                  {/* Action buttons */}
                                  <div
                                    css={{
                                      display: 'flex',
                                      gap: theme.spacing.sm,
                                      justifyContent: 'flex-end',
                                    }}
                                  >
                                    <Button
                                      componentId="mlflow.routes.model_change.cancel"
                                      onClick={() => {
                                        setSelectedModelForModelChange(null);
                                        setNewModelName('');
                                      }}
                                    >
                                      <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
                                    </Button>
                                    <Button
                                      componentId="mlflow.routes.model_change.save"
                                      type="primary"
                                      onClick={async () => {
                                        if (!route) return;
                                        try {
                                          await updateModelMutation({
                                            endpoint_id: route.endpoint_id,
                                            model_id: model.model_id,
                                            model_name: newModelName,
                                          });
                                          setSelectedModelForModelChange(null);
                                          setNewModelName('');
                                        } catch (error) {
                                          console.error('Failed to update model:', error);
                                        }
                                      }}
                                      disabled={!newModelName || newModelName === model.model_name}
                                      loading={isUpdatingModel}
                                    >
                                      <FormattedMessage defaultMessage="Update Model" description="Update model button" />
                                    </Button>
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}

                      {/* Disabled for current phase - Add Model not yet implemented */}
                      {/* <Button
                        componentId="mlflow.routes.detail_drawer.add_model"
                        icon={<PlusIcon />}
                        css={{ marginTop: theme.spacing.sm }}
                        onClick={() => setShowAddModelDialog(true)}
                      >
                        <FormattedMessage defaultMessage="Add Model" description="Add model button" />
                      </Button> */}
                    </div>
                  ) : (
                    <div>
                      <Typography.Text color="secondary" size="sm" css={{ display: 'block', marginBottom: theme.spacing.sm }}>
                        <FormattedMessage
                          defaultMessage="No models configured"
                          description="Route detail drawer > no models"
                        />
                      </Typography.Text>
                      {/* Disabled for current phase - Add Model not yet implemented */}
                      {/* <Button
                        componentId="mlflow.routes.detail_drawer.add_first_model"
                        icon={<PlusIcon />}
                        onClick={() => setShowAddModelDialog(true)}
                      >
                        <FormattedMessage defaultMessage="Add Model" description="Add model button" />
                      </Button> */}
                    </div>
                  )}
              </div>

              {/* Resources Using This Endpoint */}
              <div>
                <div
                  css={{
                    borderRadius: theme.borders.borderRadiusMd,
                    border: `1px solid ${theme.colors.border}`,
                    overflow: 'hidden',
                  }}
                >
                  <div
                    css={{
                      display: 'flex',
                      alignItems: 'center',
                      padding: theme.spacing.md,
                      backgroundColor: theme.colors.backgroundSecondary,
                      cursor: 'pointer',
                      '&:hover': {
                        backgroundColor: theme.colors.actionDefaultBackgroundHover,
                      },
                    }}
                    onClick={() => setShowResourceUsage(!showResourceUsage)}
                  >
                    {showResourceUsage ? (
                      <ChevronDownIcon
                        css={{ fontSize: 16, color: theme.colors.textSecondary, marginRight: theme.spacing.sm }}
                      />
                    ) : (
                      <ChevronRightIcon
                        css={{ fontSize: 16, color: theme.colors.textSecondary, marginRight: theme.spacing.sm }}
                      />
                    )}
                    <Typography.Text css={{ fontWeight: 500, fontSize: theme.typography.fontSizeBase }}>
                      <FormattedMessage
                        defaultMessage="Resources Using This Endpoint ({count})"
                        description="Route detail drawer > resources using endpoint header"
                        values={{ count: externalResourceBindings.length }}
                      />
                    </Typography.Text>
                  </div>

                  {showResourceUsage && (
                    <div css={{ padding: theme.spacing.md, backgroundColor: theme.colors.backgroundPrimary }}>
                      {externalResourceBindings.length === 0 ? (
                        <div
                          css={{
                            textAlign: 'center',
                            padding: `${theme.spacing.lg}px 0`,
                          }}
                        >
                          <Typography.Text color="secondary">
                            <FormattedMessage
                              defaultMessage="No external resources are currently using this endpoint"
                              description="Route detail drawer > no external resources"
                            />
                          </Typography.Text>
                        </div>
                      ) : (
                        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                          {externalResourceBindings.map((binding) => (
                            <div
                              key={binding.binding_id}
                              css={{
                                padding: theme.spacing.md,
                                borderRadius: theme.borders.borderRadiusSm,
                                border: `1px solid ${theme.colors.border}`,
                                backgroundColor: theme.colors.backgroundSecondary,
                              }}
                            >
                              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                                {/* Resource Type Badge */}
                                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                                  <Tag componentId="mlflow.routes.detail_drawer.resource_type_tag">
                                    <Typography.Text size="sm">
                                      {formatResourceType(binding.resource_type)}
                                    </Typography.Text>
                                  </Tag>
                                </div>

                                {/* Resource Name */}
                                <div>
                                  <Typography.Text color="secondary" size="sm">
                                    <FormattedMessage defaultMessage="Name" description="Resource name label" />
                                  </Typography.Text>
                                  <Typography.Paragraph
                                    css={{
                                      marginTop: theme.spacing.xs,
                                      marginBottom: 0,
                                      fontWeight: 500,
                                      fontSize: 14,
                                    }}
                                  >
                                    {getResourceName(binding)}
                                  </Typography.Paragraph>
                                </div>

                                {/* Resource ID (if different from name) */}
                                {binding.route_name && binding.resource_id !== binding.route_name && (
                                  <div>
                                    <Typography.Text color="secondary" size="sm">
                                      <FormattedMessage defaultMessage="ID" description="Resource ID label" />
                                    </Typography.Text>
                                    <Typography.Text
                                      css={{
                                        display: 'block',
                                        marginTop: theme.spacing.xs,
                                        fontFamily: 'monospace',
                                        fontSize: theme.typography.fontSizeSm,
                                        color: theme.colors.textSecondary,
                                      }}
                                    >
                                      {binding.resource_id}
                                    </Typography.Text>
                                  </div>
                                )}

                                {/* Environment Variable */}
                                {binding.field_name && (
                                  <div>
                                    <Typography.Text color="secondary" size="sm">
                                      <FormattedMessage
                                        defaultMessage="Environment Variable"
                                        description="Resource environment variable label"
                                      />
                                    </Typography.Text>
                                    <Typography.Text
                                      css={{
                                        display: 'block',
                                        marginTop: theme.spacing.xs,
                                        fontFamily: 'monospace',
                                        fontSize: theme.typography.fontSizeSm,
                                        color: theme.colors.textSecondary,
                                      }}
                                    >
                                      {binding.field_name}
                                    </Typography.Text>
                                  </div>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Delete Endpoint */}
              <div>
                {/* Delete Endpoint Button */}
                <div css={{ marginBottom: theme.spacing.md }}>
                  <Button
                    componentId="mlflow.routes.detail_drawer.delete_route_toggle"
                    icon={<TrashIcon />}
                    danger
                    type={selectedOperation === 'deleteRoute' ? 'primary' : undefined}
                    onClick={() => setSelectedOperation(selectedOperation === 'deleteRoute' ? null : 'deleteRoute')}
                  >
                    <FormattedMessage defaultMessage="Delete Endpoint" description="Delete endpoint button" />
                  </Button>
                </div>

                {/* Change Secret section removed - now per-model */}

                {/* Delete Route Content */}
                {selectedOperation === 'deleteRoute' && (
                  <div
                    css={{
                      padding: theme.spacing.md,
                      borderRadius: theme.borders.borderRadiusMd,
                      border: `2px solid ${theme.colors.border}`,
                      backgroundColor: theme.colors.backgroundPrimary,
                    }}
                  >
                    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                      {/* Secret source toggle */}
                      <div>
                        <FormUI.Label>
                          <FormattedMessage defaultMessage="API Key Source" description="API key source label" />
                        </FormUI.Label>
                        <Radio.Group
                          componentId="mlflow.routes.detail_drawer.secret_source"
                          name="secret-source"
                          value={secretSource}
                          onChange={(e) => {
                            setSecretSource(e.target.value as SecretSource);
                            setErrors({});
                          }}
                        >
                          <Radio value="existing">
                            <FormattedMessage
                              defaultMessage="Use Existing Key"
                              description="Use existing secret option"
                            />
                          </Radio>
                          <Radio value="new">
                            <FormattedMessage defaultMessage="Create New Key" description="Create new secret option" />
                          </Radio>
                        </Radio.Group>
                      </div>

                      {/* Existing secret selection */}
                      {secretSource === 'existing' && (
                        <div>
                          <DialogCombobox
                            componentId="mlflow.routes.detail_drawer.secret"
                            label={intl.formatMessage({
                              defaultMessage: 'Select API Key',
                              description: 'Select API key label',
                            })}
                            value={
                              selectedSecretId
                                ? [compatibleSecrets.find((s) => s.secret_id === selectedSecretId)?.secret_name || '']
                                : []
                            }
                          >
                            <DialogComboboxTrigger allowClear={false} placeholder="Select Existing API Key" />
                            <DialogComboboxContent>
                              <DialogComboboxOptionList>
                                {compatibleSecrets.map((secret) => (
                                  <DialogComboboxOptionListSelectItem
                                    key={secret.secret_id}
                                    checked={selectedSecretId === secret.secret_id}
                                    value={secret.secret_name}
                                    onChange={() => {
                                      setSelectedSecretId(secret.secret_id);
                                      setErrors((prev) => ({ ...prev, secretId: undefined }));
                                    }}
                                  >
                                    <Typography.Text>{secret.secret_name}</Typography.Text>
                                  </DialogComboboxOptionListSelectItem>
                                ))}
                              </DialogComboboxOptionList>
                            </DialogComboboxContent>
                          </DialogCombobox>
                          {errors.secretId && <FormUI.Message type="error" message={errors.secretId} />}
                        </div>
                      )}

                      {/* New secret creation */}
                      {secretSource === 'new' && (
                        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                          {/* Provider selector */}
                          <div>
                            <FormUI.Label htmlFor="route-detail-provider">
                              <FormattedMessage defaultMessage="Provider" description="Provider select label" />
                              <span css={{ color: theme.colors.textValidationDanger }}> *</span>
                            </FormUI.Label>
                            <SimpleSelect
                              componentId="mlflow.routes.detail_drawer.provider"
                              id="route-detail-provider"
                              label=""
                              value={newSecretProvider}
                              onChange={(e) => {
                                setNewSecretProvider(e.target.value);
                                setAuthConfig({});
                                setErrors((prev) => ({ ...prev, provider: undefined }));
                              }}
                              validationState={errors.provider ? 'error' : undefined}
                              placeholder={intl.formatMessage({
                                defaultMessage: 'Select provider',
                                description: 'Provider select placeholder',
                              })}
                            >
                              {PROVIDERS.map((provider) => (
                                <SimpleSelectOption key={provider.value} value={provider.value}>
                                  {provider.label}
                                </SimpleSelectOption>
                              ))}
                            </SimpleSelect>
                            {errors.provider && <FormUI.Message type="error" message={errors.provider} />}
                          </div>

                          <div>
                            <FormUI.Label htmlFor="route-detail-new-secret-name">
                              <FormattedMessage defaultMessage="Secret Name" description="New secret name label" />
                              <span css={{ color: theme.colors.textValidationDanger }}> *</span>
                            </FormUI.Label>
                            <Input
                              componentId="mlflow.routes.detail_drawer.new_secret_name"
                              id="route-detail-new-secret-name"
                              placeholder={intl.formatMessage({
                                defaultMessage: 'e.g., my-openai-key',
                                description: 'Secret name placeholder',
                              })}
                              value={newSecretName}
                              onChange={(e) => {
                                setNewSecretName(e.target.value);
                                setErrors((prev) => ({ ...prev, secretName: undefined }));
                              }}
                              validationState={errors.secretName ? 'error' : undefined}
                            />
                            {errors.secretName && <FormUI.Message type="error" message={errors.secretName} />}
                          </div>

                          <div>
                            <FormUI.Label htmlFor="route-detail-new-secret-value">
                              <FormattedMessage defaultMessage="API Key" description="New API key label" />
                              <span css={{ color: theme.colors.textValidationDanger }}> *</span>
                            </FormUI.Label>
                            <MaskedApiKeyInput
                              value={newSecretValue}
                              onChange={(value) => {
                                setNewSecretValue(value);
                                setErrors((prev) => ({ ...prev, secretValue: undefined }));
                              }}
                              placeholder={intl.formatMessage({
                                defaultMessage: 'Enter your API key',
                                description: 'API key placeholder',
                              })}
                              id="route-detail-new-secret-value"
                              componentId="mlflow.routes.detail_drawer.new_secret_value"
                            />
                            {errors.secretValue && <FormUI.Message type="error" message={errors.secretValue} />}
                          </div>

                          {/* Auth configuration fields */}
                          <AuthConfigFields
                            fields={selectedProviderInfo?.authConfigFields || []}
                            values={authConfig}
                            onChange={(name, value) => {
                              setAuthConfig((prev) => ({ ...prev, [name]: value }));
                            }}
                            componentIdPrefix="mlflow.routes.detail_drawer.auth_config"
                          />
                        </div>
                      )}

                      {/* Action buttons */}
                      <div
                        css={{
                          display: 'flex',
                          gap: theme.spacing.sm,
                          justifyContent: 'flex-end',
                          marginTop: theme.spacing.md,
                        }}
                      >
                        <Button
                          componentId="mlflow.routes.detail_drawer.update_cancel"
                          onClick={() => setSelectedOperation(null)}
                        >
                          <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
                        </Button>
                        <Button
                          componentId="mlflow.routes.detail_drawer.update_submit"
                          type="primary"
                          onClick={handleUpdate}
                          loading={isLoading}
                          disabled={!hasChanges}
                        >
                          <FormattedMessage defaultMessage="Update Endpoint" description="Update endpoint button" />
                        </Button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Delete Route Content */}
                {selectedOperation === 'deleteRoute' && (
                  <div
                    css={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 1fr',
                      gap: theme.spacing.lg,
                      padding: theme.spacing.md,
                      borderRadius: theme.borders.borderRadiusMd,
                      border: `2px solid ${theme.colors.borderValidationDanger}`,
                      backgroundColor: theme.colors.backgroundPrimary,
                    }}
                  >
                    {/* Left side: Route info */}
                    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                      <div>
                        <Typography.Title level={5} css={{ margin: 0, marginBottom: theme.spacing.sm }}>
                          <FormattedMessage
                            defaultMessage="Endpoint Configuration"
                            description="Endpoint detail drawer > endpoint configuration title"
                          />
                        </Typography.Title>
                        <Typography.Text color="secondary" size="sm">
                          <FormattedMessage
                            defaultMessage="This endpoint will be permanently deleted along with its configuration."
                            description="Endpoint detail drawer > delete endpoint warning"
                          />
                        </Typography.Text>
                      </div>

                      {route.models && route.models.length > 0 && (
                        <div
                          css={{
                            padding: theme.spacing.md,
                            borderRadius: theme.borders.borderRadiusMd,
                            border: `1px solid ${theme.colors.border}`,
                            backgroundColor: theme.colors.backgroundSecondary,
                          }}
                        >
                          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                            <div>
                              <Typography.Text color="secondary" size="sm">
                                <FormattedMessage
                                  defaultMessage="Models"
                                  description="Route detail drawer > models label"
                                />
                              </Typography.Text>
                              <Typography.Paragraph
                                css={{ marginTop: theme.spacing.xs, marginBottom: 0, fontWeight: 500 }}
                              >
                                <FormattedMessage
                                  defaultMessage="{count, plural, one {# model} other {# models}} configured"
                                  description="Route detail drawer > model count"
                                  values={{ count: route.models.length }}
                                />
                              </Typography.Paragraph>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Right side: Delete confirmation */}
                    <div
                      css={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: theme.spacing.md,
                        justifyContent: 'center',
                      }}
                    >
                      <div>
                        <FormUI.Label htmlFor="delete-route-confirmation-input">
                          <FormattedMessage
                            defaultMessage="To confirm deletion, type the endpoint name:"
                            description="Endpoint detail drawer > delete confirmation label"
                          />
                        </FormUI.Label>
                        <Input
                          componentId="mlflow.routes.detail_drawer.delete_confirmation_input"
                          id="delete-route-confirmation-input"
                          placeholder={route.name || route.endpoint_id}
                          value={routeDeleteConfirmation}
                          onChange={(e) => setRouteDeleteConfirmation(e.target.value)}
                          autoComplete="off"
                          css={{ marginTop: theme.spacing.xs }}
                        />
                      </div>
                      <Button
                        componentId="mlflow.routes.detail_drawer.delete_button"
                        danger
                        icon={<TrashIcon />}
                        disabled={routeDeleteConfirmation !== (route.name || route.endpoint_id)}
                        onClick={handleDelete}
                      >
                        <FormattedMessage
                          defaultMessage="Delete Endpoint"
                          description="Endpoint detail drawer > delete endpoint button"
                        />
                      </Button>
                      <Typography.Text color="secondary" size="sm">
                        <FormattedMessage
                          defaultMessage="This action cannot be undone. Secrets associated with models will remain and can be reused."
                          description="Route detail drawer > delete route note"
                        />
                      </Typography.Text>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </Drawer.Content>
      </Drawer.Root>
    </>
  );
};
