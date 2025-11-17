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
import {
  TagAssignmentRoot,
  TagAssignmentRow,
  TagAssignmentLabel,
  TagAssignmentKey,
  TagAssignmentValue,
  TagAssignmentRemoveButton,
  useTagAssignmentForm,
} from '@databricks/web-shared/unified-tagging';
import type { Route } from '../types';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import { Descriptions } from '@mlflow/mlflow/src/common/components/Descriptions';
import { KeyValueTag } from '@mlflow/mlflow/src/common/components/KeyValueTag';
import { ProviderBadge } from './ProviderBadge';
import { useListSecrets } from '../hooks/useListSecrets';
import { useListBindings } from '../hooks/useListBindings';
import { MaskedApiKeyInput } from './MaskedApiKeyInput';
import { AuthConfigFields } from './AuthConfigFields';
import { PROVIDERS } from './routeConstants';

const EMPTY_TAG_ENTITY = { key: '', value: '' };
const EMPTY_TAG_ARRAY: { key: string; value: string }[] = [];

export interface RouteDetailDrawerProps {
  route: Route | null;
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
  onDelete?: (route: Route) => void;
}

type ManagementOperation = 'changeSecret' | 'editDetails' | 'deleteRoute' | null;
type SecretSource = 'existing' | 'new';

export const RouteDetailDrawer = ({ route, open, onClose, onUpdate, onDelete }: RouteDetailDrawerProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

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

  // Edit details state
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

  // Fetch data
  const { secrets = [] } = useListSecrets({ enabled: open });
  const { bindings = [] } = useListBindings({
    secretId: route?.secret_id || '',
    enabled: open && !!route?.secret_id,
  });

  // Get the current secret
  const currentSecret = useMemo(() => {
    return secrets.find((s) => s.secret_id === route?.secret_id);
  }, [secrets, route?.secret_id]);

  // Get the binding for this specific route to show the environment variable
  const routeBinding = useMemo(() => {
    if (!route) return null;
    return bindings.find((b) => b.route_id === route.route_id);
  }, [bindings, route]);

  // Reset when route changes
  useEffect(() => {
    if (route && open) {
      setSelectedOperation(null);
      setShowResourceUsage(false);
      setRouteDeleteConfirmation('');
      setSecretSource('existing');
      setSelectedSecretId('');
      setNewSecretName('');
      setNewSecretValue('');
      const currentSecret = secrets.find((s) => s.secret_id === route.secret_id);
      setNewSecretProvider(route.provider || currentSecret?.provider || '');
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
      const initialTags = existingTags.length > 0
        ? [...existingTags, EMPTY_TAG_ENTITY]
        : [EMPTY_TAG_ENTITY];
      tagsForm.reset({ tags: initialTags });
    }
  }, [route?.route_id, open, secrets]);

  // Convert tags to array format
  const tagEntities = route?.tags
    ? Array.isArray(route.tags)
      ? route.tags
      : Object.entries(route.tags).map(([key, value]) => ({ key, value }))
    : [];

  // Format resource type for display
  const formatResourceType = (binding: any) => {
    if (binding.route_id) {
      return 'Route';
    }
    return binding.resource_type
      .split('_')
      .map((word: string) => word.charAt(0) + word.slice(1).toLowerCase())
      .join(' ');
  };

  // Get display name for resource
  const getResourceDisplay = (binding: any) => {
    if (binding.route_id) {
      return binding.route_name || binding.route_id;
    }
    return binding.resource_id;
  };

  // Filter secrets to only show those from the same provider
  const compatibleSecrets = useMemo(() => {
    if (!route) return [];
    if (!route.provider) return secrets;

    const routeProvider = route.provider;
    return secrets.filter((secret) => {
      if (secret.provider) {
        return secret.provider.toLowerCase() === routeProvider.toLowerCase();
      }
      const secretNameLower = secret.secret_name.toLowerCase();
      const providerLower = routeProvider.toLowerCase();
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
  }, [secrets, route]);

  // Get the selected provider info with auth config fields
  const selectedProviderInfo = useMemo(() => {
    return PROVIDERS.find((p) => p.value === newSecretProvider);
  }, [newSecretProvider]);

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
      } else if (selectedSecretId === route.secret_id) {
        setSelectedOperation(null);
        return;
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
        await onUpdate?.(route.route_id, { secret_id: selectedSecretId });
      } else {
        await onUpdate?.(route.route_id, {
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

  // Handle save details
  const handleSaveDetails = async () => {
    if (!route || !onUpdate) return;

    setIsLoading(true);
    try {
      const tags = tagsForm.getValues().tags.filter((tag) => tag.key.trim() || tag.value.trim());
      const route_tags = tags.length > 0 ? JSON.stringify(tags) : undefined;

      await onUpdate(route.route_id, {
        route_description: editDescription.trim() || undefined,
        route_tags,
      });

      setSelectedOperation(null);
    } catch (err) {
      console.error('Failed to update route details:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const hasChanges =
    secretSource === 'new'
      ? newSecretName.trim().length > 0 || newSecretValue.trim().length > 0
      : selectedSecretId !== '' && selectedSecretId !== route?.secret_id;

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
            <FormattedMessage defaultMessage="Route Details" description="Route detail drawer > drawer title" />
          </div>
        }
      >
          {route && (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
              {/* Header section with name, provider, and model */}
              <div
                css={{
                  padding: theme.spacing.md,
                  borderRadius: theme.borders.borderRadiusMd,
                  backgroundColor: theme.colors.backgroundSecondary,
                  border: `1px solid ${theme.colors.border}`,
                }}
              >
                <div
                  css={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginBottom: theme.spacing.sm,
                  }}
                >
                  <Typography.Title level={3} css={{ margin: 0 }}>
                    {route.name || route.route_id}
                  </Typography.Title>
                  <ProviderBadge provider={route.provider} />
                </div>

                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                  <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                    <Typography.Text color="secondary" size="sm">
                      <FormattedMessage defaultMessage="Model:" description="Route detail drawer > model label" />
                    </Typography.Text>
                    <Tag componentId="mlflow.routes.detail_drawer.model_tag">
                      <Typography.Text>{route.model_name}</Typography.Text>
                    </Tag>
                  </div>

                  {route.description && (
                    <div>
                      <Typography.Text color="secondary" size="sm">
                        <FormattedMessage
                          defaultMessage="Description:"
                          description="Route detail drawer > description label"
                        />
                      </Typography.Text>
                      <Typography.Paragraph css={{ marginTop: theme.spacing.xs, marginBottom: 0 }}>
                        {route.description}
                      </Typography.Paragraph>
                    </div>
                  )}

                  {tagEntities.length > 0 && (
                    <div>
                      <Typography.Text color="secondary" size="sm">
                        <FormattedMessage defaultMessage="Tags:" description="Route detail drawer > tags label" />
                      </Typography.Text>
                      <div css={{ display: 'flex', gap: theme.spacing.sm, flexWrap: 'wrap', marginTop: theme.spacing.xs }}>
                        {tagEntities.map((tag) => (
                          <KeyValueTag key={tag.key} tag={tag} />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Metadata section */}
              <div>
                <Typography.Title level={4} css={{ margin: 0, marginBottom: theme.spacing.md }}>
                  <FormattedMessage
                    defaultMessage="Configuration"
                    description="Route detail drawer > configuration section title"
                  />
                </Typography.Title>
                <Descriptions columns={1}>
                  {route.created_by && (
                    <Descriptions.Item
                      label={
                        <FormattedMessage
                          defaultMessage="Created By"
                          description="Route detail drawer > created by label"
                        />
                      }
                    >
                      <Typography.Text>{route.created_by}</Typography.Text>
                    </Descriptions.Item>
                  )}
                  <Descriptions.Item
                    label={
                      <FormattedMessage
                        defaultMessage="Created At"
                        description="Route detail drawer > created at label"
                      />
                    }
                  >
                    <Typography.Text>{Utils.formatTimestamp(route.created_at)}</Typography.Text>
                  </Descriptions.Item>
                  {route.last_updated_by && (
                    <Descriptions.Item
                      label={
                        <FormattedMessage
                          defaultMessage="Last Updated By"
                          description="Route detail drawer > last updated by label"
                        />
                      }
                    >
                      <Typography.Text>{route.last_updated_by}</Typography.Text>
                    </Descriptions.Item>
                  )}
                  <Descriptions.Item
                    label={
                      <FormattedMessage
                        defaultMessage="Last Updated"
                        description="Route detail drawer > last updated label"
                      />
                    }
                  >
                    <Typography.Text>{Utils.formatTimestamp(route.last_updated_at)}</Typography.Text>
                  </Descriptions.Item>
                </Descriptions>
              </div>

              {/* Assigned Secret section */}
              <div>
                <Typography.Title level={4} css={{ marginBottom: theme.spacing.md }}>
                  <FormattedMessage
                    defaultMessage="Assigned Secret"
                    description="Route detail drawer > assigned secret section title"
                  />
                </Typography.Title>

                {currentSecret && (
                  <div
                    css={{
                      padding: theme.spacing.md,
                      borderRadius: theme.borders.borderRadiusMd,
                      border: `1px solid ${theme.colors.border}`,
                      backgroundColor: theme.colors.backgroundSecondary,
                    }}
                  >
                    {/* Secret Details */}
                      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                          <div>
                            <Typography.Text color="secondary" size="sm">
                              <FormattedMessage
                                defaultMessage="Secret Name"
                                description="Route detail drawer > secret name label"
                              />
                            </Typography.Text>
                            <Typography.Paragraph css={{ marginTop: theme.spacing.xs, marginBottom: 0, fontWeight: 500 }}>
                              {currentSecret.secret_name}
                            </Typography.Paragraph>
                          </div>
                          <div>
                            <Typography.Text color="secondary" size="sm">
                              <FormattedMessage
                                defaultMessage="Masked Value"
                                description="Route detail drawer > masked value label"
                              />
                            </Typography.Text>
                            <Typography.Paragraph
                              css={{ marginTop: theme.spacing.xs, marginBottom: 0, fontFamily: 'monospace' }}
                            >
                              {currentSecret.masked_value}
                            </Typography.Paragraph>
                          </div>
                          {currentSecret.provider && (
                            <div>
                              <Typography.Text color="secondary" size="sm">
                                <FormattedMessage
                                  defaultMessage="Provider"
                                  description="Route detail drawer > provider label"
                                />
                              </Typography.Text>
                              <div css={{ marginTop: theme.spacing.xs }}>
                                <ProviderBadge provider={currentSecret.provider} />
                              </div>
                            </div>
                          )}
                          {routeBinding?.field_name && (
                            <div>
                              <Typography.Text color="secondary" size="sm">
                                <FormattedMessage
                                  defaultMessage="Environment Variable"
                                  description="Route detail drawer > environment variable label"
                                />
                              </Typography.Text>
                              <Typography.Paragraph
                                css={{
                                  marginTop: theme.spacing.xs,
                                  marginBottom: 0,
                                  fontFamily: 'monospace',
                                  fontWeight: 500,
                                  padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                                  backgroundColor: theme.colors.backgroundSecondary,
                                  borderRadius: theme.borders.borderRadiusSm,
                                  display: 'inline-block',
                                }}
                              >
                                {routeBinding.field_name}
                              </Typography.Paragraph>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                )}
              </div>

              {/* Resources Using This Secret - moved here for context */}
              {bindings.length > 0 && (
                <div>
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
                            padding: theme.spacing.sm,
                            backgroundColor: theme.colors.backgroundSecondary,
                            cursor: 'pointer',
                            '&:hover': {
                              backgroundColor: theme.colors.actionDefaultBackgroundHover,
                            },
                          }}
                          onClick={() => setShowResourceUsage(!showResourceUsage)}
                        >
                          {showResourceUsage ? (
                            <ChevronDownIcon css={{ fontSize: 16, color: theme.colors.textSecondary, marginRight: theme.spacing.xs }} />
                          ) : (
                            <ChevronRightIcon css={{ fontSize: 16, color: theme.colors.textSecondary, marginRight: theme.spacing.xs }} />
                          )}
                          <Typography.Text css={{ fontWeight: 500, fontSize: theme.typography.fontSizeSm }}>
                            <FormattedMessage
                              defaultMessage="Resources Using This Secret ({count})"
                              description="Route detail drawer > resources using secret header"
                              values={{ count: bindings.length }}
                            />
                          </Typography.Text>
                        </div>

                        {showResourceUsage && (
                          <div css={{ padding: theme.spacing.md, backgroundColor: theme.colors.backgroundPrimary }}>
                            {bindings.length === 0 ? (
                              <Typography.Text color="secondary" size="sm">
                                <FormattedMessage
                                  defaultMessage="No other resources are using this secret"
                                  description="Route detail drawer > no other resources"
                                />
                              </Typography.Text>
                            ) : (
                              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
                                {bindings.map((binding) => (
                                  <div
                                    key={binding.binding_id}
                                    css={{
                                      padding: theme.spacing.sm,
                                      borderRadius: theme.borders.borderRadiusSm,
                                      border: `1px solid ${theme.colors.border}`,
                                      backgroundColor: binding.route_id === route.route_id
                                        ? theme.colors.backgroundValidationWarning
                                        : theme.colors.backgroundSecondary,
                                    }}
                                  >
                                    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                                      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                                        <Typography.Text css={{ fontWeight: 500, fontSize: theme.typography.fontSizeSm }}>
                                          {formatResourceType(binding)}
                                        </Typography.Text>
                                        {binding.route_id === route.route_id && (
                                          <Tag componentId="mlflow.routes.detail_drawer.current_route_tag">
                                            <Typography.Text size="sm">
                                              <FormattedMessage
                                                defaultMessage="This Route"
                                                description="Route detail drawer > current route tag"
                                              />
                                            </Typography.Text>
                                          </Tag>
                                        )}
                                      </div>
                                      <Typography.Text
                                        css={{
                                          fontFamily: 'monospace',
                                          fontSize: theme.typography.fontSizeSm,
                                          color: theme.colors.textSecondary,
                                        }}
                                      >
                                        {getResourceDisplay(binding)}
                                      </Typography.Text>
                                      <Typography.Text
                                        css={{
                                          fontFamily: 'monospace',
                                          fontSize: theme.typography.fontSizeSm,
                                          color: theme.colors.textSecondary,
                                        }}
                                      >
                                        {binding.field_name}
                                      </Typography.Text>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                  </div>
                </div>
              )}

              {/* Management Operations */}
              <div>
                <Typography.Title level={4} css={{ margin: 0, marginBottom: theme.spacing.md }}>
                  <FormattedMessage
                    defaultMessage="Management"
                    description="Route detail drawer > management section title"
                  />
                </Typography.Title>

                {/* Button Group */}
                <div css={{ display: 'flex', gap: theme.spacing.sm, marginBottom: theme.spacing.md }}>
                  <Button
                    componentId="mlflow.routes.detail_drawer.edit_details_toggle"
                    icon={<PencilIcon />}
                    type={selectedOperation === 'editDetails' ? 'primary' : undefined}
                    onClick={() => setSelectedOperation(selectedOperation === 'editDetails' ? null : 'editDetails')}
                  >
                    <FormattedMessage defaultMessage="Edit Details" description="Edit details button" />
                  </Button>
                  <Button
                    componentId="mlflow.routes.detail_drawer.change_secret_toggle"
                    icon={<PencilIcon />}
                    type={selectedOperation === 'changeSecret' ? 'primary' : undefined}
                    onClick={() => setSelectedOperation(selectedOperation === 'changeSecret' ? null : 'changeSecret')}
                  >
                    <FormattedMessage defaultMessage="Change Secret" description="Change secret button" />
                  </Button>
                  <Button
                    componentId="mlflow.routes.detail_drawer.delete_route_toggle"
                    icon={<TrashIcon />}
                    danger
                    type={selectedOperation === 'deleteRoute' ? 'primary' : undefined}
                    onClick={() => setSelectedOperation(selectedOperation === 'deleteRoute' ? null : 'deleteRoute')}
                  >
                    <FormattedMessage defaultMessage="Delete Route" description="Delete route button" />
                  </Button>
                </div>

                {/* Edit Details Content */}
                {selectedOperation === 'editDetails' && (
                  <div
                    css={{
                      padding: theme.spacing.md,
                      borderRadius: theme.borders.borderRadiusMd,
                      border: `2px solid ${theme.colors.border}`,
                      backgroundColor: theme.colors.backgroundPrimary,
                    }}
                  >
                    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                      {/* Description */}
                      <div>
                        <FormUI.Label htmlFor="route-description-input">
                          <FormattedMessage defaultMessage="Description" description="Route description label" />
                        </FormUI.Label>
                        <Input
                          componentId="mlflow.routes.detail_drawer.description_input"
                          id="route-description-input"
                          value={editDescription}
                          onChange={(e) => setEditDescription(e.target.value)}
                          placeholder={intl.formatMessage({
                            defaultMessage: 'Add a description for this route',
                            description: 'Route description placeholder',
                          })}
                        />
                      </div>

                      {/* Tags */}
                      <div>
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
                      </div>

                      {/* Save button */}
                      <div css={{ display: 'flex', gap: theme.spacing.sm, justifyContent: 'flex-end', marginTop: theme.spacing.md }}>
                        <Button
                          componentId="mlflow.routes.detail_drawer.cancel_details"
                          onClick={() => setSelectedOperation(null)}
                        >
                          <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
                        </Button>
                        <Button
                          componentId="mlflow.routes.detail_drawer.save_details"
                          type="primary"
                          onClick={handleSaveDetails}
                          loading={isLoading}
                        >
                          <FormattedMessage defaultMessage="Save" description="Save button" />
                        </Button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Change Secret Content */}
                {selectedOperation === 'changeSecret' && (
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
                            <FormattedMessage defaultMessage="Use Existing Key" description="Use existing secret option" />
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
                      <div css={{ display: 'flex', gap: theme.spacing.sm, justifyContent: 'flex-end', marginTop: theme.spacing.md }}>
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
                          <FormattedMessage defaultMessage="Update Route" description="Update route button" />
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
                            defaultMessage="Route Configuration"
                            description="Route detail drawer > route configuration title"
                          />
                        </Typography.Title>
                        <Typography.Text color="secondary" size="sm">
                          <FormattedMessage
                            defaultMessage="This route will be permanently deleted along with its configuration."
                            description="Route detail drawer > delete route warning"
                          />
                        </Typography.Text>
                      </div>

                      {routeBinding && (
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
                                  defaultMessage="Secret Binding"
                                  description="Route detail drawer > secret binding label"
                                />
                              </Typography.Text>
                              <Typography.Paragraph css={{ marginTop: theme.spacing.xs, marginBottom: 0, fontWeight: 500 }}>
                                {currentSecret?.secret_name || 'Unknown'}
                              </Typography.Paragraph>
                            </div>
                            <div>
                              <Typography.Text color="secondary" size="sm">
                                <FormattedMessage
                                  defaultMessage="Environment Variable"
                                  description="Route detail drawer > environment variable label"
                                />
                              </Typography.Text>
                              <Typography.Paragraph
                                css={{
                                  marginTop: theme.spacing.xs,
                                  marginBottom: 0,
                                  fontFamily: 'monospace',
                                  fontSize: theme.typography.fontSizeSm,
                                }}
                              >
                                {routeBinding.field_name}
                              </Typography.Paragraph>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Right side: Delete confirmation */}
                    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, justifyContent: 'center' }}>
                      <div>
                        <FormUI.Label htmlFor="delete-route-confirmation-input">
                          <FormattedMessage
                            defaultMessage="To confirm deletion, type the route name:"
                            description="Route detail drawer > delete confirmation label"
                          />
                        </FormUI.Label>
                        <Input
                          componentId="mlflow.routes.detail_drawer.delete_confirmation_input"
                          id="delete-route-confirmation-input"
                          placeholder={route.name || route.route_id}
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
                        disabled={routeDeleteConfirmation !== (route.name || route.route_id)}
                        onClick={handleDelete}
                      >
                        <FormattedMessage defaultMessage="Delete Route" description="Route detail drawer > delete route button" />
                      </Button>
                      <Typography.Text color="secondary" size="sm">
                        <FormattedMessage
                          defaultMessage="This action cannot be undone. The secret will remain and can be reused."
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
